#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_test_no_jetson.py
Corridor inspection patrol using pure move_base navigation (no Jetson, no
defect detection).  move_base's global planner (Dijkstra/A* on the costmap)
handles all path planning and naturally follows curved corridors without any
custom skeleton extraction.

Two modes
---------
interactive (default) :
    Set '2D Pose Estimate' in RViz to localise the robot, then send targets
    one by one with '2D Nav Goal'.  The robot navigates to each target in
    turn.  Ctrl-C to stop.

waypoint :
    Set ~interactive false and provide ~waypoint_yaml pointing to a YAML
    file with a 'waypoints' list ({x, y, yaw}).  The robot visits each
    waypoint in order then optionally returns to start.

Recovery strategy
-----------------
On goal failure the costmaps are cleared and the goal is retried up to
~max_retries times before skipping to the next waypoint.

Required launches (before running this script)
----------------------------------------------
- YDLidar
- AGV base       : roslaunch myagv_odometry  myagv_active.launch
- Navigation     : roslaunch myagv_navigation navigation_active.launch
                   (includes map_server, AMCL, move_base)

Parameters (all optional)
--------------------------
~interactive     bool   True   — RViz 2D Nav Goal mode
~waypoint_yaml   str    ""     — path to waypoints YAML (waypoint mode only)
~return_to_start bool   True   — return to start after all waypoints
~goal_timeout    float  120.0  — seconds before a goal is considered failed
~pause_secs      float  2.0    — pause at each waypoint after arrival
~max_retries     int    2      — extra attempts on goal failure
~arrival_dist    float  0.3    — accept-as-arrived distance [m]
"""

import math
import os
import yaml
import rospy
import actionlib

from geometry_msgs.msg  import PoseWithCovarianceStamped, PoseStamped, Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
from std_srvs.srv       import Empty


def _yaw_to_quat(yaw):
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class InspectNoJetson:

    def __init__(self):
        rospy.init_node("inspect_no_jetson", anonymous=False)

        # ── parameters ────────────────────────────────────────────────────
        self.interactive     = bool(rospy.get_param("~interactive",     True))
        self.waypoint_yaml   = str(rospy.get_param("~waypoint_yaml",    ""))
        self.return_to_start = bool(rospy.get_param("~return_to_start", True))
        self.goal_timeout    = float(rospy.get_param("~goal_timeout",   120.0))
        self.pause_secs      = float(rospy.get_param("~pause_secs",       2.0))
        self.max_retries     = int(rospy.get_param("~max_retries",           2))
        self.arrival_dist    = float(rospy.get_param("~arrival_dist",      0.3))

        # ── state ─────────────────────────────────────────────────────────
        self.current_pose = None   # (x, y, yaw)
        self.start_pose   = None
        self.pending_goal = None   # written by _nav_goal_cb

        # ── ROS interfaces ────────────────────────────────────────────────
        rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped,
                         self._amcl_cb, queue_size=1)
        rospy.Subscriber("/move_base_simple/goal", PoseStamped,
                         self._nav_goal_cb, queue_size=1)
        self.cmd_pub  = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.mb       = actionlib.SimpleActionClient("move_base", MoveBaseAction)

        self._clear_srv = None  # lazy-initialised

    # ──────────────────────────────────────── callbacks ───────────────────
    def _amcl_cb(self, msg):
        p = msg.pose.pose
        q = p.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.current_pose = (p.position.x, p.position.y, yaw)
        if self.start_pose is None:
            self.start_pose = self.current_pose
            rospy.loginfo("inspect_no_jetson: start pose recorded "
                          "(%.3f, %.3f, %.1f deg)",
                          self.start_pose[0], self.start_pose[1],
                          math.degrees(self.start_pose[2]))

    def _nav_goal_cb(self, msg):
        p = msg.pose
        q = p.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.pending_goal = (p.position.x, p.position.y, yaw)
        rospy.loginfo("inspect_no_jetson: RViz goal received "
                      "(%.3f, %.3f, %.1f deg)",
                      self.pending_goal[0], self.pending_goal[1],
                      math.degrees(self.pending_goal[2]))

    # ──────────────────────────────────────── helpers ─────────────────────
    def _stop(self):
        self.cmd_pub.publish(Twist())

    def _clear_costmaps(self):
        if self._clear_srv is None:
            try:
                rospy.wait_for_service("/move_base/clear_costmaps", timeout=2.0)
                self._clear_srv = rospy.ServiceProxy(
                    "/move_base/clear_costmaps", Empty)
            except Exception:
                rospy.logwarn("inspect_no_jetson: clear_costmaps service unavailable.")
                return
        try:
            self._clear_srv()
            rospy.loginfo("inspect_no_jetson: costmaps cleared.")
        except Exception as e:
            rospy.logwarn("inspect_no_jetson: clear_costmaps call failed: %s", e)

    def _send_goal_once(self, x, y, yaw):
        """Send one move_base goal; block until result or timeout."""
        qz, qw = _yaw_to_quat(yaw)
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id    = "map"
        goal.target_pose.header.stamp       = rospy.Time.now()
        goal.target_pose.pose.position.x    = x
        goal.target_pose.pose.position.y    = y
        goal.target_pose.pose.orientation.z = qz
        goal.target_pose.pose.orientation.w = qw

        self.mb.send_goal(goal)
        finished = self.mb.wait_for_result(rospy.Duration(self.goal_timeout))
        if not finished:
            self.mb.cancel_goal()
            rospy.logwarn("inspect_no_jetson: goal timed out after %.0f s.",
                          self.goal_timeout)
            return GoalStatus.ABORTED
        return self.mb.get_state()

    def _navigate_to(self, x, y, yaw, label=""):
        """
        Navigate to (x, y, yaw) in map frame.
        On failure: clear costmaps and retry up to max_retries times.
        Returns True if arrived (or close enough), False otherwise.
        """
        attempts = self.max_retries + 1
        for attempt in range(1, attempts + 1):
            if rospy.is_shutdown():
                return False

            rospy.loginfo("inspect_no_jetson: [%s] -> (%.3f, %.3f) "
                          "attempt %d/%d", label, x, y, attempt, attempts)

            state = self._send_goal_once(x, y, yaw)

            if state == GoalStatus.SUCCEEDED:
                rospy.loginfo("inspect_no_jetson: [%s] reached (%.3f, %.3f).",
                              label, x, y)
                return True

            rospy.logwarn("inspect_no_jetson: [%s] goal state=%d.", label, state)

            if attempt < attempts:
                rospy.loginfo("inspect_no_jetson: clearing costmaps and retrying...")
                self._clear_costmaps()
                rospy.sleep(1.5)

        # Tolerate close-enough arrival even after goal failure
        if self.current_pose is not None:
            dist = math.hypot(self.current_pose[0] - x, self.current_pose[1] - y)
            if dist < self.arrival_dist:
                rospy.loginfo("inspect_no_jetson: [%s] within %.2f m "
                              "(tolerance %.2f m). Continuing.", label, dist,
                              self.arrival_dist)
                return True

        rospy.logwarn("inspect_no_jetson: [%s] giving up.", label)
        return False

    # ──────────────────────────────────────── waypoint loading ────────────
    def _load_waypoints(self):
        path = self.waypoint_yaml
        if not path or not os.path.isfile(path):
            rospy.logwarn("inspect_no_jetson: waypoint_yaml not found: '%s'", path)
            return []
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if not data or "waypoints" not in data:
            rospy.logwarn("inspect_no_jetson: no 'waypoints' key in %s", path)
            return []
        pts = []
        for wp in data["waypoints"]:
            pts.append((float(wp.get("x",   0.0)),
                        float(wp.get("y",   0.0)),
                        float(wp.get("yaw", 0.0))))
        rospy.loginfo("inspect_no_jetson: loaded %d waypoint(s) from %s",
                      len(pts), path)
        return pts

    # ──────────────────────────────────────── run modes ───────────────────
    def _run_interactive(self):
        rospy.loginfo("=" * 58)
        rospy.loginfo("inspect_no_jetson: INTERACTIVE MODE")
        rospy.loginfo("  -> In RViz: use '2D Nav Goal' to set targets.")
        rospy.loginfo("  -> Ctrl-C to stop.")
        rospy.loginfo("=" * 58)

        rate = rospy.Rate(5.0)
        while not rospy.is_shutdown():
            if self.pending_goal is None:
                rate.sleep()
                continue

            x, y, yaw     = self.pending_goal
            self.pending_goal = None

            self._navigate_to(x, y, yaw, label="GOAL")
            rospy.sleep(self.pause_secs)

        self._stop()

    def _run_waypoint(self):
        waypoints = self._load_waypoints()
        if not waypoints:
            rospy.loginfo("inspect_no_jetson: no waypoints — falling back to "
                          "interactive mode.")
            self._run_interactive()
            return

        rospy.loginfo("=" * 58)
        rospy.loginfo("inspect_no_jetson: WAYPOINT MODE (%d points)",
                      len(waypoints))
        rospy.loginfo("=" * 58)

        for i, (x, y, yaw) in enumerate(waypoints):
            if rospy.is_shutdown():
                break
            label = "WP%d" % (i + 1)
            ok = self._navigate_to(x, y, yaw, label=label)
            if ok:
                rospy.loginfo("inspect_no_jetson: [%s] pausing %.1f s...",
                              label, self.pause_secs)
            else:
                rospy.logwarn("inspect_no_jetson: [%s] skipped after failure.",
                              label)
            rospy.sleep(self.pause_secs)

        if self.return_to_start and self.start_pose is not None \
                and not rospy.is_shutdown():
            sx, sy, syaw = self.start_pose
            rospy.loginfo("inspect_no_jetson: returning to start "
                          "(%.3f, %.3f).", sx, sy)
            self._navigate_to(sx, sy, syaw, label="START")

        self._stop()
        rospy.loginfo("inspect_no_jetson: patrol complete.")

    # ──────────────────────────────────────── entry point ─────────────────
    def run(self):
        # Wait for AMCL initial pose
        rospy.loginfo("=" * 58)
        rospy.loginfo("inspect_no_jetson: waiting for AMCL pose.")
        rospy.loginfo("  -> In RViz: use '2D Pose Estimate' to set initial pose.")
        rospy.loginfo("=" * 58)
        rate = rospy.Rate(5.0)
        while not rospy.is_shutdown() and self.current_pose is None:
            rate.sleep()
        if rospy.is_shutdown():
            return

        # Wait for move_base
        rospy.loginfo("inspect_no_jetson: waiting for move_base action server...")
        if not self.mb.wait_for_server(rospy.Duration(15.0)):
            rospy.logerr("inspect_no_jetson: move_base not available. Exiting.")
            return
        rospy.loginfo("inspect_no_jetson: move_base ready.")

        if self.interactive:
            self._run_interactive()
        else:
            self._run_waypoint()


if __name__ == "__main__":
    try:
        InspectNoJetson().run()
    except rospy.ROSInterruptException:
        pass