#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_test_ver1.py
Navigate to each defect candidate recorded during exploration, pause at each
location so the Jetson camera node can capture high-resolution images, then
return to the start position.

Workflow
--------
1. Load a pre-built map via map_server (launched externally).
2. Read defect_candidates.yaml produced by nav_test_ver2.py.
3. Sort candidates by greedy nearest-neighbour to reduce travel distance.
4. For each candidate:
   a. Send move_base goal.
   b. On arrival, publish a message on /inspect_trigger so the Jetson
      camera node knows to capture and classify.
   c. Wait for inspection_pause_secs, then proceed to next.
5. After all candidates visited (or list empty), return to start → DONE.

Required launches (before running this script)
-----------------------------------------------
- YDLidar
- AGV base (myagv_active.launch)
- map_server with the saved map
- AMCL localisation (myagv_navigation amcl.launch or similar)
- move_base (exploration_nav.launch)
"""

import math
import os
import yaml
import rospy
import tf
import actionlib

from geometry_msgs.msg   import Twist, PoseStamped
from std_msgs.msg        import String
from move_base_msgs.msg  import MoveBaseAction, MoveBaseGoal


def _yaw_from_quat(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y),
                      1.0 - 2.0 * (y * y + z * z))


class InspectPatrol:

    def __init__(self):
        rospy.init_node("inspect_patrol", anonymous=False)

        # ── parameters ────────────────────────────────────────────────
        self.defect_yaml = rospy.get_param(
            "~defect_yaml",
            os.path.expanduser("~/myagv_ros/src/myagv_navigation/map/defect_candidates.yaml"))
        self.goal_timeout         = float(rospy.get_param("~goal_timeout", 120.0))
        self.inspection_pause_secs = float(rospy.get_param("~inspection_pause_secs", 5.0))
        self.arrival_tolerance    = float(rospy.get_param("~arrival_tolerance", 0.3))

        # ── state ─────────────────────────────────────────────────────
        self.robot_x   = 0.0
        self.robot_y   = 0.0
        self.robot_yaw = 0.0
        self.start_x   = None
        self.start_y   = None
        self.start_yaw = None

        # ── TF ────────────────────────────────────────────────────────
        self.tf_listener = tf.TransformListener()

        # ── ROS interface ─────────────────────────────────────────────
        self.cmd_pub     = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.trigger_pub = rospy.Publisher("/inspect_trigger", String, queue_size=1)
        self.mb_client   = actionlib.SimpleActionClient("move_base", MoveBaseAction)

        rospy.loginfo("inspect_patrol: node initialised.")

    # ─────────────────────────────────── helpers ────────────────────────
    def _stop(self):
        msg = Twist()
        self.cmd_pub.publish(msg)

    def _update_pose(self):
        try:
            (trans, rot) = self.tf_listener.lookupTransform(
                "map", "base_footprint", rospy.Time(0))
            self.robot_x   = trans[0]
            self.robot_y   = trans[1]
            self.robot_yaw = _yaw_from_quat(rot[0], rot[1], rot[2], rot[3])
            if self.start_x is None:
                self.start_x   = self.robot_x
                self.start_y   = self.robot_y
                self.start_yaw = self.robot_yaw
            return True
        except (tf.LookupException, tf.ConnectivityException,
                tf.ExtrapolationException):
            return False

    def _send_goal(self, x, y, yaw):
        """Send a move_base goal and block until result or timeout."""
        half = yaw / 2.0
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id    = "map"
        goal.target_pose.header.stamp       = rospy.Time.now()
        goal.target_pose.pose.position.x    = x
        goal.target_pose.pose.position.y    = y
        goal.target_pose.pose.orientation.z = math.sin(half)
        goal.target_pose.pose.orientation.w = math.cos(half)

        self.mb_client.send_goal(goal)
        finished = self.mb_client.wait_for_result(rospy.Duration(self.goal_timeout))
        state = self.mb_client.get_state()

        if finished and state == 3:
            return True
        else:
            rospy.logwarn("inspect_patrol: goal (%.2f, %.2f) %s (state=%d)",
                          x, y,
                          "timed out" if not finished else "failed",
                          state)
            self.mb_client.cancel_goal()
            rospy.sleep(0.3)
            return False

    # ─────────────────────────────────── load candidates ────────────────
    def _load_candidates(self):
        if not os.path.isfile(self.defect_yaml):
            rospy.logerr("inspect_patrol: file not found: %s", self.defect_yaml)
            return []

        with open(self.defect_yaml, "r") as f:
            data = yaml.safe_load(f)

        if data is None or "candidates" not in data:
            rospy.logwarn("inspect_patrol: no candidates in %s", self.defect_yaml)
            return []

        candidates = []
        for c in data["candidates"]:
            candidates.append({
                "x": float(c.get("x", 0.0)),
                "y": float(c.get("y", 0.0)),
                "yaw": float(c.get("yaw", 0.0)),
            })

        rospy.loginfo("inspect_patrol: loaded %d candidate(s) from %s",
                      len(candidates), self.defect_yaml)
        return candidates

    def _sort_nearest(self, candidates):
        """Greedy nearest-neighbour ordering starting from robot position."""
        if not candidates:
            return []
        remaining = list(candidates)
        ordered   = []
        cx, cy    = self.robot_x, self.robot_y
        while remaining:
            best_i = 0
            best_d = math.hypot(remaining[0]["x"] - cx, remaining[0]["y"] - cy)
            for i in range(1, len(remaining)):
                d = math.hypot(remaining[i]["x"] - cx, remaining[i]["y"] - cy)
                if d < best_d:
                    best_d = d
                    best_i = i
            nxt = remaining.pop(best_i)
            ordered.append(nxt)
            cx, cy = nxt["x"], nxt["y"]
        return ordered

    # ─────────────────────────────────── main ──────────────────────────
    def run(self):
        # Wait for TF
        r = rospy.Rate(5.0)
        for _ in range(50):
            if rospy.is_shutdown():
                return
            if self._update_pose():
                break
            r.sleep()
        else:
            rospy.logerr("inspect_patrol: TF not available, exiting.")
            return

        rospy.loginfo("inspect_patrol: robot at (%.2f, %.2f)", self.robot_x, self.robot_y)

        # Wait for move_base
        rospy.loginfo("inspect_patrol: waiting for move_base...")
        if not self.mb_client.wait_for_server(rospy.Duration(15.0)):
            rospy.logerr("inspect_patrol: move_base not available, exiting.")
            return

        # Load and sort candidates
        candidates = self._load_candidates()
        if not candidates:
            rospy.loginfo("inspect_patrol: no candidates to inspect. Returning to start.")
        else:
            candidates = self._sort_nearest(candidates)

        # Visit each candidate
        for i, c in enumerate(candidates):
            if rospy.is_shutdown():
                break

            rospy.loginfo("inspect_patrol: [%d/%d] navigating to (%.2f, %.2f)",
                          i + 1, len(candidates), c["x"], c["y"])

            # Face toward the defect location from approach direction
            approach_yaw = math.atan2(c["y"] - self.robot_y,
                                      c["x"] - self.robot_x)
            success = self._send_goal(c["x"], c["y"], c.get("yaw", approach_yaw))

            self._update_pose()

            if success:
                rospy.loginfo("inspect_patrol: arrived at candidate %d. Triggering inspection.",
                              i + 1)
            else:
                rospy.logwarn("inspect_patrol: could not reach candidate %d. "
                              "Triggering inspection from current position.", i + 1)

            # Publish inspection trigger (Jetson camera node listens)
            trigger_msg = String()
            trigger_msg.data = yaml.dump({
                "candidate_index": i,
                "target_x": c["x"],
                "target_y": c["y"],
                "target_yaw": c.get("yaw", 0.0),
                "robot_x": self.robot_x,
                "robot_y": self.robot_y,
                "robot_yaw": self.robot_yaw,
            })
            self.trigger_pub.publish(trigger_msg)

            # Pause for camera capture
            rospy.loginfo("inspect_patrol: pausing %.1f s for inspection...",
                          self.inspection_pause_secs)
            rospy.sleep(self.inspection_pause_secs)

        # Return to start
        if self.start_x is not None and not rospy.is_shutdown():
            rospy.loginfo("inspect_patrol: returning to start (%.2f, %.2f)",
                          self.start_x, self.start_y)
            success = self._send_goal(self.start_x, self.start_y, self.start_yaw)
            if success:
                rospy.loginfo("inspect_patrol: returned to start. DONE.")
            else:
                rospy.logwarn("inspect_patrol: failed to return to start. DONE.")

        self._stop()
        rospy.loginfo("inspect_patrol: inspection patrol complete.")


if __name__ == "__main__":
    try:
        InspectPatrol().run()
    except rospy.ROSInterruptException:
        pass
    