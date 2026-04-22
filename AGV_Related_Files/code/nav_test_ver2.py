#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nav_test_ver2.py
Autonomous corridor exploration: DT-ridgeline centering + frontier-based
navigation + return-to-origin.

Motion-control logic is identical to nav_test_ver1.py (bend detection,
EMA front smoothing, wheel-space differential limits, per-wheel rate limiter,
persistent-split watchdog, wall recovery before move_base).

ver2-exclusive additions
------------------------
- Subscribes to /defect_candidates (geometry_msgs/PoseStamped) published by
  the Jetson detection node; accumulates all candidate poses.
- Saves candidates to a YAML file (defect_save_path) when exploration ends
  (RETURN start or DONE state).

States
------
FOLLOW       : Reactive corridor centering via local DT ridgeline (lidar).
SEEK_PATH    : Rotate in-place to find a clear, unexplored heading.
NAV_FRONTIER : move_base navigates to the highest-scoring frontier cluster.
RETURN       : move_base goal back to the recorded start position.
DONE         : Publish zero velocity and idle.
"""

import math
import os
import yaml
import numpy as np
import rospy
import tf
import actionlib
import scipy.ndimage as ndi
from scipy.ndimage import binary_dilation, label as ndi_label

from sensor_msgs.msg    import LaserScan
from nav_msgs.msg       import OccupancyGrid
from geometry_msgs.msg  import Twist, PoseStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal


# ═══════════════════════════════════════════════════════ helpers ══════════

def _yaw_from_quat(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y),
                      1.0 - 2.0 * (y * y + z * z))


def _angle_diff(a, b):
    """Signed difference (a − b) wrapped to (−π, π]."""
    d = a - b
    while d >  math.pi: d -= 2 * math.pi
    while d < -math.pi: d += 2 * math.pi
    return d


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _norm_deg(deg):
    while deg >  180: deg -= 360
    while deg < -180: deg += 360
    return deg


# ═══════════════════════════════════════════════════════ main node ════════

class NavExplore:

    # ─────────────────────────────────────────── __init__ ────────────────
    def __init__(self):
        rospy.init_node("nav_explore", anonymous=False)

        # ── motion ────────────────────────────────────────────────────
        self.forward_speed    = float(rospy.get_param("~forward_speed",    0.08))
        self.turn_speed       = float(rospy.get_param("~turn_speed",       0.45))
        self.max_angular      = float(rospy.get_param("~max_angular",      0.35))
        self.kp_center        = float(rospy.get_param("~kp_center",        0.15))
        self.kp_heading       = float(rospy.get_param("~kp_heading",       0.10))
        self.side_error_db    = float(rospy.get_param("~side_error_deadband", 0.08))
        self.wall_clearance   = float(rospy.get_param("~wall_clearance",   0.35))
        self.max_angular_rate = float(rospy.get_param("~max_angular_rate", 0.12))
        self.control_rate     = float(rospy.get_param("~control_rate",     10.0))

        # ── distance thresholds ───────────────────────────────────────
        self.front_block_dist = float(rospy.get_param("~front_block_dist", 0.65))
        self.front_clear_dist = float(rospy.get_param("~front_clear_dist", 0.55))
        self.side_open_dist   = float(rospy.get_param("~side_open_dist",   1.50))
        self.side_wall_min    = float(rospy.get_param("~side_wall_min",    0.20))
        self.side_wall_max    = float(rospy.get_param("~side_wall_max",    2.00))
        self.min_valid_range  = float(rospy.get_param("~min_valid_range",  0.05))
        self.max_valid_range  = float(rospy.get_param("~max_valid_range",  8.00))

        # ── SEEK_PATH parameters ──────────────────────────────────────
        self.seek_trigger_count = int(  rospy.get_param("~seek_trigger_count", 10))
        self.seek_check_dist    = float(rospy.get_param("~seek_check_dist",    3.0))
        self.seek_timeout       = float(rospy.get_param("~seek_timeout",       35.0))
        self.min_seek_rotation  = float(rospy.get_param("~min_seek_rotation",  math.radians(60)))
        self.post_turn_frames   = int(  rospy.get_param("~post_turn_frames",   20))
        self.startup_grace_count= int(  rospy.get_param("~startup_grace_count",20))
        self.min_inter_turn_secs= float(rospy.get_param("~min_inter_turn_secs",2.0))

        # ── bend detection ────────────────────────────────────────────
        self.bend_side_thresh = float(rospy.get_param("~bend_side_thresh", 0.80))
        self.bend_steer_gain  = float(rospy.get_param("~bend_steer_gain",  0.40))

        # ── front distance EMA smoothing ──────────────────────────────
        self.front_ema_alpha = float(rospy.get_param("~front_ema_alpha", 0.35))

        # ── speed-angular coupling ────────────────────────────────────
        self.ang_speed_reduce_thresh = float(rospy.get_param("~ang_speed_reduce_thresh", 0.05))

        # ── wheel kinematics + limits ─────────────────────────────────
        self.wheel_k         = float(rospy.get_param("~wheel_k",         0.12))
        self.max_wheel_speed = float(rospy.get_param("~max_wheel_speed", 0.08))
        self.max_wheel_ratio = float(rospy.get_param("~max_wheel_ratio", 1.2))
        self.max_wheel_accel = float(rospy.get_param("~max_wheel_accel", 0.002))
        self.last_wL = 0.0
        self.last_wR = 0.0

        # ── persistent wheel-split detection ──────────────────────────
        self.split_detect_frames = int(  rospy.get_param("~split_detect_frames", 5))
        self.split_delta_eps     = float(rospy.get_param("~split_delta_eps",     0.0015))
        self.split_speed_floor   = float(rospy.get_param("~split_speed_floor",   0.006))
        self.split_cooldown_secs = float(rospy.get_param("~split_cooldown_secs", 1.5))
        self.split_persist_count = 0
        self.split_last_sign     = 0
        self.split_last_trigger_time = -999.0
        self.split_state_label   = "IDLE"

        # ── wall recovery before move_base ────────────────────────────
        self.recovery_clearance  = float(rospy.get_param("~recovery_clearance",  0.35))
        self.recovery_back_speed = float(rospy.get_param("~recovery_back_speed", 0.05))
        self.recovery_max_secs   = float(rospy.get_param("~recovery_max_secs",   5.0))

        # ── frontier / navigation ─────────────────────────────────────
        self.min_frontier_size      = int(  rospy.get_param("~min_frontier_size",      8))
        self.min_frontier_dist      = float(rospy.get_param("~min_frontier_dist",      0.3))
        self.failed_frontier_radius = float(rospy.get_param("~failed_frontier_radius", 0.8))
        self.nav_goal_timeout       = float(rospy.get_param("~nav_goal_timeout",       80.0))
        self.goal_timeout           = float(rospy.get_param("~goal_timeout",           80.0))
        self.min_explore_secs       = float(rospy.get_param("~min_explore_secs",       20.0))

        # ── global timeout + no-progress watchdog ─────────────────────
        self.max_explore_secs    = float(rospy.get_param("~max_explore_secs",    100.0))
        self.no_progress_timeout = float(rospy.get_param("~no_progress_timeout", 30.0))
        self.no_progress_radius  = float(rospy.get_param("~no_progress_radius",  0.5))

        # ── ver2: defect candidate output path ────────────────────────
        self.defect_save_path = rospy.get_param(
            "~defect_save_path",
            os.path.expanduser(
                "~/myagv_ros/src/myagv_navigation/map/defect_candidates.yaml"))

        # ── runtime state ─────────────────────────────────────────────
        self.state = "FOLLOW"
        self.scan  = None

        self.map_array = None
        self.map_info  = None

        self.angle_offset = None

        self.robot_x   = 0.0
        self.robot_y   = 0.0
        self.robot_yaw = 0.0

        self.start_x   = None
        self.start_y   = None
        self.start_yaw = None
        self._explore_start_time = None

        # ── smoothing ─────────────────────────────────────────────────
        self.last_angular     = 0.0
        self.last_speed       = 0.0
        self.last_front_valid = self.front_block_dist
        self.last_width       = 1.2

        # ── EMA smoothed front distance ───────────────────────────────
        self._front_ema = None

        # ── FOLLOW counters ───────────────────────────────────────────
        self.blocked_frames_count = 0
        self.last_turn_time       = -100.0
        self.post_turn_counter    = 0

        # ── SEEK_PATH state ───────────────────────────────────────────
        self.seek_total_rot   = 0.0
        self.seek_last_yaw    = None
        self.seek_start_time  = 0.0
        self.seek_turn_dir    = 1.0
        self.seek_frontiers   = []

        # ── NAV_FRONTIER state ────────────────────────────────────────
        self.frontier_targets    = []
        self._nav_frontier_idx   = 0
        self._nav_frontier_start = 0.0
        self._failed_frontiers   = []

        # ── RETURN state ──────────────────────────────────────────────
        self._return_start_time = 0.0

        # ── frontier cache ────────────────────────────────────────────
        self._frontier_cache_time = -999.0
        self._frontier_cache      = []

        # ── no-progress watchdog state ────────────────────────────────
        self._progress_x    = 0.0
        self._progress_y    = 0.0
        self._progress_time = 0.0

        # ── ver2: defect candidates list ──────────────────────────────
        self._defect_candidates = []
        self._defects_saved     = False

        # ── TF ────────────────────────────────────────────────────────
        self.tf_listener = tf.TransformListener()

        # ── ROS interface ─────────────────────────────────────────────
        rospy.Subscriber("/scan", LaserScan,     self._scan_cb, queue_size=1)
        rospy.Subscriber("/map",  OccupancyGrid, self._map_cb,  queue_size=1)
        rospy.Subscriber("/defect_candidates", PoseStamped,
                         self._defect_cb, queue_size=10)
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        self.mb_client = actionlib.SimpleActionClient("move_base", MoveBaseAction)

        rospy.loginfo("nav_explore ver2: node initialised.")

    # ─────────────────────────────────────────── callbacks ───────────────
    def _scan_cb(self, msg):
        self.scan = msg

    def _map_cb(self, msg):
        self.map_info  = msg.info
        self.map_array = np.array(msg.data, dtype=np.int8).reshape(
                             msg.info.height, msg.info.width)

    def _defect_cb(self, msg):
        """Receive a defect candidate from the Jetson detection node."""
        entry = {
            "x":     round(msg.pose.position.x, 3),
            "y":     round(msg.pose.position.y, 3),
            "z":     round(msg.pose.position.z, 3),
            "yaw":   round(_yaw_from_quat(
                msg.pose.orientation.x, msg.pose.orientation.y,
                msg.pose.orientation.z, msg.pose.orientation.w), 4),
            "stamp": msg.header.stamp.to_sec(),
        }
        self._defect_candidates.append(entry)
        rospy.loginfo("nav_explore: defect candidate #%d at (%.2f, %.2f)",
                      len(self._defect_candidates), entry["x"], entry["y"])

    # ─────────────────────────────────────────── TF pose ─────────────────
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
                self._explore_start_time = rospy.get_time()
                self._progress_x    = self.robot_x
                self._progress_y    = self.robot_y
                self._progress_time = rospy.get_time()
                rospy.loginfo("nav_explore: start pose recorded (%.2f, %.2f)",
                              self.start_x, self.start_y)
            return True
        except (tf.LookupException, tf.ConnectivityException,
                tf.ExtrapolationException):
            return False

    # ─────────────────────────────────────────── laser helpers ───────────
    def _resolve_angle_offset(self):
        if self.angle_offset is not None:
            return True
        if self.scan is None:
            return False
        try:
            frame = self.scan.header.frame_id
            (_, rot) = self.tf_listener.lookupTransform(
                "base_footprint", frame, rospy.Time(0))
            self.angle_offset = math.degrees(2.0 * math.atan2(rot[2], rot[3]))
            rospy.loginfo("nav_explore: laser offset = %.1f deg", self.angle_offset)
            return True
        except (tf.LookupException, tf.ConnectivityException,
                tf.ExtrapolationException):
            return False

    def _angle_to_idx(self, scan, deg):
        rad = math.radians(deg)
        idx = int(round((rad - scan.angle_min) / scan.angle_increment))
        return _clamp(idx, 0, len(scan.ranges) - 1)

    def _sector_dist(self, scan, deg_center, deg_width,
                     inf_fallback=None, use_min=True):
        vals = []
        d = deg_center - deg_width / 2.0
        while d <= deg_center + deg_width / 2.0:
            r = scan.ranges[self._angle_to_idx(scan, _norm_deg(d))]
            if math.isfinite(r) and self.min_valid_range < r < self.max_valid_range:
                vals.append(r)
            elif inf_fallback is not None:
                vals.append(inf_fallback)
            d += 1.0
        if not vals:
            return None
        vals.sort()
        return vals[0] if use_min else vals[len(vals) // 2]

    def _get_distances(self):
        if self.scan is None or not self._resolve_angle_offset():
            return None
        o = self.angle_offset
        s = self.scan
        M = self.max_valid_range
        return {
            "front":        self._sector_dist(s,   0+o, 40, inf_fallback=None),
            "front_narrow": self._sector_dist(s,   0+o, 20, inf_fallback=None),
            "left":         self._sector_dist(s, +90+o, 30, inf_fallback=M),
            "right":        self._sector_dist(s, -90+o, 30, inf_fallback=M),
            "front_left":   self._sector_dist(s, +45+o, 25, inf_fallback=M),
            "front_right":  self._sector_dist(s, -45+o, 25, inf_fallback=M),
        }

    # ─────────────────────────────────────────── publish ─────────────────
    def _pub(self, lin, ang):
        msg = Twist()
        msg.linear.x  = lin
        msg.angular.z = ang
        self.cmd_pub.publish(msg)

    def _stop(self):
        self._pub(0.0, 0.0)
        self.last_wL = 0.0
        self.last_wR = 0.0

    def _reset_speed_planner(self):
        """Reset wheel / speed / smoothing state after an emergency pause."""
        self.last_speed = 0.0
        self.last_angular = 0.0
        self.last_wL = 0.0
        self.last_wR = 0.0
        self.blocked_frames_count = 0
        self._front_ema = None

    def _handle_persistent_wheel_split(self, dist, wL, wR):
        """
        Detect persistent drift: one wheel accelerating while the other
        decelerates for N consecutive frames.  Forces stop + SEEK_PATH replan.
        """
        now = rospy.get_time()
        if (now - self.split_last_trigger_time) < self.split_cooldown_secs:
            cooldown_left = max(0.0, self.split_cooldown_secs -
                                (now - self.split_last_trigger_time))
            self.split_state_label = "COOLDOWN %.2fs" % cooldown_left
            return False

        dL = wL - self.last_wL
        dR = wR - self.last_wR
        eps = self.split_delta_eps

        split_sign  = 0
        split_label = "IDLE"
        if dL > eps and dR < -eps:
            split_sign  = +1
            split_label = "LEFT_UP_RIGHT_DOWN"
        elif dL < -eps and dR > eps:
            split_sign  = -1
            split_label = "LEFT_DOWN_RIGHT_UP"

        moving_enough = max(abs(wL), abs(wR)) >= self.split_speed_floor
        unbalanced    = abs(wL - wR) >= (2.0 * eps)

        if split_sign != 0 and moving_enough and unbalanced:
            if split_sign == self.split_last_sign:
                self.split_persist_count += 1
            else:
                self.split_persist_count = 1
                self.split_last_sign = split_sign
            self.split_state_label = "%s %d/%d" % (
                split_label, self.split_persist_count, self.split_detect_frames)
            rospy.loginfo(
                "split=%s count=%d/%d dL=%.4f dR=%.4f wL=%.3f wR=%.3f",
                split_label, self.split_persist_count, self.split_detect_frames,
                dL, dR, wL, wR)
        else:
            if split_sign == 0:
                self.split_state_label = (
                    "IDLE low_speed" if not moving_enough else
                    "IDLE balanced"  if not unbalanced    else "IDLE")
            else:
                self.split_state_label = "%s rejected" % split_label
            self.split_persist_count = 0
            self.split_last_sign = 0
            return False

        if self.split_persist_count < self.split_detect_frames:
            return False

        left  = dist["left"]  if dist and dist.get("left")  is not None else 0.0
        right = dist["right"] if dist and dist.get("right") is not None else 0.0
        prefer_left = (left >= right)

        rospy.logwarn(
            "nav_explore: persistent wheel split detected for %d frames "
            "(mode=%s | wL=%.3f dL=%.4f | wR=%.3f dR=%.4f) -> stop and replan",
            self.split_persist_count, split_label, wL, dL, wR, dR)

        self._stop()
        rospy.sleep(0.15)
        self._reset_speed_planner()
        self.split_persist_count = 0
        self.split_last_sign     = 0
        self.split_state_label   = "TRIGGERED -> SEEK_PATH"
        self.split_last_trigger_time = now
        self._enter_seek_path(prefer_left)
        return True

    def _clamp_cmd(self, lin, ang):
        """
        Wheel-speed normalisation (move_base style).
        Stage 1: proportional max-speed scaling (preserves curvature).
        Stage 2: wheel ratio limit (|wfast/wslow| <= max_wheel_ratio).
        Stage 3: absolute angular ceiling.
        """
        K = self.wheel_k

        ws = max(abs(lin - K * ang), abs(lin + K * ang))
        if ws > self.max_wheel_speed and ws > 1e-6:
            scale = self.max_wheel_speed / ws
            lin  *= scale
            ang  *= scale

        if lin > 1e-3:
            R = self.max_wheel_ratio
            ang_ratio_limit = lin * (R - 1.0) / (K * (R + 1.0))
            ang = _clamp(ang, -ang_ratio_limit, ang_ratio_limit)

        ang = _clamp(ang, -self.max_angular, self.max_angular)
        return lin, ang

    # ─────────────────────────── map queries ─────────────────────────────
    def _has_unknown_ahead(self, yaw):
        if self.map_array is None or self.map_info is None:
            return True

        res = self.map_info.resolution
        ox  = self.map_info.origin.position.x
        oy  = self.map_info.origin.position.y
        H, W = self.map_array.shape

        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        d = self.front_block_dist
        while d < self.seek_check_dist:
            px = self.robot_x + d * cos_y
            py = self.robot_y + d * sin_y
            mx = int((px - ox) / res)
            my = int((py - oy) / res)
            if not (0 <= mx < W and 0 <= my < H):
                break
            cell = self.map_array[my, mx]
            if cell > 50:
                return False
            if cell < 0:
                return True
            d += res
        return False

    def _find_frontiers(self):
        if self.map_array is None or self.map_info is None:
            return []

        grid = self.map_array
        res  = self.map_info.resolution
        ox   = self.map_info.origin.position.x
        oy   = self.map_info.origin.position.y

        free    = (grid == 0)
        unknown = (grid < 0)

        dt          = ndi.distance_transform_edt(free) * res
        unk_dilated = binary_dilation(unknown, structure=np.ones((3, 3), dtype=bool))
        frontier    = free & unk_dilated

        labeled, n = ndi_label(frontier)
        if n == 0:
            return []

        results = []
        for lbl in range(1, n + 1):
            region = (labeled == lbl)
            if int(region.sum()) < self.min_frontier_size:
                continue
            ys, xs = np.where(region)
            cy = float(ys.mean())
            cx = float(xs.mean())
            wx = ox + (cx + 0.5) * res
            wy = oy + (cy + 0.5) * res

            robot_dist = math.hypot(wx - self.robot_x, wy - self.robot_y)
            if robot_dist < self.min_frontier_dist:
                continue
            if any(math.hypot(wx - fx, wy - fy) < self.failed_frontier_radius
                   for (fx, fy) in self._failed_frontiers):
                continue

            dt_val = float(dt[int(round(cy)), int(round(cx))])
            area   = int(region.sum()) * res * res
            results.append((area * dt_val / (1.0 + robot_dist), wx, wy))

        results.sort(reverse=True)
        return [(wx, wy) for (_, wx, wy) in results]

    def _find_frontiers_cached(self, max_age=1.0):
        now = rospy.get_time()
        if (now - self._frontier_cache_time) > max_age:
            self._frontier_cache      = self._find_frontiers()
            self._frontier_cache_time = now
        return self._frontier_cache

    # ─────────────────────────── global checks ───────────────────────────
    def _check_global_timeout(self):
        if self._explore_start_time is None:
            return False

        now     = rospy.get_time()
        elapsed = now - self._explore_start_time

        if elapsed >= self.max_explore_secs:
            rospy.logwarn("nav_explore: global timeout (%.0f s) -> RETURN", elapsed)
            return True

        if self.state == "FOLLOW":
            moved = math.hypot(self.robot_x - self._progress_x,
                               self.robot_y - self._progress_y)
            if moved > self.no_progress_radius:
                self._progress_x    = self.robot_x
                self._progress_y    = self.robot_y
                self._progress_time = now

            if (now - self._progress_time) >= self.no_progress_timeout:
                rospy.logwarn("nav_explore: no progress for %.0f s -> RETURN",
                              now - self._progress_time)
                return True
        else:
            self._progress_time = now

        return False

    def _force_return(self):
        if self.state == "NAV_FRONTIER":
            self.mb_client.cancel_goal()
            rospy.sleep(0.2)
        self._stop()
        self._enter_return()

    # ─────────────────────────── ver2: save defects ───────────────────────
    def _save_defects(self):
        """Write collected defect candidates to YAML (idempotent)."""
        if self._defects_saved:
            return
        self._defects_saved = True

        save_dir = os.path.dirname(self.defect_save_path)
        if save_dir and not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        data = {
            "start_pose": {
                "x":   round(self.start_x,   3) if self.start_x   is not None else 0.0,
                "y":   round(self.start_y,   3) if self.start_y   is not None else 0.0,
                "yaw": round(self.start_yaw, 4) if self.start_yaw is not None else 0.0,
            },
            "candidates": self._defect_candidates,
        }

        with open(self.defect_save_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

        rospy.loginfo("nav_explore: saved %d defect candidate(s) to %s",
                      len(self._defect_candidates), self.defect_save_path)

    # ─────────────────────────── FOLLOW state ────────────────────────────
    def _follow_step(self, dist):
        """
        Reactive DT-ridgeline centering.
        Logic is identical to nav_test_ver1.py.
        """
        left        = dist["left"]
        right       = dist["right"]
        front_left  = dist["front_left"]
        front_right = dist["front_right"]
        front_raw   = dist["front"]

        left_real  = (left  is not None and left  < self.side_open_dist)
        right_real = (right is not None and right < self.side_open_dist)

        if front_raw is None:
            decay = (self.forward_speed / self.control_rate) * 2.0
            self.last_front_valid = max(self.last_front_valid - decay,
                                        self.min_valid_range)
            front = self.last_front_valid
        else:
            front = front_raw
            self.last_front_valid = front

        front_narrow = dist.get("front_narrow", front_raw)
        if front_narrow is None:
            front_narrow = front

        is_blocked = (front_narrow < self.front_block_dist)

        if self._front_ema is None:
            self._front_ema = front_narrow
        else:
            alpha = self.front_ema_alpha
            self._front_ema = alpha * front_narrow + (1.0 - alpha) * self._front_ema
        front_smooth = min(self._front_ema, front_narrow)

        rospy.loginfo("blk=%d fn=%.2f fs=%.2f spd=%.3f ang=%.2f blocked=%s",
                      self.blocked_frames_count, front_narrow,
                      front_smooth, self.last_speed, self.last_angular,
                      str(is_blocked))

        if self.startup_grace_count > 0:
            self.startup_grace_count -= 1

        fl_open = (front_left  is not None and front_left  > self.bend_side_thresh)
        fr_open = (front_right is not None and front_right > self.bend_side_thresh)
        is_bend = is_blocked and (fl_open or fr_open)

        if is_bend:
            is_blocked = False
            rospy.loginfo_throttle(1.0,
                "nav_explore: BEND detected (fl=%.2f fr=%.2f) - centering handles turn",
                front_left if front_left else 0,
                front_right if front_right else 0)

        if is_blocked and self.startup_grace_count <= 0:
            self.blocked_frames_count += 1
            if self.blocked_frames_count >= self.seek_trigger_count:
                rospy.loginfo("ENTER SEEK_PATH triggered! blocked_count=%d front=%.2f",
                              self.blocked_frames_count, front)
                l = left  if left  is not None else 0.0
                r = right if right is not None else 0.0
                prefer_left = (l >= r)
                self.blocked_frames_count = 0
                rospy.loginfo("nav_explore: FOLLOW -> SEEK_PATH (blocked %d frames)",
                              self.seek_trigger_count)
                self._enter_seek_path(prefer_left)
                return False
            self._pub(0.0, 0.0)
            self.last_speed   = 0.0
            self.last_angular = 0.0
            self.last_wL = 0.0
            self.last_wR = 0.0
            return True
        else:
            if front_narrow > self.front_clear_dist and self.last_speed > 0:
                self.blocked_frames_count = 0

        if left_real and right_real:
            self.last_width = left + right
            raw_error = left - right
        elif left_real:
            raw_error = (left - self.last_width / 2.0) if self.post_turn_counter > 0 \
                        else min(0.0, left - self.wall_clearance)
        elif right_real:
            raw_error = (self.last_width / 2.0 - right) if self.post_turn_counter > 0 \
                        else max(0.0, self.wall_clearance - right)
        else:
            raw_error = 0.0

        error = 0.0 if abs(raw_error) < self.side_error_db else raw_error

        heading_err = 0.0
        if self.post_turn_counter > 0:
            self.post_turn_counter -= 1
            fl_close = (front_left  is not None and front_left  < self.side_open_dist)
            fr_close = (front_right is not None and front_right < self.side_open_dist)
            if fl_close and fr_close:
                heading_err = front_left - front_right

        angular = _clamp(self.kp_center * error + self.kp_heading * heading_err,
                         -self.max_angular, self.max_angular)
        angular = _clamp(angular,
                         self.last_angular - self.max_angular_rate,
                         self.last_angular + self.max_angular_rate)
        self.last_angular = angular

        # ── speed target ──────────────────────────────────────────────
        if front_smooth < 1.0:
            ratio = (front_smooth - self.front_block_dist) / \
                    max(0.01, 1.0 - self.front_block_dist)
            ratio = max(0.15, ratio)
            target_speed = self.forward_speed * max(0.0, ratio)
        else:
            target_speed = self.forward_speed

        if self.startup_grace_count > 0:
            target_speed = min(target_speed, self.forward_speed * 0.5)

        if is_bend and front_smooth < self.front_block_dist:
            target_speed = 0.0

        abs_ang = abs(angular)
        if abs_ang > self.ang_speed_reduce_thresh:
            ang_ratio = (abs_ang - self.ang_speed_reduce_thresh) / \
                        max(0.01, self.max_angular - self.ang_speed_reduce_thresh)
            ang_ratio = min(1.0, ang_ratio)
            target_speed *= (1.0 - 0.5 * ang_ratio)

        if target_speed > self.last_speed:
            speed = min(target_speed, self.last_speed + 0.005)
        else:
            speed = max(target_speed, self.last_speed - 0.06)
        if speed < 0.003:
            speed = 0.0
        self.last_speed = speed

        speed, angular = self._clamp_cmd(speed, angular)
        self.last_angular = angular
        self.last_speed   = speed

        if speed == 0.0 and not is_bend and self.startup_grace_count <= 0:
            self.blocked_frames_count += 1
            if self.blocked_frames_count >= self.seek_trigger_count:
                rospy.loginfo("nav_explore: speed dead zone -> SEEK_PATH "
                              "(fn=%.2f fs=%.2f)", front_narrow, front_smooth)
                l = left  if left  is not None else 0.0
                r = right if right is not None else 0.0
                self.blocked_frames_count = 0
                self._enter_seek_path(l >= r)
                return False

        K  = self.wheel_k
        da = self.max_wheel_accel
        wL_t = speed - K * angular
        wR_t = speed + K * angular
        wL = _clamp(wL_t, self.last_wL - da, self.last_wL + da)
        wR = _clamp(wR_t, self.last_wR - da, self.last_wR + da)

        if self._handle_persistent_wheel_split(dist, wL, wR):
            return False

        self.last_wL = wL
        self.last_wR = wR
        speed_out   = (wL + wR) / 2.0
        angular_out = (wR - wL) / (2.0 * K)
        self.last_speed   = speed_out
        self.last_angular = angular_out
        rospy.loginfo("cmd v=%.3f w=%.3f | wL=%.3f wR=%.3f | split=%s",
                      speed_out, angular_out, wL, wR, self.split_state_label)
        self._pub(speed_out, angular_out)
        return True

    # ─────────────────────────── SEEK_PATH state ─────────────────────────
    def _enter_seek_path(self, prefer_left=True):
        self._stop()
        if self.last_front_valid < self.front_block_dist + 0.15:
            self._pub(-self.forward_speed, 0.0)
            rospy.sleep(1.0)
            self._stop()
        self.state           = "SEEK_PATH"
        self.seek_turn_dir   = 1.0 if prefer_left else -1.0
        self.seek_total_rot  = 0.0
        self.seek_last_yaw   = self.robot_yaw
        self.seek_start_time = rospy.get_time()
        self.seek_frontiers  = []
        self.last_angular    = 0.0
        self.last_speed      = 0.0
        self.last_wL         = 0.0
        self.last_wR         = 0.0
        self.split_persist_count = 0
        self.split_last_sign     = 0
        self._frontier_cache_time = -999.0
        if hasattr(self, '_seek_fallback_yaw'):
            delattr(self, '_seek_fallback_yaw')
        if hasattr(self, '_seek_fallback_front'):
            delattr(self, '_seek_fallback_front')
        rospy.loginfo("nav_explore: SEEK_PATH - rotating to find clear+unexplored direction "
                      "(%s first).", "CCW" if prefer_left else "CW")

    def _seek_path_step(self, dist):
        self._update_pose()

        if self.seek_last_yaw is not None:
            self.seek_total_rot += _angle_diff(self.robot_yaw, self.seek_last_yaw)
        self.seek_last_yaw = self.robot_yaw

        front_raw = dist["front"] if dist is not None else None
        if front_raw is not None:
            self.last_front_valid = front_raw
        front = front_raw if front_raw is not None else self.last_front_valid

        front_clear = (front is not None and front > self.front_clear_dist)
        min_rot_met = abs(self.seek_total_rot) >= self.min_seek_rotation

        if front_clear and min_rot_met:
            if self.map_array is None:
                unknown_ok = True
            else:
                unknown_ok = self._has_unknown_ahead(self.robot_yaw)

            if unknown_ok:
                rospy.loginfo("nav_explore: SEEK_PATH -> FOLLOW "
                              "(front=%.2f m, rot=%.0f deg, yaw=%.2f rad)",
                              front, math.degrees(abs(self.seek_total_rot)),
                              self.robot_yaw)
                self._stop()
                self.state              = "FOLLOW"
                self.last_turn_time     = rospy.get_time()
                self.last_angular       = 0.0
                self.last_speed         = 0.0
                self.post_turn_counter  = self.post_turn_frames
                self.blocked_frames_count = 0
                self._front_ema = None
                return

        if front_clear and min_rot_met and self.map_array is not None:
            if not hasattr(self, '_seek_fallback_yaw') or \
               front > getattr(self, '_seek_fallback_front', 0.0):
                self._seek_fallback_yaw   = self.robot_yaw
                self._seek_fallback_front = front

        for (wx, wy) in self._find_frontiers_cached(max_age=1.0):
            already = any(math.hypot(wx - ex, wy - ey) < 0.3
                          for (ex, ey) in self.seek_frontiers)
            if not already:
                self.seek_frontiers.append((wx, wy))

        full_rotation = abs(self.seek_total_rot) >= (2 * math.pi - 0.15)
        timed_out     = (rospy.get_time() - self.seek_start_time > self.seek_timeout)

        if full_rotation or timed_out:
            if self.seek_frontiers:
                rospy.loginfo("nav_explore: SEEK_PATH 360 - %d frontier(s) -> NAV_FRONTIER",
                              len(self.seek_frontiers))
                self.frontier_targets  = list(self.seek_frontiers)
                self._nav_frontier_idx = 0
                self._enter_nav_frontier()
            elif hasattr(self, '_seek_fallback_yaw'):
                rospy.loginfo("nav_explore: SEEK_PATH 360 - no unexplored dir; "
                              "aligning to best clear yaw=%.2f rad -> FOLLOW",
                              self._seek_fallback_yaw)
                self._stop()
                self.state = "FOLLOW"
                self.last_turn_time    = rospy.get_time()
                self.last_angular      = 0.0
                self.last_speed        = 0.0
                self.post_turn_counter = self.post_turn_frames
                self.blocked_frames_count = 0
                self._front_ema = None
                delattr(self, '_seek_fallback_yaw')
                delattr(self, '_seek_fallback_front')
            else:
                elapsed = (rospy.get_time() - self._explore_start_time
                           if self._explore_start_time else 0.0)
                if elapsed < self.min_explore_secs:
                    rospy.loginfo("nav_explore: SEEK_PATH - no clear direction yet "
                                  "(%.0f s elapsed) -> FOLLOW", elapsed)
                    self._stop()
                    self.state = "FOLLOW"
                    self.last_turn_time = rospy.get_time()
                    self._front_ema = None
                else:
                    rospy.loginfo("nav_explore: SEEK_PATH - fully enclosed -> RETURN")
                    self._stop()
                    self._enter_return()
            return

        target_ang = self.seek_turn_dir * self.turn_speed
        angular    = _clamp(target_ang,
                            self.last_angular - self.max_angular_rate * 3,
                            self.last_angular + self.max_angular_rate * 3)
        _, angular = self._clamp_cmd(0.0, angular)
        K  = self.wheel_k
        da = self.max_wheel_accel
        wL = _clamp(-K * angular, self.last_wL - da, self.last_wL + da)
        wR = _clamp(+K * angular, self.last_wR - da, self.last_wR + da)
        self.last_wL = wL
        self.last_wR = wR
        angular = (wR - wL) / (2.0 * K)
        self.last_angular = angular
        self._pub(0.0, angular)

    # ─────────────────────────── wall recovery ───────────────────────────
    def _recover_from_wall(self):
        """Back away from walls before sending a move_base goal."""
        rate = rospy.Rate(self.control_rate)
        start_time = rospy.get_time()

        while not rospy.is_shutdown():
            elapsed = rospy.get_time() - start_time
            if elapsed > self.recovery_max_secs:
                rospy.logwarn("nav_explore: wall recovery timeout (%.1f s)", elapsed)
                break

            self._update_pose()
            dist = self._get_distances()
            if dist is None:
                rate.sleep()
                continue

            front_d = dist["front"] if dist["front"] is not None else 999.0
            left_d  = dist["left"]  if dist["left"]  is not None else 999.0
            right_d = dist["right"] if dist["right"] is not None else 999.0
            min_d   = min(front_d, left_d, right_d)

            if min_d >= self.recovery_clearance:
                rospy.loginfo("nav_explore: wall recovery done, clearance=%.2f m "
                              "(took %.1f s)", min_d, elapsed)
                break

            back_speed = -self.recovery_back_speed
            ang = 0.0
            if left_d < self.recovery_clearance and right_d >= self.recovery_clearance:
                ang = -0.3
            elif right_d < self.recovery_clearance and left_d >= self.recovery_clearance:
                ang =  0.3
            elif left_d < right_d:
                ang = -0.2
            elif right_d < left_d:
                ang =  0.2

            self._pub(back_speed, ang)
            rospy.loginfo_throttle(1.0,
                "nav_explore: recovering... front=%.2f left=%.2f right=%.2f "
                "back=%.3f ang=%.2f", front_d, left_d, right_d, back_speed, ang)
            rate.sleep()

        self._stop()
        rospy.sleep(0.3)

    # ─────────────────────────── NAV_FRONTIER state ──────────────────────
    def _enter_nav_frontier(self):
        if self._nav_frontier_idx >= len(self.frontier_targets):
            rospy.loginfo("nav_explore: all frontier targets exhausted -> RETURN")
            self._stop()
            self._enter_return()
            return

        if not self.mb_client.wait_for_server(rospy.Duration(3.0)):
            rospy.logerr("nav_explore: move_base not available -> RETURN")
            self._stop()
            self._enter_return()
            return

        self._recover_from_wall()

        wx, wy = self.frontier_targets[self._nav_frontier_idx]
        yaw    = math.atan2(wy - self.robot_y, wx - self.robot_x)
        half   = yaw / 2.0

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id    = "map"
        goal.target_pose.header.stamp       = rospy.Time.now()
        goal.target_pose.pose.position.x    = wx
        goal.target_pose.pose.position.y    = wy
        goal.target_pose.pose.orientation.x = 0.0
        goal.target_pose.pose.orientation.y = 0.0
        goal.target_pose.pose.orientation.z = math.sin(half)
        goal.target_pose.pose.orientation.w = math.cos(half)

        self.mb_client.send_goal(goal)
        self._nav_frontier_start = rospy.get_time()
        self.state = "NAV_FRONTIER"
        rospy.loginfo("nav_explore: NAV_FRONTIER %d/%d -> (%.2f, %.2f)",
                      self._nav_frontier_idx + 1, len(self.frontier_targets), wx, wy)

    def _nav_frontier_step(self):
        mb_state = self.mb_client.get_state()
        elapsed  = rospy.get_time() - self._nav_frontier_start

        if mb_state == 3:   # SUCCEEDED
            wx, wy = self.frontier_targets[self._nav_frontier_idx]
            rospy.loginfo("nav_explore: frontier reached (%.2f, %.2f) -> FOLLOW", wx, wy)
            self.state              = "FOLLOW"
            self.last_turn_time     = rospy.get_time()
            self.last_angular       = 0.0
            self.last_speed         = 0.0
            self.post_turn_counter  = self.post_turn_frames
            self.blocked_frames_count = 0
            self._front_ema = None
            return

        goal_failed = mb_state in (2, 4, 5, 9)
        goal_timed  = elapsed > self.nav_goal_timeout

        if goal_failed or goal_timed:
            wx, wy = self.frontier_targets[self._nav_frontier_idx]
            reason = "timeout" if goal_timed else "aborted (state=%d)" % mb_state
            rospy.logwarn("nav_explore: frontier (%.2f, %.2f) %s -> trying next.",
                          wx, wy, reason)
            self.mb_client.cancel_goal()
            rospy.sleep(0.2)
            self._failed_frontiers.append((wx, wy))
            self._nav_frontier_idx += 1
            self._enter_nav_frontier()

    # ─────────────────────────── RETURN state ────────────────────────────
    def _enter_return(self):
        """Save defects then send move_base goal to start position."""
        self._save_defects()   # ver2: persist before returning

        if self.start_x is None:
            rospy.logwarn("nav_explore: no start pose recorded; stopping.")
            self.state = "DONE"
            return

        rospy.loginfo("nav_explore: RETURN - connecting to move_base...")
        if not self.mb_client.wait_for_server(rospy.Duration(10.0)):
            rospy.logerr("nav_explore: move_base unavailable - stopping.")
            self.state = "DONE"
            return

        self._recover_from_wall()

        half = self.start_yaw / 2.0
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id    = "map"
        goal.target_pose.header.stamp       = rospy.Time.now()
        goal.target_pose.pose.position.x    = self.start_x
        goal.target_pose.pose.position.y    = self.start_y
        goal.target_pose.pose.orientation.x = 0.0
        goal.target_pose.pose.orientation.y = 0.0
        goal.target_pose.pose.orientation.z = math.sin(half)
        goal.target_pose.pose.orientation.w = math.cos(half)

        rospy.loginfo("nav_explore: returning to (%.2f, %.2f)",
                      self.start_x, self.start_y)
        self.mb_client.send_goal(goal)
        self._return_start_time = rospy.get_time()
        self.state = "RETURN"

    def _return_step(self):
        mb_state = self.mb_client.get_state()
        elapsed  = rospy.get_time() - self._return_start_time

        if mb_state == 3:   # SUCCEEDED
            rospy.loginfo("nav_explore: returned to start -> DONE")
            self.state = "DONE"
            return

        goal_failed = mb_state in (2, 4, 5, 9)
        goal_timed  = elapsed > self.goal_timeout

        if goal_failed or goal_timed:
            reason = ("timeout (%.0f s)" % elapsed if goal_timed
                      else "failed (state=%d)" % mb_state)
            rospy.logwarn("nav_explore: return goal %s -> DONE", reason)
            self.mb_client.cancel_goal()
            self.state = "DONE"
            return

        rospy.loginfo_throttle(3.0,
            "nav_explore: RETURN in progress... %.0f s elapsed, mb_state=%d",
            elapsed, mb_state)

    # ─────────────────────────── main loop ───────────────────────────────
    def run(self):
        rate = rospy.Rate(self.control_rate)

        while not rospy.is_shutdown():

            self._update_pose()
            dist = self._get_distances()

            if dist is None:
                rospy.logwarn_throttle(2.0,
                    "nav_explore: waiting for /scan and TF...")
                self._stop()
                rate.sleep()
                continue

            rospy.loginfo_throttle(
                1.0,
                "state=%-12s  front=%s  left=%s  right=%s  pos=(%.1f,%.1f)",
                self.state,
                "%.2f" % dist["front"]  if dist["front"]  is not None else "-",
                "%.2f" % dist["left"]   if dist["left"]   is not None else "-",
                "%.2f" % dist["right"]  if dist["right"]  is not None else "-",
                self.robot_x, self.robot_y,
            )

            if self.state in ("FOLLOW", "SEEK_PATH", "NAV_FRONTIER"):
                if self._check_global_timeout():
                    self._force_return()
                    rate.sleep()
                    continue

            if self.state == "FOLLOW":
                self._follow_step(dist)

            elif self.state == "SEEK_PATH":
                self._seek_path_step(dist)

            elif self.state == "NAV_FRONTIER":
                self._nav_frontier_step()

            elif self.state == "RETURN":
                self._return_step()

            elif self.state == "DONE":
                self._save_defects()   # ver2: ensure saved even if DONE was reached directly
                self._stop()
                rospy.loginfo_throttle(5.0,
                    "nav_explore: exploration complete. Robot stopped.")

            rate.sleep()


# ═══════════════════════════════════════════════════════ entry point ══════

if __name__ == "__main__":
    try:
        NavExplore().run()
    except rospy.ROSInterruptException:
        pass
