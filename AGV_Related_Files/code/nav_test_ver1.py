#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nav_test.py  v4
Autonomous corridor exploration: DT-ridgeline centering + frontier-based
navigation + return-to-origin.

v4 changes (vs v3):
  - [FIX-1] Bend detection: front blocked + side open = steer, not SEEK_PATH
  - [FIX-2] Narrower blocking cone (20°) separate from speed cone (40°)
  - [FIX-3] Front distance EMA smoothing to prevent sudden speed jumps
  - [FIX-4] Speed-angular coupling: reduce linear speed when turning hard
  - [FIX-5] Slower acceleration ramp to reduce jerkiness
  - [FIX-6] Increased seek_trigger_count for less premature SEEK_PATH entry

States
------
FOLLOW       : Reactive corridor centering via local DT ridgeline (lidar).
               Enters SEEK_PATH as soon as the front is confirmed blocked.
SEEK_PATH    : Robot stops and rotates continuously until BOTH conditions hold:
                 (1) lidar front reading > front_clear_dist  (physically passable)
                 (2) map ray-cast finds unknown cells ahead  (not yet scanned)
               Only accepts a direction that leads somewhere new.
               After a full 360° without success → NAV_FRONTIER (move_base).
NAV_FRONTIER : move_base navigates to the highest-scoring frontier cluster.
               Handles furniture / obstacles that pure reactive control cannot
               bypass.  Failed targets are blacklisted to avoid infinite retries.
RETURN       : move_base goal back to the recorded start position.
DONE         : Publish zero velocity and idle.
"""

import math
import numpy as np
import rospy
import tf
import actionlib
import scipy.ndimage as ndi
from scipy.ndimage import binary_dilation, label as ndi_label

from sensor_msgs.msg   import LaserScan
from nav_msgs.msg      import OccupancyGrid
from geometry_msgs.msg import Twist
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
        # seek_trigger_count : consecutive blocked frames before SEEK_PATH
        # seek_check_dist    : how far ahead to ray-cast for unknown cells (m)
        # seek_timeout       : max time in SEEK_PATH before fallback (s)
        # min_seek_rotation  : robot must rotate at least this many radians
        #                      before any "clear ahead" direction is accepted.
        #                      Prevents micro-turns (e.g. 3°) that barely miss
        #                      the wall.  Default 60° = 1.047 rad.
        # ──────────────────────────────────────────────────────────────
        # [FIX-6] 增大 seek_trigger_count: 5 → 10 帧 (1秒@10Hz)
        # 原来 0.5 秒就进 SEEK_PATH，弯道中来不及转向就被判定阻塞。
        # ──────────────────────────────────────────────────────────────
        self.seek_trigger_count = int(  rospy.get_param("~seek_trigger_count", 10))
        self.seek_check_dist    = float(rospy.get_param("~seek_check_dist",    3.0))
        self.seek_timeout       = float(rospy.get_param("~seek_timeout",       35.0))
        self.min_seek_rotation  = float(rospy.get_param("~min_seek_rotation",  math.radians(60)))
        self.post_turn_frames   = int(  rospy.get_param("~post_turn_frames",   20))
        self.startup_grace_count= int(  rospy.get_param("~startup_grace_count",20))
        self.min_inter_turn_secs= float(rospy.get_param("~min_inter_turn_secs",2.0))

        # ── [FIX-1] bend detection parameters ─────────────────────────
        # bend_side_thresh: 前方被挡但侧前方距离超过此值，判定为弯道
        # bend_steer_gain: 弯道转向的增益
        self.bend_side_thresh = float(rospy.get_param("~bend_side_thresh", 0.80))
        self.bend_steer_gain  = float(rospy.get_param("~bend_steer_gain",  0.40))

        # ── [FIX-3] front distance smoothing ──────────────────────────
        # EMA alpha: 0.3 = 平滑强度高，变化慢；0.7 = 更跟随原始值
        self.front_ema_alpha = float(rospy.get_param("~front_ema_alpha", 0.35))

        # ── [FIX-4] speed-angular coupling ────────────────────────────
        # 转弯角速度超过此阈值时，开始降低线速度
        self.ang_speed_reduce_thresh = float(rospy.get_param("~ang_speed_reduce_thresh", 0.05))

        # ── [FIX-10] curvature constraint (prevent wheel reversal) ────
        # K ≈ 轮距/2.  轮子反转条件: |ω| > v / K.
        # FOLLOW 状态下自动约束 angular ≤ speed / K.
        self.wheel_k = float(rospy.get_param("~wheel_k", 0.12))

        # ── [FIX-11] max wheel speed + ratio (move_base style) ──────────────
        # 左轮 wL = v − K·ω，右轮 wR = v + K·ω.
        # max_wheel_speed: 外轮绝对速度上限，等比例缩小 (v, ω)。
        # max_wheel_ratio: 快轮/慢轮速度比上限（防极端差速漂移）。
        #   导出角速度限制: |ω| ≤ v*(R-1) / (K*(R+1)), R=max_wheel_ratio.
        #   R=4 → ang_limit=0.6*v/K; R→∞ → v/K (退化为防反转).
        self.max_wheel_speed = float(rospy.get_param("~max_wheel_speed", 0.08))
        self.max_wheel_ratio = float(rospy.get_param("~max_wheel_ratio", 1.2))

        # ── [FIX-13] per-wheel rate limiter ───────────────────────────────────
        # 直接对每个轮子的输出速度做斜率约束（单位: m/s/帧）。
        # 比对 v 和 ω 分别限幅更直接：两个方向的变化叠加仍受同一上限。
        self.max_wheel_accel = float(rospy.get_param("~max_wheel_accel", 0.002))
        self.last_wL = 0.0
        self.last_wR = 0.0

        # ── persistent wheel-split detection ───────────────────────────────
        # 检测“一侧轮持续加速、另一侧轮持续减速”的漂移模式。
        # 连续满足 split_detect_frames 帧后，立即刹停并进入 SEEK_PATH，
        # 重新做方向与速度规划，避免差速持续累积导致车体横漂/偏航。
        self.split_detect_frames = int(rospy.get_param("~split_detect_frames", 5))
        self.split_delta_eps     = float(rospy.get_param("~split_delta_eps", 0.0015))
        self.split_speed_floor   = float(rospy.get_param("~split_speed_floor", 0.006))
        self.split_cooldown_secs = float(rospy.get_param("~split_cooldown_secs", 1.5))
        self.split_persist_count = 0
        self.split_last_sign     = 0   # +1: 左加右减, -1: 左减右加
        self.split_last_trigger_time = -999.0
        self.split_state_label   = "IDLE"

        # ── [FIX-7] wall recovery before move_base ────────────────────
        # 当 AGV 贴墙太近时，move_base 因起点在 costmap 膨胀区内而
        # 无法规划路径。在发送 move_base 目标之前，先后退脱离墙壁。
        # recovery_clearance: 至少退到此距离以外 (m)
        # recovery_back_speed: 后退速度 (m/s)，负值表示倒车
        # recovery_max_secs: 恢复动作最长执行时间 (s)
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
        self.max_explore_secs       = float(rospy.get_param("~max_explore_secs",       100.0))
        self.no_progress_timeout    = float(rospy.get_param("~no_progress_timeout",    30.0))
        self.no_progress_radius     = float(rospy.get_param("~no_progress_radius",     0.5))

        # ── runtime state ─────────────────────────────────────────────
        self.state = "FOLLOW"
        self.scan  = None

        self.map_array = None    # np.ndarray int8, shape (H, W)
        self.map_info  = None    # nav_msgs/MapMetaData

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

        # ── [FIX-3] EMA smoothed front distance ──────────────────────
        self._front_ema = None   # None = not yet initialised

        # ── FOLLOW counters ───────────────────────────────────────────
        self.blocked_frames_count = 0
        self.last_turn_time       = -100.0
        self.post_turn_counter    = 0

        # ── SEEK_PATH state ───────────────────────────────────────────
        self.seek_total_rot   = 0.0
        self.seek_last_yaw    = None
        self.seek_start_time  = 0.0
        self.seek_turn_dir    = 1.0      # +1 = CCW, -1 = CW
        self.seek_frontiers   = []       # frontiers collected during rotation

        # ── NAV_FRONTIER state ────────────────────────────────────────
        self.frontier_targets    = []
        self._nav_frontier_idx   = 0
        self._nav_frontier_start = 0.0
        self._failed_frontiers   = []

        # ── RETURN state ──────────────────────────────────────────────
        self._return_start_time  = 0.0

        # ── frontier cache ────────────────────────────────────────────
        self._frontier_cache_time = -999.0
        self._frontier_cache      = []

        # ── no-progress watchdog state ────────────────────────────────
        self._progress_x    = 0.0
        self._progress_y    = 0.0
        self._progress_time = 0.0

        # ── TF ────────────────────────────────────────────────────────
        self.tf_listener = tf.TransformListener()

        # ── ROS interface ─────────────────────────────────────────────
        rospy.Subscriber("/scan", LaserScan,     self._scan_cb, queue_size=1)
        rospy.Subscriber("/map",  OccupancyGrid, self._map_cb,  queue_size=1)
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        self.mb_client = actionlib.SimpleActionClient("move_base", MoveBaseAction)

        rospy.loginfo("nav_explore v4: node initialised.")

    # ─────────────────────────────────────────── callbacks ───────────────
    def _scan_cb(self, msg):
        self.scan = msg

    def _map_cb(self, msg):
        self.map_info  = msg.info
        self.map_array = np.array(msg.data, dtype=np.int8).reshape(
                             msg.info.height, msg.info.width)

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
            "front":       self._sector_dist(s,   0+o, 40, inf_fallback=None),
            # ──────────────────────────────────────────────────────────
            # [FIX-2] 新增窄扇区 front_narrow (20°)，用于阻塞判定。
            # 原来40°的扇区在弯道中容易扫到内壁导致误判阻塞。
            # 窄扇区只看正前方±10°，更准确地判断"真正前方是否被挡"。
            # ──────────────────────────────────────────────────────────
            "front_narrow": self._sector_dist(s,  0+o, 20, inf_fallback=None),
            "left":        self._sector_dist(s, +90+o, 30, inf_fallback=M),
            "right":       self._sector_dist(s, -90+o, 30, inf_fallback=M),
            "front_left":  self._sector_dist(s, +45+o, 25, inf_fallback=M),
            "front_right": self._sector_dist(s, -45+o, 25, inf_fallback=M),
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
        Detect persistent drift pattern:
          one wheel keeps accelerating while the other keeps decelerating.
        After N consecutive frames, force stop and re-enter SEEK_PATH so the
        robot replans heading and speed control from zero.
        Also expose per-frame split status in logs for debugging.
        """
        now = rospy.get_time()
        if (now - self.split_last_trigger_time) < self.split_cooldown_secs:
            cooldown_left = max(0.0, self.split_cooldown_secs - (now - self.split_last_trigger_time))
            self.split_state_label = "COOLDOWN %.2fs" % cooldown_left
            return False

        dL = wL - self.last_wL
        dR = wR - self.last_wR
        eps = self.split_delta_eps

        split_sign = 0
        split_label = "IDLE"
        if dL > eps and dR < -eps:
            split_sign = +1   # 左加速，右减速
            split_label = "LEFT_UP_RIGHT_DOWN"
        elif dL < -eps and dR > eps:
            split_sign = -1   # 左减速，右加速
            split_label = "LEFT_DOWN_RIGHT_UP"

        moving_enough = max(abs(wL), abs(wR)) >= self.split_speed_floor
        unbalanced = abs(wL - wR) >= (2.0 * eps)

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
                if not moving_enough:
                    self.split_state_label = "IDLE low_speed"
                elif not unbalanced:
                    self.split_state_label = "IDLE balanced"
                else:
                    self.split_state_label = "IDLE"
            else:
                self.split_state_label = "%s rejected" % split_label
            self.split_persist_count = 0
            self.split_last_sign = 0
            return False

        if self.split_persist_count < self.split_detect_frames:
            return False

        left = dist["left"] if dist and dist.get("left") is not None else 0.0
        right = dist["right"] if dist and dist.get("right") is not None else 0.0
        prefer_left = (left >= right)

        rospy.logwarn(
            "nav_explore: persistent wheel split detected for %d frames "
            "(mode=%s | wL=%.3f dL=%.4f | wR=%.3f dR=%.4f) -> stop and replan via SEEK_PATH",
            self.split_persist_count, split_label, wL, dL, wR, dR)

        self._stop()
        rospy.sleep(0.15)
        self._reset_speed_planner()
        self.split_persist_count = 0
        self.split_last_sign = 0
        self.split_state_label = "TRIGGERED -> SEEK_PATH"
        self.split_last_trigger_time = now
        self._enter_seek_path(prefer_left)
        return True

    def _clamp_cmd(self, lin, ang):
        """
        Wheel-speed normalisation (move_base style), two-stage:

        Stage 1 – proportional scaling:
          wL = v − K·ω,  wR = v + K·ω.
          If either wheel exceeds max_wheel_speed, scale (v, ω) together
          so turning curvature is preserved but the faster wheel is capped.

        Stage 2 – no-reversal clamp (restores FIX-10):
          During forward motion (v > 0), limit |ω| ≤ v/K so neither wheel
          runs backward.  This is the constraint that actually prevents
          the differential drift at low speeds where Stage 1 never fires
          (actual wheel speeds << max_wheel_speed).
        """
        K = self.wheel_k

        # ── Stage 1: proportional max-speed scaling ───────────────────────
        ws = max(abs(lin - K * ang), abs(lin + K * ang))
        if ws > self.max_wheel_speed and ws > 1e-6:
            scale = self.max_wheel_speed / ws
            lin  *= scale
            ang  *= scale

        # ── Stage 2: wheel ratio limit (replaces simple no-reversal) ────────
        # Limit wL/wR ≤ max_wheel_ratio, derived as |ω| ≤ v*(R-1)/(K*(R+1)).
        # R=4 → limit = 0.6*v/K; at full speed this is looser than max_angular.
        # Tighter than plain no-reversal (v/K) so wR stays meaningfully positive.
        if lin > 1e-3:
            R = self.max_wheel_ratio
            ang_ratio_limit = lin * (R - 1.0) / (K * (R + 1.0))
            ang = _clamp(ang, -ang_ratio_limit, ang_ratio_limit)

        # ── Stage 3: absolute angular ceiling ─────────────────────────────
        ang = _clamp(ang, -self.max_angular, self.max_angular)
        return lin, ang

    # ─────────────────────────── map queries ─────────────────────────────
    def _has_unknown_ahead(self, yaw):
        """
        Ray-cast from the robot in direction `yaw` and return True if an
        unknown (unscanned) cell is found before hitting an obstacle.

        This is the key check in SEEK_PATH: a direction is only accepted if
        BOTH the lidar confirms it is physically clear AND this function
        confirms there is still unexplored space along that bearing.
        Prevents the robot from driving back into already-mapped corridors.

        If no map is available yet, returns True (assume unexplored).
        """
        if self.map_array is None or self.map_info is None:
            return True   # map not ready → treat everywhere as unexplored

        res = self.map_info.resolution
        ox  = self.map_info.origin.position.x
        oy  = self.map_info.origin.position.y
        H, W = self.map_array.shape

        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        # Start checking from front_block_dist to avoid the robot's own body.
        d = self.front_block_dist
        while d < self.seek_check_dist:
            px = self.robot_x + d * cos_y
            py = self.robot_y + d * sin_y
            mx = int((px - ox) / res)
            my = int((py - oy) / res)
            if not (0 <= mx < W and 0 <= my < H):
                break                 # left the map boundary
            cell = self.map_array[my, mx]
            if cell > 50:             # occupied → wall before unknown
                return False
            if cell < 0:              # unknown → unscanned space found
                return True
            d += res
        return False

    def _find_frontiers(self):
        """
        Return frontier cluster centroids scored by:
          score = cluster_area_m² × DT_at_centroid / (1 + robot_distance)

        DT (distance_transform_edt) scores openness: high DT = ridgeline of
        free space = ideal corridor centre.  Larger, more central, nearer
        frontiers score highest.  Previously failed positions are excluded.
        """
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

    # ─────────────────────────── global checks ─────────────────────────
    def _check_global_timeout(self):
        """Return True if we should force RETURN due to global timeout or
        no-progress watchdog."""
        if self._explore_start_time is None:
            return False

        now     = rospy.get_time()
        elapsed = now - self._explore_start_time

        # ── global time limit ─────────────────────────────────────────
        if elapsed >= self.max_explore_secs:
            rospy.logwarn("nav_explore: global timeout (%.0f s) -> RETURN",
                          elapsed)
            return True

        # ── no-progress watchdog (only in FOLLOW state) ───────────────
        # SEEK_PATH rotates in place (no translation), so we only track
        # progress when the robot is supposed to be moving forward.
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
            # Reset progress timer when not in FOLLOW so that entering
            # SEEK_PATH / NAV_FRONTIER doesn't accumulate stale time.
            self._progress_time = now

        return False

    def _force_return(self):
        """Cancel any active move_base goal and switch to RETURN."""
        if self.state == "NAV_FRONTIER":
            self.mb_client.cancel_goal()
            rospy.sleep(0.2)
        self._stop()
        self._enter_return()

    # ─────────────────────────── FOLLOW state ────────────────────────────
    def _follow_step(self, dist):
        """
        Reactive DT-ridgeline centering.

        Centering: error = left_dist − right_dist
        The ridgeline is where error = 0 (maximum of the local DT slice).
        A P-controller drives the robot onto the ridge.

        Blocked detection: if front stays below front_block_dist for
        seek_trigger_count consecutive frames → enter SEEK_PATH.
        We deliberately do NOT apply angular correction when blocked so that
        SEEK_PATH owns all turning behaviour; the centering controller is
        only active when the robot is actually moving forward.

        v4: bend detection, front EMA smoothing, speed-angular coupling.
        """
        left        = dist["left"]
        right       = dist["right"]
        front_left  = dist["front_left"]
        front_right = dist["front_right"]
        front_raw   = dist["front"]

        left_real  = (left  is not None and left  < self.side_open_dist)
        right_real = (right is not None and right < self.side_open_dist)

        # Front distance: conservative decay when lidar returns None.
        if front_raw is None:
            decay = (self.forward_speed / self.control_rate) * 2.0
            self.last_front_valid = max(self.last_front_valid - decay,
                                        self.min_valid_range)
            front = self.last_front_valid
        else:
            front = front_raw
            self.last_front_valid = front

        # ──────────────────────────────────────────────────────────────
        # [FIX-2] 使用窄扇区(20°)判定阻塞，而非原来的40°宽扇区。
        # 40°扇区在弯道中会扫到弯道内壁导致误判，20°只看正前方。
        # ──────────────────────────────────────────────────────────────
        front_narrow = dist.get("front_narrow", front_raw)
        if front_narrow is None:
            front_narrow = front   # fallback to wide sector

        is_blocked = (front_narrow < self.front_block_dist)

        # ──────────────────────────────────────────────────────────────
        # [FIX-3] 非对称 EMA 平滑前方距离，用于速度控制。
        #
        # 核心原则：加速慢，减速快。
        #   - 距离增大（远离墙壁）→ EMA 缓慢跟随 → 速度渐升
        #   - 距离减小（接近墙壁）→ 直接取原始值 → 速度立刻降
        #
        # 原来的对称 EMA 在接近墙壁时会严重滞后：实际距离已经
        # 0.37m 了，EMA 还停在 1.0m，AGV 以全速撞向墙壁。
        # ──────────────────────────────────────────────────────────────
        if self._front_ema is None:
            self._front_ema = front_narrow
        else:
            alpha = self.front_ema_alpha
            self._front_ema = alpha * front_narrow + (1.0 - alpha) * self._front_ema
        # 非对称：取 EMA 和原始值中的较小值
        # 距离增大时 front_narrow > EMA → 用 EMA（平滑加速）
        # 距离减小时 front_narrow < EMA → 用 front_narrow（立即减速）
        front_smooth = min(self._front_ema, front_narrow)

        rospy.loginfo("blk=%d fn=%.2f fs=%.2f spd=%.3f ang=%.2f blocked=%s",
              self.blocked_frames_count, front_narrow,
              front_smooth, self.last_speed, self.last_angular,
              str(is_blocked))

        if self.startup_grace_count > 0:
            self.startup_grace_count -= 1

        # ──────────────────────────────────────────────────────────────
        # [FIX-1] 弯道检测（被动策略）
        #
        # 如果正前方被挡但侧前方有通路，判定为弯道：
        #   - 设 is_blocked=False，让 centering 控制器自然转弯
        #   - 不重置 blocked_frames_count（关键！）
        #     在真弯道中 AGV 会前进，speed>0 时计数器自然清零；
        #     在假弯道（死胡同入口反复振荡）中 AGV 不动，
        #     FIX-8 可以持续累积计数器直到触发 SEEK_PATH。
        # ──────────────────────────────────────────────────────────────
        fl_open = (front_left  is not None and front_left  > self.bend_side_thresh)
        fr_open = (front_right is not None and front_right > self.bend_side_thresh)
        is_bend = is_blocked and (fl_open or fr_open)

        if is_bend:
            # 注意：不清零 blocked_frames_count！
            is_blocked = False   # fall through 到正常控制
            rospy.loginfo_throttle(1.0,
                "nav_explore: BEND detected (fl=%.2f fr=%.2f) - centering handles turn",
                front_left if front_left else 0,
                front_right if front_right else 0)

        # ── blocked → count frames; enter SEEK_PATH when threshold reached ─
        if is_blocked and self.startup_grace_count <= 0:
            self.blocked_frames_count += 1
            if self.blocked_frames_count >= self.seek_trigger_count:
                rospy.loginfo("ENTER SEEK_PATH triggered! blocked_count=%d front=%.2f",
                                self.blocked_frames_count, front)
                # Choose rotation direction: toward the more open side.
                l = left  if left  is not None else 0.0
                r = right if right is not None else 0.0
                prefer_left = (l >= r)
                self.blocked_frames_count = 0
                rospy.loginfo("nav_explore: FOLLOW → SEEK_PATH (blocked %d frames)",
                              self.seek_trigger_count)
                self._enter_seek_path(prefer_left)
                return False
            # While counting, publish zero velocity (no angular correction).
            self._pub(0.0, 0.0)
            self.last_speed   = 0.0
            self.last_angular = 0.0
            self.last_wL = 0.0
            self.last_wR = 0.0
            return True
        else:
            # 仅在 AGV 实际移动时清零 blocked 计数。
            # 如果 last_speed=0（速度死区），不清零，让 FIX-8 累积。
            if front_narrow > self.front_clear_dist and self.last_speed > 0:
                self.blocked_frames_count = 0

        # ── DT ridgeline centering ─────────────────────────────────────
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

        # Post-turn heading alignment.
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

        # ── speed control ──────────────────────────────────────────────
        # ──────────────────────────────────────────────────────────────
        # [FIX-3+9] 速度计算：消除速度死区
        # 原问题：front_smooth 在 0.65~0.69 范围时，speed 接近 0 但
        #         AGV 不算 blocked，导致既不动也不找路。
        # 修复：未 blocked 时保证最低 15% 速度（0.012 m/s），
        #       足以让 AGV 缓慢前进或触发真正的阻塞。
        # ──────────────────────────────────────────────────────────────
        if front_smooth < 1.0:
            ratio = (front_smooth - self.front_block_dist) / max(0.01, 1.0 - self.front_block_dist)
            ratio = max(0.15, ratio)   # 最低 15% 速度
            target_speed = self.forward_speed * max(0.0, ratio)
        else:
            target_speed = self.forward_speed

        if self.startup_grace_count > 0:
            target_speed = min(target_speed, self.forward_speed * 0.5)

        # ──────────────────────────────────────────────────────────────
        # [FIX-12] BEND 模式下已进入阻塞距离时，强制线速度为 0。
        #
        # 根因：FIX-9 的 15% 速度下限在 front_smooth < front_block_dist
        # 时仍然生效，导致 AGV 在 BEND 状态下以 0.012 m/s 继续向障碍物
        # 前进（撞箱子 / 蹭箱子），同时低速+角速度触发 wL/wR 4:1 漂移。
        #
        # 修复：BEND 且已在阻塞距离内 → target_speed=0 → 原地旋转。
        #   原地旋转时 lin=0，_clamp_cmd Stage2 不激活，角速度不受
        #   轮速比约束 → 转弯更锐利，wL=-K*ω / wR=+K*ω 完全对称。
        #   fn 一旦重新超过 front_block_dist，15% 下限自然恢复。
        # ──────────────────────────────────────────────────────────────
        if is_bend and front_smooth < self.front_block_dist:
            target_speed = 0.0

        # ──────────────────────────────────────────────────────────────
        # [FIX-4] 速度-角速度耦合：转弯越急，线速度越低。
        # 防止 AGV 在高角速度时同时高速前进（急转+加速）。
        # 当 |angular| > ang_speed_reduce_thresh 时，
        # 线速度按比例衰减，最低降到 50%。
        # ──────────────────────────────────────────────────────────────
        abs_ang = abs(angular)
        if abs_ang > self.ang_speed_reduce_thresh:
            ang_ratio = (abs_ang - self.ang_speed_reduce_thresh) / \
                        max(0.01, self.max_angular - self.ang_speed_reduce_thresh)
            ang_ratio = min(1.0, ang_ratio)
            speed_factor = 1.0 - 0.5 * ang_ratio
            target_speed *= speed_factor

        # ──────────────────────────────────────────────────────────────
        # [FIX-5] 加速斜率: 0.005/帧 (从0到满速需1.6秒)，减速 0.06/帧。
        # 麦克纳姆轮 AGV 在加速同时转弯时体感很急，放慢加速。
        # ──────────────────────────────────────────────────────────────
        if target_speed > self.last_speed:
            speed = min(target_speed, self.last_speed + 0.005)
        else:
            speed = max(target_speed, self.last_speed - 0.06)
        if speed < 0.003:
            speed = 0.0
        self.last_speed = speed

        # ──────────────────────────────────────────────────────────────
        # [FIX-10/11] 归一化轮速：等比例限制 (v, ω) 使外轮不超过
        # max_wheel_speed，同时防止轮子反转（原 FIX-10 的功能被
        # _clamp_cmd 的等比例缩放完全覆盖，不再单独处理）。
        # ──────────────────────────────────────────────────────────────
        speed, angular = self._clamp_cmd(speed, angular)
        self.last_angular = angular
        self.last_speed   = speed   # 等比例缩放可能降低线速度

        # ──────────────────────────────────────────────────────────────
        # [FIX-8] 消除速度死区：如果速度被截为 0 但没有进入 SEEK_PATH，
        # 说明 front_narrow 刚好卡在 front_block_dist 附近的死区里。
        # 此时 AGV 既不动也不找路，会被 no_progress 超时强制 RETURN。
        # 修复：速度为 0 时也累积 blocked 帧数，这样死区中的 AGV
        # 会在 seek_trigger_count 帧后进入 SEEK_PATH 主动寻路。
        # ──────────────────────────────────────────────────────────────
        if speed == 0.0 and not is_bend and self.startup_grace_count <= 0:
            self.blocked_frames_count += 1
            if self.blocked_frames_count >= self.seek_trigger_count:
                rospy.loginfo("nav_explore: speed dead zone → SEEK_PATH "
                              "(fn=%.2f fs=%.2f)", front_narrow, front_smooth)
                l = left  if left  is not None else 0.0
                r = right if right is not None else 0.0
                self.blocked_frames_count = 0
                self._enter_seek_path(l >= r)
                return False

        # ──────────────────────────────────────────────────────────────
        # [FIX-13] Per-wheel rate limiter.
        # 对每个轮子的速度变化率直接限幅，避免 v 和 ω 叠加后单轮跳变过大。
        # ──────────────────────────────────────────────────────────────
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
        # Recover (v, ω) from smoothed wheel speeds for publishing
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
        """
        Stop and begin deliberate rotation to find a clear, unexplored heading.

        prefer_left : rotate CCW first (toward the more open side).
        """
        self._stop()
        # Back up briefly if close to front wall - gives clearance for rotation
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
        self.split_last_sign = 0
        self._frontier_cache_time = -999.0   # force fresh map read
        # Clear fallback heading from any previous SEEK_PATH cycle.
        if hasattr(self, '_seek_fallback_yaw'):
            delattr(self, '_seek_fallback_yaw')
        if hasattr(self, '_seek_fallback_front'):
            delattr(self, '_seek_fallback_front')
        rospy.loginfo("nav_explore: SEEK_PATH - rotating to find clear+unexplored direction "
                      "(%s first).", "CCW" if prefer_left else "CW")

    def _seek_path_step(self, dist):
        """
        Each control cycle:
          1. Accumulate signed rotation.
          2. Check: lidar front clear AND map ray finds unknown cells.
          3. If both true → resume FOLLOW in current heading.
          4. Simultaneously collect all frontier clusters for fallback.
          5. After 360° (or timeout) without success → NAV_FRONTIER or RETURN.

        The combined check (lidar + map) ensures the robot only proceeds
        into directions that are both physically passable and not yet mapped.
        """
        self._update_pose()

        # ── accumulate rotation ───────────────────────────────────────
        if self.seek_last_yaw is not None:
            self.seek_total_rot += _angle_diff(self.robot_yaw, self.seek_last_yaw)
        self.seek_last_yaw = self.robot_yaw

        # ── evaluate current heading ──────────────────────────────────
        front_raw = dist["front"] if dist is not None else None
        if front_raw is not None:
            self.last_front_valid = front_raw
        front = front_raw if front_raw is not None else self.last_front_valid

        front_clear = (front is not None and front > self.front_clear_dist)

        # ── Bug fix: require minimum rotation before accepting any direction ──
        # Without this guard, the robot turns only 2-3° (barely clears the
        # wall reading) and immediately snaps back to FOLLOW, achieving no
        # real obstacle avoidance.
        # With min_seek_rotation (default 60°) the robot must turn a meaningful
        # angle before the "clear ahead" check is even evaluated.
        min_rot_met = abs(self.seek_total_rot) >= self.min_seek_rotation

        # ── Bug fix: don't use map=None as a free pass for unknown_ahead ──
        # When the map hasn't been published yet, _has_unknown_ahead returns
        # True unconditionally.  Previously this combined with a slightly-off-
        # angle lidar reading to exit SEEK_PATH after just a few degrees.
        # Now we only query _has_unknown_ahead when the map IS available.
        # If the map is absent, we accept any clear direction (after min rot).
        if front_clear and min_rot_met:
            if self.map_array is None:
                # No map yet: accept any clear direction.
                unknown_ok = True
            else:
                # Map available: prefer unexplored directions; fall back to
                # any clear direction only if a full rotation found nothing.
                unknown_ok = self._has_unknown_ahead(self.robot_yaw)

            if unknown_ok:
                rospy.loginfo("nav_explore: SEEK_PATH → FOLLOW "
                              "(front=%.2f m, rot=%.0f°, yaw=%.2f rad)",
                              front, math.degrees(abs(self.seek_total_rot)),
                              self.robot_yaw)
                self._stop()
                self.state              = "FOLLOW"
                self.last_turn_time     = rospy.get_time()
                self.last_angular       = 0.0
                self.last_speed         = 0.0
                self.post_turn_counter  = self.post_turn_frames
                self.blocked_frames_count = 0
                # [FIX-3] 重置 EMA，防止旧的平滑值影响新方向的速度
                self._front_ema = None
                return

        # ── record best "clear but already-mapped" fallback yaw ──────
        # If the map shows this direction is already explored, save it as a
        # last-resort heading in case the full 360° finds no unknown direction.
        if front_clear and min_rot_met and self.map_array is not None:
            if not hasattr(self, '_seek_fallback_yaw') or \
               front > getattr(self, '_seek_fallback_front', 0.0):
                self._seek_fallback_yaw   = self.robot_yaw
                self._seek_fallback_front = front

        # ── collect frontiers for fallback (throttled 1 Hz) ──────────
        for (wx, wy) in self._find_frontiers_cached(max_age=1.0):
            already = any(math.hypot(wx - ex, wy - ey) < 0.3
                          for (ex, ey) in self.seek_frontiers)
            if not already:
                self.seek_frontiers.append((wx, wy))

        # ── completion check ──────────────────────────────────────────
        full_rotation = abs(self.seek_total_rot) >= (2 * math.pi - 0.15)
        timed_out     = (rospy.get_time() - self.seek_start_time > self.seek_timeout)

        if full_rotation or timed_out:
            if self.seek_frontiers:
                # Frontier clusters found → let move_base plan a route to them.
                rospy.loginfo("nav_explore: SEEK_PATH 360° - %d frontier(s) → NAV_FRONTIER",
                              len(self.seek_frontiers))
                self.frontier_targets  = list(self.seek_frontiers)
                self._nav_frontier_idx = 0
                self._enter_nav_frontier()
            elif hasattr(self, '_seek_fallback_yaw'):
                # No unexplored direction found, but at least one physically
                # clear direction was recorded during rotation.  Rotate to it
                # then resume FOLLOW; the robot will move until blocked again.
                rospy.loginfo("nav_explore: SEEK_PATH 360° - no unexplored dir; "
                              "aligning to best clear yaw=%.2f rad → FOLLOW",
                              self._seek_fallback_yaw)
                self._stop()
                self.state = "FOLLOW"
                self.last_turn_time   = rospy.get_time()
                self.last_angular     = 0.0
                self.last_speed       = 0.0
                self.post_turn_counter = self.post_turn_frames
                self.blocked_frames_count = 0
                # [FIX-3] 重置 EMA
                self._front_ema = None
                delattr(self, '_seek_fallback_yaw')
                delattr(self, '_seek_fallback_front')
            else:
                elapsed = (rospy.get_time() - self._explore_start_time
                           if self._explore_start_time else 0.0)
                if elapsed < self.min_explore_secs:
                    rospy.loginfo("nav_explore: SEEK_PATH - no clear direction yet "
                                  "(%.0f s elapsed) → FOLLOW", elapsed)
                    self._stop()
                    self.state = "FOLLOW"
                    self.last_turn_time = rospy.get_time()
                    self._front_ema = None  # [FIX-3]
                else:
                    rospy.loginfo("nav_explore: SEEK_PATH - fully enclosed → RETURN")
                    self._stop()
                    self._enter_return()
            return

        # ── keep rotating ─────────────────────────────────────────────
        target_ang = self.seek_turn_dir * self.turn_speed
        angular    = _clamp(target_ang,
                            self.last_angular - self.max_angular_rate * 3,
                            self.last_angular + self.max_angular_rate * 3)
        _, angular = self._clamp_cmd(0.0, angular)   # cap wheel speed during spin
        # [FIX-13] per-wheel rate limit during rotation (lin=0 → wL=-K*ω, wR=+K*ω)
        K  = self.wheel_k
        da = self.max_wheel_accel
        wL = _clamp(-K * angular, self.last_wL - da, self.last_wL + da)
        wR = _clamp(+K * angular, self.last_wR - da, self.last_wR + da)
        self.last_wL = wL
        self.last_wR = wR
        angular = (wR - wL) / (2.0 * K)
        self.last_angular = angular
        self._pub(0.0, angular)

    # ─────────────────────────── [FIX-7] wall recovery ───────────────
    def _recover_from_wall(self):
        """
        阻塞式恢复动作：检查前方/左/右距离，如果任一方向贴墙太近
        （< recovery_clearance），则后退+转离墙壁，直到获得足够空间。

        这解决了 move_base 因起始点在 costmap 膨胀区内无法规划的问题。
        在 _enter_nav_frontier() 和 _enter_return() 之前调用。
        """
        rate = rospy.Rate(self.control_rate)
        start_time = rospy.get_time()

        while not rospy.is_shutdown():
            elapsed = rospy.get_time() - start_time
            if elapsed > self.recovery_max_secs:
                rospy.logwarn("nav_explore: wall recovery timeout (%.1f s)", elapsed)
                break

            # 刷新传感器数据
            self._update_pose()
            dist = self._get_distances()
            if dist is None:
                rate.sleep()
                continue

            front_d = dist["front"] if dist["front"] is not None else 999.0
            left_d  = dist["left"]  if dist["left"]  is not None else 999.0
            right_d = dist["right"] if dist["right"] is not None else 999.0
            min_d   = min(front_d, left_d, right_d)

            # 已经有足够空间，停止恢复
            if min_d >= self.recovery_clearance:
                rospy.loginfo("nav_explore: wall recovery done, clearance=%.2f m "
                              "(took %.1f s)", min_d, elapsed)
                break

            # 后退 + 根据左右距离转向（远离更近的墙）
            back_speed = -self.recovery_back_speed

            # 前方太近：纯后退
            # 左边太近：后退 + 右转（angular 负）
            # 右边太近：后退 + 左转（angular 正）
            ang = 0.0
            if left_d < self.recovery_clearance and right_d >= self.recovery_clearance:
                ang = -0.3   # 右转远离左墙
            elif right_d < self.recovery_clearance and left_d >= self.recovery_clearance:
                ang =  0.3   # 左转远离右墙
            elif left_d < right_d:
                ang = -0.2   # 两边都近，偏向更远的一边
            elif right_d < left_d:
                ang =  0.2

            self._pub(back_speed, ang)
            rospy.loginfo_throttle(1.0,
                "nav_explore: recovering... front=%.2f left=%.2f right=%.2f "
                "back=%.3f ang=%.2f", front_d, left_d, right_d, back_speed, ang)
            rate.sleep()

        self._stop()
        rospy.sleep(0.3)   # 让底盘完全停稳

    # ─────────────────────────── NAV_FRONTIER state ──────────────────────
    def _enter_nav_frontier(self):
        """
        Send a move_base goal to frontier_targets[_nav_frontier_idx].
        move_base uses the global planner and can route around furniture /
        beds that pure reactive control cannot bypass.
        """
        if self._nav_frontier_idx >= len(self.frontier_targets):
            rospy.loginfo("nav_explore: all frontier targets exhausted → RETURN")
            self._stop()
            self._enter_return()
            return

        if not self.mb_client.wait_for_server(rospy.Duration(3.0)):
            rospy.logerr("nav_explore: move_base not available → RETURN")
            self._stop()
            self._enter_return()
            return

        # [FIX-7] 发送目标前，先确保不贴墙
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
        rospy.loginfo("nav_explore: NAV_FRONTIER %d/%d → (%.2f, %.2f)",
                      self._nav_frontier_idx + 1, len(self.frontier_targets), wx, wy)

    def _nav_frontier_step(self):
        """
        Non-blocking poll of the current move_base goal.
        SUCCEEDED  → FOLLOW
        ABORTED / REJECTED / LOST / timeout → blacklist + try next frontier
        """
        mb_state = self.mb_client.get_state()
        elapsed  = rospy.get_time() - self._nav_frontier_start

        if mb_state == 3:   # SUCCEEDED
            wx, wy = self.frontier_targets[self._nav_frontier_idx]
            rospy.loginfo("nav_explore: frontier reached (%.2f, %.2f) → FOLLOW", wx, wy)
            self.state              = "FOLLOW"
            self.last_turn_time     = rospy.get_time()
            self.last_angular       = 0.0
            self.last_speed         = 0.0
            self.post_turn_counter  = self.post_turn_frames
            self.blocked_frames_count = 0
            self._front_ema = None  # [FIX-3] 重置 EMA
            return

        goal_failed = mb_state in (2, 4, 5, 9)
        goal_timed  = elapsed > self.nav_goal_timeout

        if goal_failed or goal_timed:
            wx, wy = self.frontier_targets[self._nav_frontier_idx]
            reason = "timeout" if goal_timed else "aborted (state=%d)" % mb_state
            rospy.logwarn("nav_explore: frontier (%.2f, %.2f) %s → trying next.",
                          wx, wy, reason)
            self.mb_client.cancel_goal()
            rospy.sleep(0.2)
            self._failed_frontiers.append((wx, wy))
            self._nav_frontier_idx += 1
            self._enter_nav_frontier()

    # ─────────────────────────── RETURN state ────────────────────────────
    def _enter_return(self):
        """Send move_base goal to start position (non-blocking)."""
        if self.start_x is None:
            rospy.logwarn("nav_explore: no start pose recorded; stopping.")
            self.state = "DONE"
            return

        rospy.loginfo("nav_explore: RETURN - connecting to move_base...")
        if not self.mb_client.wait_for_server(rospy.Duration(10.0)):
            rospy.logerr("nav_explore: move_base unavailable - stopping.")
            self.state = "DONE"
            return

        # [FIX-7] 发送目标前，先确保不贴墙
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
        """Non-blocking poll of the return goal (same pattern as NAV_FRONTIER)."""
        mb_state = self.mb_client.get_state()
        elapsed  = rospy.get_time() - self._return_start_time

        if mb_state == 3:   # SUCCEEDED
            rospy.loginfo("nav_explore: returned to start -> DONE")
            self.state = "DONE"
            return

        goal_failed = mb_state in (2, 4, 5, 9)
        goal_timed  = elapsed > self.goal_timeout

        if goal_failed or goal_timed:
            reason = "timeout (%.0f s)" % elapsed if goal_timed else "failed (state=%d)" % mb_state
            rospy.logwarn("nav_explore: return goal %s -> DONE", reason)
            self.mb_client.cancel_goal()
            self.state = "DONE"
            return

        # Log progress periodically
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
                    "nav_explore: waiting for /scan and TF…")
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

            # ── global timeout / no-progress check ─────────────────
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
                self._return_step()   # non-blocking poll

            elif self.state == "DONE":
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