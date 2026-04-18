#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
defect_localizer_node.py  ── Jetson 端 ROS 缺陷定位节点
========================================================
功能：
  1. 启动 RealSense D435 彩色流 + 深度流（深度帧对齐到彩色帧）
  2. 加载 TensorRT 引擎（erosion 分割 + crack 检测）
  3. 通过串口控制双轴云台（SimpleFOC，指令格式: "T{angle_rad}\\n"）
  4. 检测到缺陷并完成对准（进入 DEADZONE）后：
       a. 读取 RealSense 对齐深度值（窗口中值滤波）
       b. 利用相机内参反投影得到相机光学坐标系 3D 点
       c. 经由云台安装外参 + 当前云台角度逐步转换到 AGV base_footprint 坐标系
       d. 通过 ROS TF 转换到全局 map 坐标系
       e. 发布 /defect_candidates（geometry_msgs/PoseStamped，frame_id="map"）
  5. 同时将检测图像保存到本地（~/defect_records/）

运行前提：
  - Jetson 已安装 ROS Noetic，已 source /opt/ros/noetic/setup.bash
  - export ROS_MASTER_URI=http://<AGV_IP>:11311
  - export ROS_IP=<JETSON_IP>
  - AGV 上 SLAM 正在运行（提供 TF: map → base_footprint）
  - TensorRT 引擎文件存在（见 ENGINE_SEG / ENGINE_DET 路径）
  - 两块 STM32 通过 USB 连接到 Jetson

用法：
  python3 defect_localizer_node.py

作者注：所有标注 [CALIBRATION NEEDED] 的参数均需实际测试后确认。
"""

import os
import sys
import math
import time
import ctypes
from datetime import datetime

import numpy as np
import cv2
import serial
import serial.tools.list_ports
import pyrealsense2 as rs

# ── ROS ───────────────────────────────────────────────────────────────────────
try:
    import rospy
    import tf
    import tf.transformations
    from geometry_msgs.msg import PoseStamped
except ImportError:
    print("ERROR: ROS Python packages not found.")
    print("Please run:  source /opt/ros/noetic/setup.bash")
    sys.exit(1)

# ── TensorRT ──────────────────────────────────────────────────────────────────
try:
    import tensorrt as trt
except ImportError:
    print("ERROR: tensorrt not found. Is it installed on this Jetson?")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  ① TensorRT 引擎路径
# ══════════════════════════════════════════════════════════════════════════════

# libcudart 路径（Jetson Nano JetPack 4.x 默认位置）
LIBCUDART_PATH = "/usr/local/cuda-10.2/targets/aarch64-linux/lib/libcudart.so.10.2"

# TRT 引擎文件（erosion = 分割模型，crack = 检测模型）
ENGINE_SEG = "/home/fyp/robot_project/defect_detection/erosion_fp16.engine"
ENGINE_DET = "/home/fyp/robot_project/defect_detection/crack_fp16.engine"

# 推理输入尺寸（与训练时一致）
INPUT_W    = 640
INPUT_H    = 640

# 推理置信度阈值
CONF_DET    = 0.25
CONF_SEG    = 0.25
IOU_THRESH  = 0.50
MASK_THRESH = 0.50


# ══════════════════════════════════════════════════════════════════════════════
#  ② RealSense 相机参数
# ══════════════════════════════════════════════════════════════════════════════

COLOR_W   = 640
COLOR_H   = 480
COLOR_FPS = 30

# 深度有效范围（米）。超出范围的深度值视为无效，不触发定位。
DEPTH_MIN_M = 0.15   # 太近（< 15 cm）的深度通常是噪声
DEPTH_MAX_M = 5.0    # 太远（> 5 m）D435 精度已很差

# 深度采样窗口半径（像素）
# 在检测框中心点周围 ±DEPTH_SAMPLE_RADIUS 像素范围内取中位数，减少噪声影响
DEPTH_SAMPLE_RADIUS = 5


# ══════════════════════════════════════════════════════════════════════════════
#  ③ 视觉伺服参数
# ══════════════════════════════════════════════════════════════════════════════

# 比例系数：像素误差 → 每帧角度增量（弧度）
KP_YAW   = 0.0005
KP_PITCH = 0.0005

# [CALIBRATION NEEDED] 方向符号 (+1 或 -1)
# 测试方法：固定 pitch，让相机看到画面右侧目标（err_x > 0），
# 观察云台是否向右转（相机中心趋近目标）。
#   若向右转 → DIR_YAW = +1（当前值）
#   若向左转 → DIR_YAW = -1（需要修改）
# 同理对 DIR_PITCH：目标在画面下方（err_y > 0）时，云台应向下倾。
DIR_YAW   = 1    # [CALIBRATION NEEDED]
DIR_PITCH = 1    # [CALIBRATION NEEDED]

# [CALIBRATION NEEDED] 云台角度限位（弧度）
# ⚠️ 当前值 ±1.57 rad（≈ ±90°）仅为临时占位值！
# 过大的限位可能导致线缆缠绕或电机堵转损坏。
# 测试方法：手动发送角度指令（用 motor_TM.py），观察云台实际能安全运动的范围，
# 然后将此处的值设为比安全边界还小约 0.1 rad 的值。
MAX_YAW_ANGLE   =  2.0    # 实测：正右极限
MIN_YAW_ANGLE   = -1.4   # 实测：正左极限
MAX_PITCH_ANGLE =  2.8   # 实测：最低/平视（数值最大）
MIN_PITCH_ANGLE =  2.4   # 实测：最高仰角（数值最小）

# 工作起始位置（Arduino init 后电机在机械零点，需主动移到此处）
YAW_INIT   = 0.3    # 实测：正前方对应的 yaw 角
PITCH_INIT = 2.8    # 实测：平视对应的 pitch 角

# 对准死区（像素）：检测框中心距画面中心 < DEADZONE 时触发定位
DEADZONE = 30

# 相邻两次缺陷记录之间的最短间隔（秒），防止对同一位置重复记录
PUBLISH_COOLDOWN = 5.0

# 图像保存目录
SAVE_DIR = os.path.expanduser("~/defect_records")


# ══════════════════════════════════════════════════════════════════════════════
#  ④ 坐标变换外参  [CALIBRATION NEEDED - 全部需要测量/标定]
# ══════════════════════════════════════════════════════════════════════════════

# ─── 4-A. 云台底座在 AGV base_footprint 坐标系中的安装位置（平移，单位：米）───
#
# base_footprint 坐标系约定（ROS 标准）：
#   X 轴 → AGV 前进方向
#   Y 轴 → AGV 左侧
#   Z 轴 → 垂直向上
#
# 当前估算依据：用户反映相机约在车头中心，高于地面约 0.20 m。
# 请用卷尺实际测量后替换以下三个值。
GIMBAL_MOUNT_X = 0.0    # [CALIBRATION NEEDED] 云台底座前后偏移（m），正=偏前
GIMBAL_MOUNT_Y = 0.0    # [CALIBRATION NEEDED] 云台底座左右偏移（m），正=偏左
GIMBAL_MOUNT_Z = 0.20   # [CALIBRATION NEEDED] 云台底座距地面高度（m）
                         # 初始估算值 0.20 m，请测量后修正

# ─── 4-B. 云台底座相对 base_footprint 的固定旋转（弧度）───────────────────────
#
# 如果云台安装时其正前方与 AGV 前进方向完全一致，则三个值都是 0。
# 如果安装时有侧向偏转（例如云台侧装在车体上），则需要在 GIMBAL_MOUNT_YAW 中补偿。
#
# 测试方法：
#   1. 让云台回到零位（T0.0），观察相机光轴方向
#   2. 用量角器或激光笔测量相机光轴与 AGV 前进方向的夹角
#   3. 若光轴偏左 θ° → GIMBAL_MOUNT_YAW = +θ*π/180
#      若光轴偏右 θ° → GIMBAL_MOUNT_YAW = -θ*π/180
GIMBAL_MOUNT_YAW   = 0.0   # [CALIBRATION NEEDED] 云台朝向偏转（绕 Z 轴，弧度）
GIMBAL_MOUNT_PITCH = 0.0   # [CALIBRATION NEEDED] 云台仰角偏转（绕 Y 轴，弧度）
                             # 若云台零位时相机略微向下倾斜，此处填负值补偿
GIMBAL_MOUNT_ROLL  = 0.0   # [CALIBRATION NEEDED] 云台滚转偏转（绕 X 轴，弧度）
                             # 通常为 0，除非安装时横向倾斜

# ─── 4-C. 相机光心相对云台旋转中心的固定偏移（云台头坐标系，单位：米）─────────
#
# 即 RealSense D435 的光学中心与云台两轴交点之间的距离。
# 若认为相机基本安装在旋转中心上，可先保持全 0，观察定位误差后再补偿。
#
# 坐标系约定（云台头局部坐标系，此时 yaw=0, pitch=0）：
#   X 轴 → 相机正前方
#   Y 轴 → 相机左方
#   Z 轴 → 相机上方
CAM_OFFSET_X = 0.0   # [CALIBRATION NEEDED] 相机光心前向偏移（m）
CAM_OFFSET_Y = 0.0   # [CALIBRATION NEEDED] 相机光心横向偏移（m）
CAM_OFFSET_Z = 0.0   # [CALIBRATION NEEDED] 相机光心垂直偏移（m）

# ─── 4-D. Yaw 轴旋转方向符号 ──────────────────────────────────────────────────
#
# 约定：SimpleFOC 电机的 current_yaw 正值 → 相机在空间中向哪个方向转？
#
# 测试方法：
#   1. 运行 motor_TM.py，发送 T0.5（+0.5 rad）
#   2. 从 AGV 上方俯视，观察云台旋转方向：
#      逆时针（左转）→ 符合 ROS Z 轴右手定则 → YAW_ROT_SIGN = +1
#      顺时针（右转）→ 与 ROS 约定相反         → YAW_ROT_SIGN = -1
YAW_ROT_SIGN = -1    # [CALIBRATION NEEDED] 默认假设正 yaw = 向右（顺时针俯视）

# ─── 4-E. Pitch 轴旋转方向符号 ────────────────────────────────────────────────
#
# 测试方法：
#   1. 发送正 pitch 角（T0.5）到 pitch 电机
#   2. 观察相机抬头（仰角增加）还是低头（俯角增加）：
#      低头（俯视）→ PITCH_ROT_SIGN = +1（正 pitch = 向下）
#      抬头（仰视）→ PITCH_ROT_SIGN = -1
PITCH_ROT_SIGN = 1   # [CALIBRATION NEEDED] 默认假设正 pitch = 低头

# ─── 4-F. 旋转轴方向 ──────────────────────────────────────────────────────────
#
# 通常云台：Yaw 绕竖直轴（Z 轴）旋转，Pitch 绕横轴（Y 轴）旋转。
# 如果机械设计不同，可在此修改（可选值: 'X', 'Y', 'Z'）。
YAW_AXIS   = 'Z'   # [CALIBRATION NEEDED] Yaw 轴旋转轴
PITCH_AXIS = 'Y'   # [CALIBRATION NEEDED] Pitch 轴旋转轴


# ══════════════════════════════════════════════════════════════════════════════
#  CUDA 基础函数（与 visual_servoing_test_ver2.py 相同）
# ══════════════════════════════════════════════════════════════════════════════

_libcudart = ctypes.CDLL(LIBCUDART_PATH)
_cudaMemcpyHostToDevice = 1
_cudaMemcpyDeviceToHost = 2


def _cuda_check(status, msg):
    if status != 0:
        raise RuntimeError(f"{msg} (cudaError={status})")


def _cuda_malloc(nbytes):
    ptr = ctypes.c_void_p()
    _cuda_check(_libcudart.cudaMalloc(ctypes.byref(ptr), nbytes), "cudaMalloc")
    return ptr.value


def _cuda_free(ptr):
    _cuda_check(_libcudart.cudaFree(ctypes.c_void_p(ptr)), "cudaFree")


def _cuda_memcpy_htod(dst, src_host):
    _cuda_check(
        _libcudart.cudaMemcpy(ctypes.c_void_p(dst), ctypes.c_void_p(src_host.ctypes.data),
                              src_host.nbytes, _cudaMemcpyHostToDevice),
        "cudaMemcpy H2D")


def _cuda_memcpy_dtoh(dst_host, src):
    _cuda_check(
        _libcudart.cudaMemcpy(ctypes.c_void_p(dst_host.ctypes.data), ctypes.c_void_p(src),
                              dst_host.nbytes, _cudaMemcpyDeviceToHost),
        "cudaMemcpy D2H")


# ══════════════════════════════════════════════════════════════════════════════
#  图像预处理工具（与 visual_servoing_test_ver2.py 相同）
# ══════════════════════════════════════════════════════════════════════════════

def _letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = im.shape[:2]
    nh, nw = new_shape
    r = min(nh / h, nw / w)
    resized = (int(round(w * r)), int(round(h * r)))
    im2 = cv2.resize(im, resized, interpolation=cv2.INTER_LINEAR)
    padw = nw - resized[0]
    padh = nh - resized[1]
    left  = int(round(padw / 2 - 0.1))
    right = int(round(padw / 2 + 0.1))
    top   = int(round(padh / 2 - 0.1))
    bot   = int(round(padh / 2 + 0.1))
    return cv2.copyMakeBorder(im2, top, bot, left, right,
                               cv2.BORDER_CONSTANT, value=color), r, (left, top)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _xywh2xyxy(x):
    y = x.copy()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def _box_iou(box1, box2):
    x1 = np.maximum(box1[0], box2[:, 0])
    y1 = np.maximum(box1[1], box2[:, 1])
    x2 = np.minimum(box1[2], box2[:, 2])
    y2 = np.minimum(box1[3], box2[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    return inter / (a1 + a2 - inter + 1e-9)


def _nms(boxes, scores, iou_thres):
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        ious = _box_iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_thres]
    return keep


# ══════════════════════════════════════════════════════════════════════════════
#  TRT 推理器（与 visual_servoing_test_ver2.py 相同）
# ══════════════════════════════════════════════════════════════════════════════

class TRTRunner:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load TRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()

        self.in_idxs, self.out_idxs = [], []
        for i in range(self.engine.num_bindings):
            (self.in_idxs if self.engine.binding_is_input(i) else self.out_idxs).append(i)

        self.in_idx  = self.in_idxs[0]
        self.in_name = self.engine.get_binding_name(self.in_idx)
        in_shape = tuple(self.context.get_binding_shape(self.in_idx))
        if -1 in in_shape:
            self.context.set_binding_shape(self.in_idx, (1, 3, INPUT_H, INPUT_W))

        self.h = {}
        self.d = {}
        self.bindings = [0] * self.engine.num_bindings
        for i in range(self.engine.num_bindings):
            name  = self.engine.get_binding_name(i)
            shape = tuple(self.context.get_binding_shape(i))
            host  = np.empty(shape, dtype=np.float32)
            dev   = _cuda_malloc(host.nbytes)
            self.h[name] = host
            self.d[name] = dev
            self.bindings[i] = dev
        self.out_names = [self.engine.get_binding_name(i) for i in self.out_idxs]

    def infer(self, inp):
        np.copyto(self.h[self.in_name], inp)
        _cuda_memcpy_htod(self.d[self.in_name], self.h[self.in_name])
        self.context.execute_v2(self.bindings)
        outs = {}
        for name in self.out_names:
            _cuda_memcpy_dtoh(self.h[name], self.d[name])
            outs[name] = self.h[name].copy()
        return outs

    def __del__(self):
        try:
            for ptr in self.d.values():
                _cuda_free(ptr)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
#  后处理函数（与 visual_servoing_test_ver2.py 相同）
# ══════════════════════════════════════════════════════════════════════════════

def _postprocess_seg(outs, orig_shape, r, pad):
    det = proto = None
    for v in outs.values():
        if v.ndim == 3 and v.shape[0] == 1 and v.shape[1] >= 6 and v.shape[2] >= 1000:
            det = v
        elif v.ndim == 4 and v.shape[0] == 1:
            proto = v
    if det is None or proto is None:
        return [], [], [], []

    det    = det[0].transpose(1, 0)
    boxes  = det[:, 0:4]
    scores = det[:, 4]
    mcoefs = det[:, 5:37]

    keep = scores > CONF_SEG
    if not np.any(keep):
        return [], [], [], []
    boxes, scores, mcoefs = boxes[keep], scores[keep], mcoefs[keep]
    boxes = _xywh2xyxy(boxes)
    ki    = _nms(boxes, scores, IOU_THRESH)
    boxes, scores, mcoefs = boxes[ki], scores[ki], mcoefs[ki]

    proto    = proto[0]
    nm, mh, mw = proto.shape
    proto_flat = proto.reshape(nm, -1)
    padw, padh = pad
    h0, w0 = orig_shape

    boxes_out, masks_out = [], []
    for i in range(boxes.shape[0]):
        x1, y1, x2, y2 = boxes[i]
        boxes_out.append([
            max(0, min(w0 - 1, (x1 - padw) / r)),
            max(0, min(h0 - 1, (y1 - padh) / r)),
            max(0, min(w0 - 1, (x2 - padw) / r)),
            max(0, min(h0 - 1, (y2 - padh) / r)),
        ])
        m = _sigmoid(mcoefs[i] @ proto_flat).reshape(mh, mw).astype(np.float32)
        m = cv2.resize(m, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
        xs, ys = int(padw), int(padh)
        xe, ye = int(padw + r * w0), int(padh + r * h0)
        m_orig = cv2.resize(m[ys:ye, xs:xe], (w0, h0), interpolation=cv2.INTER_LINEAR)
        masks_out.append((m_orig > MASK_THRESH).astype(np.uint8))

    return boxes_out, scores.tolist(), [0] * len(scores), masks_out


def _postprocess_det(outs, orig_shape, r, pad):
    out = list(outs.values())[0]
    if out.ndim != 3 or out.shape[0] != 1:
        return [], [], []
    _, a, b = out.shape
    pred = out[0].transpose(1, 0) if b >= a else out[0]
    N    = pred.shape[0]

    boxes = pred[:, 0:4]
    rest  = pred[:, 4:]
    if rest.shape[1] >= 2:
        obj = rest[:, 0]
        cls = rest[:, 1:]
        cid = np.argmax(cls, axis=1)
        scores = obj * cls[np.arange(N), cid]
    else:
        cid    = np.zeros(N, dtype=np.int32)
        scores = rest[:, 0] if rest.shape[1] >= 1 else np.zeros(N)

    keep = scores > CONF_DET
    if not np.any(keep):
        return [], [], []
    boxes, scores, cid = _xywh2xyxy(boxes[keep]), scores[keep], cid[keep]

    padw, padh = pad
    h0, w0 = orig_shape
    final_boxes, final_scores, final_cls = [], [], []
    for c in np.unique(cid):
        idx = np.where(cid == c)[0]
        ki  = _nms(boxes[idx], scores[idx], IOU_THRESH)
        for k in ki:
            x1, y1, x2, y2 = boxes[idx[k]]
            final_boxes.append([
                max(0, min(w0 - 1, (x1 - padw) / r)),
                max(0, min(h0 - 1, (y1 - padh) / r)),
                max(0, min(w0 - 1, (x2 - padw) / r)),
                max(0, min(h0 - 1, (y2 - padh) / r)),
            ])
            final_scores.append(float(scores[idx[k]]))
            final_cls.append(int(c))
    return final_boxes, final_scores, final_cls


# ══════════════════════════════════════════════════════════════════════════════
#  坐标变换辅助函数
# ══════════════════════════════════════════════════════════════════════════════

def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float64)


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float64)


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)


def _rot_by_axis(axis, angle):
    """按轴名称（'X'/'Y'/'Z'）和角度生成旋转矩阵。"""
    return {'X': _rot_x, 'Y': _rot_y, 'Z': _rot_z}[axis](angle)


# RealSense 光学坐标系 → ROS 标准坐标系（X前, Y左, Z上）变换矩阵
# RealSense 光学系: X 右, Y 下, Z 前（正前方）
# ROS 标准系:      X 前, Y 左, Z 上
# 推导：new_X=old_Z, new_Y=-old_X, new_Z=-old_Y
_R_OPT_TO_ROS = np.array([
    [ 0,  0,  1],   # new X (前) = old Z
    [-1,  0,  0],   # new Y (左) = -old X
    [ 0, -1,  0],   # new Z (上) = -old Y
], dtype=np.float64)

# 云台底座的固定安装旋转矩阵（在程序启动时根据参数计算一次）
def _build_mount_rotation():
    """将 GIMBAL_MOUNT_YAW/PITCH/ROLL 合成为一个旋转矩阵。"""
    return (_rot_z(GIMBAL_MOUNT_YAW)
            @ _rot_y(GIMBAL_MOUNT_PITCH)
            @ _rot_x(GIMBAL_MOUNT_ROLL))


_R_MOUNT = _build_mount_rotation()   # 启动时计算一次，之后不变


def transform_to_map(point_cam_optical, current_yaw, current_pitch, tf_listener):
    """
    将 RealSense 光学坐标系中的 3D 点变换到 ROS map 坐标系。

    变换步骤：
      [相机光学系] → [ROS相机系] → [云台头系(加偏移)] →
      [应用pitch旋转] → [应用yaw旋转] → [base_footprint系] → [map系]

    Parameters
    ----------
    point_cam_optical : list/array [Xc, Yc, Zc]
        RealSense rs2_deproject_pixel_to_point 的输出，单位：米
    current_yaw : float
        当前云台 yaw 角（弧度），由串口指令追踪值
    current_pitch : float
        当前云台 pitch 角（弧度），由串口指令追踪值
    tf_listener : tf.TransformListener

    Returns
    -------
    np.ndarray shape (3,) : map 坐标 [x, y, z]（米），或 None（TF 不可用时）
    """
    P = np.array(point_cam_optical, dtype=np.float64)

    # ── Step 1: 相机光学坐标系 → ROS 标准坐标系 ──────────────────────────────
    P = _R_OPT_TO_ROS @ P

    # ── Step 2: 加上相机在云台头上的固定安装偏移 ─────────────────────────────
    # [CALIBRATION NEEDED] 见 CAM_OFFSET_X/Y/Z 说明
    P = P + np.array([CAM_OFFSET_X, CAM_OFFSET_Y, CAM_OFFSET_Z])

    # ── Step 3: 应用 Pitch 旋转 ────────────────────────────────────────────────
    # Pitch 电机先旋转（离相机更近的轴），Yaw 后旋转（根部轴）。
    # [CALIBRATION NEEDED] PITCH_ROT_SIGN, PITCH_AXIS
    R_pitch = _rot_by_axis(PITCH_AXIS, PITCH_ROT_SIGN * current_pitch)
    P = R_pitch @ P

    # ── Step 4: 应用 Yaw 旋转 ─────────────────────────────────────────────────
    # [CALIBRATION NEEDED] YAW_ROT_SIGN, YAW_AXIS
    R_yaw = _rot_by_axis(YAW_AXIS, YAW_ROT_SIGN * current_yaw)
    P = R_yaw @ P

    # ── Step 5: 云台底座 → base_footprint ─────────────────────────────────────
    # 先旋转（云台底座朝向补偿），再加平移（云台安装位置）
    # [CALIBRATION NEEDED] 见 _R_MOUNT 和 GIMBAL_MOUNT_X/Y/Z 说明
    P = _R_MOUNT @ P + np.array([GIMBAL_MOUNT_X, GIMBAL_MOUNT_Y, GIMBAL_MOUNT_Z])

    # ── Step 6: base_footprint → map（通过 ROS TF）────────────────────────────
    try:
        (trans, rot) = tf_listener.lookupTransform(
            "map", "base_footprint", rospy.Time(0))
        T = tf.transformations.quaternion_matrix(rot)   # 4×4 齐次矩阵
        P_map = T[:3, :3] @ P + np.array(trans)
        return P_map
    except (tf.LookupException, tf.ConnectivityException,
            tf.ExtrapolationException) as e:
        rospy.logwarn("defect_localizer: TF map←base_footprint failed: %s", e)
        return None


def get_robust_depth(depth_frame, u, v):
    """
    在像素 (u, v) 周围 DEPTH_SAMPLE_RADIUS 像素半径内取深度中值。
    滤除无效（0）和超出范围的点，提高噪声鲁棒性。

    Returns
    -------
    float 深度值（米），或 None（无有效读数时）
    """
    h = depth_frame.get_height()
    w = depth_frame.get_width()
    r = DEPTH_SAMPLE_RADIUS

    depths = []
    for row in range(max(0, v - r), min(h, v + r + 1)):
        for col in range(max(0, u - r), min(w, u + r + 1)):
            d = depth_frame.get_distance(col, row)
            if DEPTH_MIN_M < d < DEPTH_MAX_M:
                depths.append(d)

    if not depths:
        return None
    return float(np.median(depths))


# ══════════════════════════════════════════════════════════════════════════════
#  主节点类
# ══════════════════════════════════════════════════════════════════════════════

class DefectLocalizerNode:

    _BOX_COLORS = {
        "erosion": (0, 255, 0),   # 绿色
        "crack":   (0, 0, 255),   # 红色
    }

    def __init__(self):
        rospy.init_node("defect_localizer", anonymous=False)
        rospy.loginfo("defect_localizer: initialising node...")

        os.makedirs(SAVE_DIR, exist_ok=True)
        rospy.loginfo("defect_localizer: saving images to %s", SAVE_DIR)

        # TF 监听器（用于 base_footprint → map 变换）
        self.tf_listener = tf.TransformListener()

        # 发布者：将 map 坐标系下的缺陷位置发送给 nav_test_ver2.py（AGV 端）
        self.defect_pub = rospy.Publisher(
            "/defect_candidates", PoseStamped, queue_size=10)

        # 当前云台角度（弧度）—— 与 YAW_INIT/PITCH_INIT 一致（init 后会移到此处）
        self.current_yaw   = YAW_INIT
        self.current_pitch = PITCH_INIT
        self._last_publish_time = 0.0

        # 组件初始化
        self._init_gimbal()
        self._init_trt()
        self._init_realsense()

        rospy.loginfo("defect_localizer: all components ready.")

    # ──────────────────────────────────────────── 云台初始化 ─────────────────
    def _init_gimbal(self):
        rospy.loginfo("defect_localizer: scanning for STM32 serial ports...")
        ports = [p.device for p in serial.tools.list_ports.comports()
                 if "ACM" in p.device or "USB" in p.device]

        self.ser_yaw   = None
        self.ser_pitch = None

        if not ports:
            rospy.logwarn("defect_localizer: no STM32 found — gimbal servo disabled.")
            return

        rospy.loginfo("defect_localizer: found %d port(s): %s", len(ports), ports)

        # [CALIBRATION NEEDED] 串口分配
        # 当前规则：ports[0] → yaw 轴，ports[1] → pitch 轴。
        # 若上电后发现 yaw/pitch 控制了错误的轴，只需交换下面两行的端口赋值即可。
        # 判断依据：用 motor_TM.py 对每个端口单独发 T0.5，观察哪个轴动了。
        if len(ports) >= 1:
            self.ser_yaw = serial.Serial(ports[0], 115200, timeout=0.1)
            rospy.loginfo("defect_localizer: yaw  port → %s  [CALIBRATION: confirm correct axis]",
                          ports[0])
        if len(ports) >= 2:
            self.ser_pitch = serial.Serial(ports[1], 115200, timeout=0.1)
            rospy.loginfo("defect_localizer: pitch port → %s  [CALIBRATION: confirm correct axis]",
                          ports[1])
        else:
            rospy.logwarn("defect_localizer: only 1 STM32 found — pitch axis disabled.")

        rospy.loginfo("defect_localizer: waiting 5 s for FOC alignment...")
        rospy.sleep(5.0)
        if self.ser_yaw:   self.ser_yaw.reset_input_buffer()
        if self.ser_pitch: self.ser_pitch.reset_input_buffer()

        # Arduino init 完成后，移至工作起始位置（正前方/平视）
        rospy.loginfo("defect_localizer: moving gimbal to working start position...")
        if self.ser_yaw:
            self.ser_yaw.write(f"T{YAW_INIT:.4f}\n".encode())
        if self.ser_pitch:
            self.ser_pitch.write(f"T{PITCH_INIT:.4f}\n".encode())
        rospy.sleep(2.0)  # 等待电机到位
        rospy.loginfo("defect_localizer: gimbal ready.")

    def _send_gimbal(self, yaw_rad, pitch_rad):
        """
        向云台发送角度指令并更新内部状态追踪。
        SimpleFOC Commander 格式: "T{angle}\\n"
        """
        if self.ser_yaw:
            self.ser_yaw.write(f"T{yaw_rad:.4f}\n".encode())
        if self.ser_pitch:
            self.ser_pitch.write(f"T{pitch_rad:.4f}\n".encode())
        # 用发送的目标角追踪当前角度（FOC 为闭环位置控制，目标角 ≈ 实际角）
        self.current_yaw   = yaw_rad
        self.current_pitch = pitch_rad

    def _return_to_zero(self):
        """节点退出时将云台归回工作起始位置（正前方/平视），而非机械零点。"""
        rospy.loginfo("defect_localizer: returning gimbal to start position...")
        self._send_gimbal(YAW_INIT, PITCH_INIT)
        rospy.sleep(1.0)

    # ──────────────────────────────────────────── TRT 初始化 ─────────────────
    def _init_trt(self):
        rospy.loginfo("defect_localizer: loading TRT engines...")
        self.seg_runner = TRTRunner(ENGINE_SEG)
        self.det_runner = TRTRunner(ENGINE_DET)
        rospy.loginfo("defect_localizer: TRT engines loaded.")

    # ──────────────────────────────────────────── RealSense 初始化 ───────────
    def _init_realsense(self):
        rospy.loginfo("defect_localizer: starting RealSense D435...")
        self.pipeline = rs.pipeline()
        cfg = rs.config()

        # 彩色流（用于 TRT 推理和可视化）
        cfg.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, COLOR_FPS)

        # 深度流（新增！用于 3D 坐标反投影）
        # 使用与彩色流相同的分辨率，方便 rs.align 对齐
        cfg.enable_stream(rs.stream.depth, COLOR_W, COLOR_H, rs.format.z16, COLOR_FPS)

        self.rs_profile = self.pipeline.start(cfg)

        # 深度帧对齐到彩色帧
        # 对齐后：depth_frame.get_distance(u, v) 与 color_frame 的 (u, v) 像素对应同一空间点
        self.align = rs.align(rs.stream.color)

        # 获取彩色流相机内参（fx, fy, cx, cy），用于反投影
        color_stream = self.rs_profile.get_stream(rs.stream.color)
        self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        rospy.loginfo(
            "defect_localizer: color intrinsics — fx=%.2f fy=%.2f cx=%.2f cy=%.2f",
            self.intrinsics.fx, self.intrinsics.fy,
            self.intrinsics.ppx, self.intrinsics.ppy)
        rospy.loginfo("defect_localizer: RealSense started.")

    # ──────────────────────────────────────────── TRT 推理 ───────────────────
    def _run_inference(self, color_image):
        """
        对彩色帧运行 TRT 推理。
        优先返回 crack（检测模型），如无则返回 erosion（分割模型）。

        Returns
        -------
        ((x1,y1,x2,y2), confidence, class_name) 或 None
        """
        h0, w0 = color_image.shape[:2]
        img_lb, r, pad = _letterbox(color_image, (INPUT_H, INPUT_W))
        inp = np.ascontiguousarray(
            np.transpose(
                cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0,
                (2, 0, 1))[None],
            dtype=np.float32)

        outs_seg = self.seg_runner.infer(inp)
        outs_det = self.det_runner.infer(inp)

        c_boxes, c_scores, _ = _postprocess_det(outs_det, (h0, w0), r, pad)
        e_boxes, e_scores, _, _ = _postprocess_seg(outs_seg, (h0, w0), r, pad)

        if c_boxes:
            b = c_boxes[0]
            return (b[0], b[1], b[2], b[3]), c_scores[0], "crack"
        if e_boxes:
            b = e_boxes[0]
            return (b[0], b[1], b[2], b[3]), e_scores[0], "erosion"
        return None

    # ──────────────────────────────────────────── 发布缺陷 ───────────────────
    def _publish_defect(self, map_xyz, class_name, confidence):
        msg = PoseStamped()
        msg.header.stamp    = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.pose.position.x = float(map_xyz[0])
        msg.pose.position.y = float(map_xyz[1])
        msg.pose.position.z = float(map_xyz[2])
        # 方向设为单位四元数（朝向未知时的占位值）
        msg.pose.orientation.w = 1.0
        self.defect_pub.publish(msg)
        rospy.loginfo(
            "defect_localizer: PUBLISHED [%s conf=%.2f] map=(%.3f, %.3f, %.3f)",
            class_name, confidence, map_xyz[0], map_xyz[1], map_xyz[2])

    # ──────────────────────────────────────────── 主循环 ─────────────────────
    def run(self):
        CENTER_X = COLOR_W / 2.0
        CENTER_Y = COLOR_H / 2.0

        rospy.loginfo("defect_localizer: entering main loop. Press q or Ctrl-C to exit.")
        rospy.loginfo("defect_localizer: ⚠️  All [CALIBRATION NEEDED] parameters are "
                      "placeholders — verify before trusting published coordinates.")

        try:
            while not rospy.is_shutdown():
                # ── 1. 获取对齐后的彩色 + 深度帧 ──────────────────────────────
                frames   = self.pipeline.wait_for_frames(timeout_ms=1000)
                aligned  = self.align.process(frames)
                c_frame  = aligned.get_color_frame()
                d_frame  = aligned.get_depth_frame()
                if not c_frame or not d_frame:
                    continue

                color_image  = np.asanyarray(c_frame.get_data())
                display      = color_image.copy()
                depth_text   = ""

                # ── 2. TRT 推理 ──────────────────────────────────────────────
                result = self._run_inference(color_image)

                if result is not None:
                    (x1, y1, x2, y2), conf, cls_name = result
                    color = self._BOX_COLORS.get(cls_name, (0, 255, 255))

                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    err_x = cx - CENTER_X
                    err_y = cy - CENTER_Y

                    # ── 3. 视觉伺服：未对准时调整云台 ───────────────────────
                    if abs(err_x) > DEADZONE or abs(err_y) > DEADZONE:
                        new_yaw   = self.current_yaw   + err_x * KP_YAW   * DIR_YAW
                        new_pitch = self.current_pitch + err_y * KP_PITCH * DIR_PITCH
                        new_yaw   = max(MIN_YAW_ANGLE,   min(MAX_YAW_ANGLE,   new_yaw))
                        new_pitch = max(MIN_PITCH_ANGLE, min(MAX_PITCH_ANGLE, new_pitch))
                        self._send_gimbal(new_yaw, new_pitch)

                    else:
                        # ── 4. 已对准 → 坐标定位 ───────────────────────────
                        now = time.time()
                        if now - self._last_publish_time > PUBLISH_COOLDOWN:
                            u = int(round(cx))
                            v = int(round(cy))

                            # 4a. 鲁棒深度读取（中值滤波，单位：米）
                            depth_m = get_robust_depth(d_frame, u, v)

                            if depth_m is None:
                                rospy.logwarn(
                                    "defect_localizer: no valid depth at (%d,%d) "
                                    "— skipping this detection.", u, v)
                            else:
                                depth_text = f"d={depth_m:.2f}m"

                                # 4b. 像素 + 深度 → 相机光学坐标系 3D 点
                                # rs2_deproject_pixel_to_point 输出单位：米
                                # 注意：u, v 要用 float，并且已经是对齐后彩色帧的像素坐标
                                point_cam = rs.rs2_deproject_pixel_to_point(
                                    self.intrinsics, [float(u), float(v)], depth_m)
                                rospy.logdebug(
                                    "defect_localizer: camera frame P=[%.3f, %.3f, %.3f] m",
                                    *point_cam)

                                # 4c. 完整变换链 → map 坐标
                                map_xyz = transform_to_map(
                                    point_cam,
                                    self.current_yaw,
                                    self.current_pitch,
                                    self.tf_listener)

                                if map_xyz is not None:
                                    # 4d. 发布 /defect_candidates
                                    self._publish_defect(map_xyz, cls_name, conf)

                                    # 4e. 保存图像（原始彩色帧）
                                    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    img = os.path.join(SAVE_DIR,
                                                        f"{cls_name}_{ts}.jpg")
                                    cv2.imwrite(img, color_image)
                                    rospy.loginfo(
                                        "defect_localizer: image saved → %s", img)

                                    self._last_publish_time = now

                    # ── 5. 可视化叠加 ─────────────────────────────────────
                    cv2.rectangle(display,
                                  (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    label = f"{cls_name} {conf:.2f} {depth_text}"
                    cv2.putText(display, label,
                                (int(x1), max(int(y1) - 10, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.circle(display, (int(cx), int(cy)), 5, color, -1)

                # 画面中心准星
                cxi, cyi = int(CENTER_X), int(CENTER_Y)
                cv2.line(display, (cxi - 20, cyi), (cxi + 20, cyi), (0, 255, 0), 2)
                cv2.line(display, (cxi, cyi - 20), (cxi, cyi + 20), (0, 255, 0), 2)

                # 右上角显示当前云台角度（调试用）
                cv2.putText(display,
                            f"yaw={math.degrees(self.current_yaw):.1f}  "
                            f"pitch={math.degrees(self.current_pitch):.1f}",
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (200, 200, 0), 1)

                cv2.imshow("Defect Localizer", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    rospy.loginfo("defect_localizer: q pressed, exiting.")
                    break

        except KeyboardInterrupt:
            rospy.loginfo("defect_localizer: Ctrl-C received.")
        finally:
            self._return_to_zero()
            self.pipeline.stop()
            if self.ser_yaw   and self.ser_yaw.is_open:   self.ser_yaw.close()
            if self.ser_pitch and self.ser_pitch.is_open: self.ser_pitch.close()
            cv2.destroyAllWindows()
            rospy.loginfo("defect_localizer: shutdown complete.")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    try:
        DefectLocalizerNode().run()
    except rospy.ROSInterruptException:
        pass