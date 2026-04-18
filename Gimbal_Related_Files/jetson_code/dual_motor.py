#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Jetson 双 Arduino FOC 电机控制脚本
架构：两个 Arduino 各自通过 USB 串口独立连接 Jetson，各自运行 SimpleFOC 单电机闭环控制。
      每个 Arduino 的 Commander 只监听 'T' 命令（目标角度）。
      本脚本打开两个串口，分别控制电机 A 和电机 B。
"""

import serial
import serial.tools.list_ports
import time

BAUD_RATE = 115200
INIT_WAIT = 5  # 等待 SimpleFOC 完成对齐的时间（秒）


def find_arduino_ports():
    """扫描并返回所有找到的 ACM/USB 串口设备列表"""
    print("正在扫描可用的串口设备...")
    ports = [p.device for p in serial.tools.list_ports.comports()
             if 'ACM' in p.device or 'USB' in p.device]

    if len(ports) == 0:
        print("未发现任何串口设备！请检查 USB 连线。")
    elif len(ports) == 1:
        print(f"警告：只发现一个串口 {ports[0]}，双电机控制需要两个串口。")
    else:
        print(f"找到串口：{ports}")

    return ports


def open_serial(port):
    """打开指定串口，失败时返回 None"""
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=0.5)
        print(f"  已打开串口: {port}")
        return ser
    except Exception as e:
        print(f"  无法打开串口 {port}: {e}")
        return None


def wait_for_init(ser, label):
    """等待 Arduino 完成 FOC 对齐，打印初始化日志"""
    print(f"\n[{label}] 等待 SimpleFOC 校准（约 {INIT_WAIT} 秒）...")
    time.sleep(INIT_WAIT)
    while ser.in_waiting:
        msg = ser.readline().decode('utf-8', errors='ignore').strip()
        if msg:
            print(f"  [{label} Init] {msg}")
    ser.reset_input_buffer()
    print(f"  [{label}] 校准完成。")


def send_angle(ser, label, angle):
    """向指定串口发送目标角度指令，并打印回音"""
    cmd = f"T{angle}\n"
    ser.write(cmd.encode('utf-8'))
    print(f"  [{label}] 发送: {cmd.strip()}")
    time.sleep(0.05)
    while ser.in_waiting:
        resp = ser.readline().decode('utf-8', errors='ignore').strip()
        if resp:
            print(f"  [{label} 回音] {resp}")


def close_all(ser_a, ser_b):
    """安全归零并关闭两个串口"""
    for ser, label in [(ser_a, "电机A"), (ser_b, "电机B")]:
        if ser and ser.is_open:
            ser.write(b"T0\n")
    time.sleep(0.1)
    for ser, label in [(ser_a, "电机A"), (ser_b, "电机B")]:
        if ser and ser.is_open:
            ser.close()
            print(f"  [{label}] 串口已关闭。")


def main():
    ports = find_arduino_ports()

    if len(ports) < 2:
        print("需要至少两个串口才能控制双电机，程序退出。")
        return

    # 让用户确认哪个串口对应哪个电机，避免接错
    print("\n检测到以下串口：")
    for i, p in enumerate(ports):
        print(f"  [{i}] {p}")

    print("\n默认：第 0 个串口 -> 电机A，第 1 个串口 -> 电机B")
    confirm = input("是否使用默认分配？(y/n，回车默认 y): ").strip().lower()

    if confirm == 'n':
        try:
            idx_a = int(input(f"  请输入电机 A 对应的串口编号 (0~{len(ports)-1}): ").strip())
            idx_b = int(input(f"  请输入电机 B 对应的串口编号 (0~{len(ports)-1}): ").strip())
            port_a, port_b = ports[idx_a], ports[idx_b]
        except (ValueError, IndexError):
            print("输入无效，使用默认分配。")
            port_a, port_b = ports[0], ports[1]
    else:
        port_a, port_b = ports[0], ports[1]

    print(f"\n电机 A -> {port_a}")
    print(f"电机 B -> {port_b}")

    ser_a = open_serial(port_a)
    ser_b = open_serial(port_b)

    if not ser_a or not ser_b:
        print("串口打开失败，程序退出。")
        close_all(ser_a, ser_b)
        return

    try:
        # 两个电机并行等待初始化（两个 Arduino 同时上电对齐）
        wait_for_init(ser_a, "电机A")
        wait_for_init(ser_b, "电机B")

        print("\n双电机校准完成，控制链路已打通！")
        print("-" * 40)
        print("请选择测试模式：")
        print("1: 手动模式  — 分别输入两个电机的目标角度")
        print("2: 自动摇摆  — 双电机反向在 -1.57 ~ 1.57 弧度间摇摆")
        print("q: 退出")
        print("-" * 40)

        mode = input("请输入选择 (1 / 2 / q): ").strip()

        if mode == '1':
            print("\n已进入手动模式。每次依次输入电机 A 和电机 B 的目标角度（弧度）。")
            print("输入 'q' 退出。\n")
            while True:
                val_a = input("电机 A 目标角度（弧度，或 q 退出）: ").strip()
                if val_a.lower() == 'q':
                    break
                val_b = input("电机 B 目标角度（弧度，或 q 退出）: ").strip()
                if val_b.lower() == 'q':
                    break
                try:
                    angle_a = float(val_a)
                    angle_b = float(val_b)
                    send_angle(ser_a, "电机A", angle_a)
                    send_angle(ser_b, "电机B", angle_b)
                except ValueError:
                    print("请输入有效的数字！")

        elif mode == '2':
            print("\n已进入自动双轴摇摆模式。电机 A 和 B 反向转动。按 Ctrl+C 停止。")
            target_angles = [0.0, 1.57, 0.0, -1.57]
            idx = 0
            while True:
                angle = target_angles[idx % len(target_angles)]
                send_angle(ser_a, "电机A",  angle)
                send_angle(ser_b, "电机B", -angle)
                idx += 1
                time.sleep(2)

        elif mode.lower() == 'q':
            print("退出测试。")

    except KeyboardInterrupt:
        print("\n收到中断信号，程序停止。")
    except Exception as e:
        print(f"\n发生错误: {e}")
    finally:
        close_all(ser_a, ser_b)
        print("程序已安全退出。")


if __name__ == "__main__":
    main()