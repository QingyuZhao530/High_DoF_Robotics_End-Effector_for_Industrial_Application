#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Jetson 单 STM32 FOC 电机控制测试脚本!
功能：自动寻找 STM32 串口，等待 SimpleFOC 校准完成后，下发角度控制指令。
"""

import serial
import serial.tools.list_ports
import time
import sys

def find_stm32_port():
    """扫描并返回第一个找到的 ACM 或 USB 串口"""
    print("正在扫描可用的串口设备...")
    ports = [p.device for p in serial.tools.list_ports.comports() if 'ACM' in p.device or 'USB' in p.device]
    
    if not ports:
        print("未发现任何串口设备！请检查 USB 连线。")
        return None
    
    print(f"找到设备: {ports[0]}")
    return ports[0]

def main():
    target_port = find_stm32_port()
    if not target_port:
        return

    try:
        # 打开串口，波特率必须与 SimpleFOC 里的 Serial.begin(115200) 一致
        ser = serial.Serial(target_port, 115200, timeout=0.5)
        
        print("\n正在等待 SimpleFOC 芯片复位与电机校准...")
        print("提示: 此时电机应该会发出轻微的电流声并微微转动对齐。")
        
        # 给足时间让 STM32 跑完 motor.init() 和 motor.initFOC()
        # 通常需要 3~5 秒
        time.sleep(5) 
        
        # 读取并打印初始化期间 STM32 吐出的日志 (如 "Motor ready.")
        while ser.in_waiting:
            msg = ser.readline().decode('utf-8', errors='ignore').strip()
            if msg: print(f"[STM32 Init] {msg}")

        # 清空缓冲区，准备发送指令
        ser.reset_input_buffer()
        print("\n电机校准完成，控制链路已打通！")
        print("-" * 40)
        print("请选择测试模式：")
        print("1: 手动输入目标角度")
        print("2: 自动摇摆测试 (在 -3 到 3 弧度间自动转动)")
        print("q: 退出")
        print("-" * 40)

        mode = input("请输入选择 (1 / 2 / q): ").strip()

        if mode == '1':
            print("已进入手动模式。请输入目标角度(弧度)，例如 1.57, -3.14。输入 'q' 退出。")
            while True:
                val = input("请输入目标角度: ").strip()
                if val.lower() == 'q':
                    break
                try:
                    angle = float(val)
                    # 组装指令，例如 "T1.57\n"
                    # SimpleFOC 的 Commander 会解析 'T' 后面的数字
                    cmd = f"T{angle}\n"
                    ser.write(cmd.encode('utf-8'))
                    print(f"  [Jetson] 发送指令: {cmd.strip()}")
                    
                    # 稍微等一下，如果有回音则打印
                    time.sleep(0.05)
                    while ser.in_waiting:
                        resp = ser.readline().decode('utf-8', errors='ignore').strip()
                        if resp: print(f"  [STM32] {resp}")
                        
                except ValueError:
                    print("请输入有效的数字！")
                    
        elif mode == '2':
            print("已进入自动摇摆模式。按 Ctrl+C 停止。")
            target_angles = [0.0, 3.14, 0.0, -3.14]
            idx = 0
            while True:
                angle = target_angles[idx % len(target_angles)]
                cmd = f"T{angle}\n"
                ser.write(cmd.encode('utf-8'))
                print(f"  [Jetson] 发送摇摆指令: {cmd.strip()}")
                idx += 1
                time.sleep(2) # 每隔2秒变动一次目标角度

        elif mode.lower() == 'q':
            print("退出测试。")

    except KeyboardInterrupt:
        print("\n收到中断信号，程序停止。")
    except Exception as e:
        print(f"\n发生错误: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            # 退出前为了安全，可以让电机回到 0 度
            ser.write(b"T0\n")
            time.sleep(0.1)
            ser.close()
            print("串口已安全关闭。")

if __name__ == "__main__":
    main()