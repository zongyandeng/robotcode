# -*- coding: utf-8 -*-
"""自主射擊除錯記錄器：從 roboRIO 的 NetworkTables 以約 20 Hz 記錄「發射條件」相關的值，存成 CSV。

用法（筆電接機器人網路）：
    ..\\.venv_vision\\Scripts\\python.exe shoot_logger.py            # 開始記錄，Ctrl+C 結束
記錄：仰角（目標／實際／到位）、發射輪（上下輪目標／實際／到位）、車頭誤差、即時瞄準狀態、
      目前目標孔、已發射球數、Feeder／Orbit 狀態、電池電壓。
"""
import csv
import os
import sys
import time

import ntcore

SERVER = sys.argv[1] if len(sys.argv) > 1 else "10.94.27.2"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), time.strftime("shoot_log_%Y%m%d_%H%M%S.csv"))

KEYS = [
    "Robot/Enabled", "Auto/Phase",
    "AutoShoot/Target Index", "AutoShoot/Balls Fired",
    "Hood/Target (Rotations)", "Hood/Position (Degrees)", "Hood/At Target", "Hood/Motor Voltage (V)",
    "Shooter/TopTargetRPS", "Shooter/Top Actual RPS", "Shooter/BottomTargetRPS", "Shooter/Bottom Actual RPS",
    "Shooter/At Target Velocity",
    "AutoShoot/Heading Error deg", "AutoShoot/Live Aim", "AutoShoot/Live Bearing deg",
    "AutoShoot/Live Aim Corrections", "AutoShoot/Live Elev deg",
    "AutoShoot/Hood Target deg", "AutoShoot/Hood Trim deg",
    "Feeder/State", "Feeder/Position (Rotations)", "Orbit/Velocity (RPS)",
    "Robot/Battery V",
    # 2026-11-19：送球偵測（Orbit ID56 電流）、卡球、底盤守點、影像定位
    "Orbit/Current (A)", "Orbit/Ball Detect", "Orbit/Feed Floor A", "Orbit/Feed Peak A", "Orbit/Feed Rise A",
    "Tune/Orbit Rise Amps", "Tune/Orbit Detect Enabled", "AutoShoot/Jam Not Fired", "Feeder/Jam Count",
    "AutoShoot/Chassis", "AutoShoot/Hold Error cm", "AutoShoot/Relocalize", "AutoShoot/Replan",
    "Align/Status", "AutoShoot/Strategy",
    "AutoShoot/Hole Source", "AutoShoot/Aim Source", "Vision/Locked", "Tune/Hole Table Enabled", "Tune/Live Elev Enabled",
    "AutoShoot/Shooter RPS", "Orbit/Feed Rise A",
]

inst = ntcore.NetworkTableInstance.create()
inst.startClient4("shoot-logger")
inst.setServer(SERVER)
entries = [inst.getEntry("/SmartDashboard/" + k) for k in KEYS]

t0 = time.time()
while not inst.isConnected() and time.time() - t0 < 10:
    time.sleep(0.1)
if not inst.isConnected():
    print("連不上 roboRIO", SERVER)
    sys.exit(1)
print("已連線，記錄到", OUT, "（Ctrl+C 結束）")

with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["t"] + KEYS)
    start = time.time()
    try:
        while True:
            row = [f"{time.time() - start:.3f}"]
            for e in entries:
                v = e.getValue()
                row.append(v.value() if v.isValid() else "")
            w.writerow(row)
            f.flush()
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("結束")
