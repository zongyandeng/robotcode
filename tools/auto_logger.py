# -*- coding: utf-8 -*-
"""自主流程除錯記錄器：從 roboRIO 的 NetworkTables 以約 50 Hz 記錄位置與 swerve 模組狀態，存成 CSV。

用法（筆電接機器人網路）：
    ..\\.venv_vision\\Scripts\\python.exe auto_logger.py            # 開始記錄，Ctrl+C 結束
記錄：時間、機器人是否啟用、模式、/Pose/robotPose (x, y, deg)、Auto/Phase、
      四個模組的「目標」與「實際」角度 (度) 與速度 (m/s)（DriveState/ModuleTargets、ModuleStates）。
"""
import csv
import math
import os
import struct
import sys
import time

import ntcore

SERVER = sys.argv[1] if len(sys.argv) > 1 else "10.94.27.2"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), time.strftime("auto_log_%Y%m%d_%H%M%S.csv"))

inst = ntcore.NetworkTableInstance.create()
inst.startClient4("auto-logger")
inst.setServer(SERVER)
opts = ntcore.PubSubOptions(sendAll=True, periodic=0.01)
pose = inst.getDoubleArrayTopic("/Pose/robotPose").subscribe([], opts)
states = inst.getRawTopic("/DriveState/ModuleStates").subscribe("struct:SwerveModuleState[]", b"", opts)
targets = inst.getRawTopic("/DriveState/ModuleTargets").subscribe("struct:SwerveModuleState[]", b"", opts)
phase = inst.getStringTopic("/SmartDashboard/Auto/Phase").subscribe("")
mode = inst.getStringTopic("/SmartDashboard/Robot/Mode").subscribe("")
enabled = inst.getBooleanTopic("/SmartDashboard/Robot/Enabled").subscribe(False)


def modules(raw):
    """SwerveModuleState struct = speed (double, m/s) + angle (Rotation2d = double rad)，4 個模組 = 64 bytes。"""
    if not raw or len(raw) < 64:
        return [math.nan] * 8
    out = []
    for i in range(4):
        sp, ang = struct.unpack_from("<dd", raw, 16 * i)
        out += [round(sp, 3), round(math.degrees(ang), 1)]
    return out


names = ["FL", "FR", "BL", "BR"]
hdr = ["t", "enabled", "mode", "phase", "x", "y", "deg"]
hdr += [f"{n}_tgt_{k}" for n in names for k in ("speed", "deg")]
hdr += [f"{n}_act_{k}" for n in names for k in ("speed", "deg")]
print("記錄到", OUT, "（Ctrl+C 結束）")
t0 = time.time()
n = 0
with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(hdr)
    try:
        while True:
            p = pose.get()
            p = list(p) + [math.nan] * (3 - len(p)) if len(p) < 3 else list(p)[:3]
            w.writerow([round(time.time() - t0, 3), int(enabled.get()), mode.get(), phase.get(),
                        *[round(v, 4) for v in p], *modules(targets.get()), *modules(states.get())])
            n += 1
            if n % 50 == 0:
                f.flush()
                print(f"\r{n} 筆  啟用={enabled.get()}  位置=({p[0]:.2f}, {p[1]:.2f}, {p[2]:.1f}°)", end="", flush=True)
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
print(f"\n完成，共 {n} 筆：{OUT}")
