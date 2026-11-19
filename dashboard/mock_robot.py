# -*- coding: utf-8 -*-
"""假的機器人：在這台電腦開 NetworkTables 伺服器（port 5810）與 MJPEG 串流（port 1181），
發布和真機器人同名的數值，讓儀表板在沒有機器人時也能測試畫面。

用法（需要 pyntcore、pillow）：
    python mock_robot.py [--images ../data]
然後把 dashboard.html 右上角的 roboRIO 欄位改成 localhost，相機網址填 http://localhost:1181/stream.mjpg
"""
import argparse
import glob
import io
import math
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import ntcore
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--images", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
ap.add_argument("--no-vision", action="store_true",
                help="不發布假的 Vision 表與相機串流（改由真的 vision_node.py 連到這台發布）")
a = ap.parse_args()

# ---- MJPEG：輪流播放資料夾裡的照片 ----
files = sorted(glob.glob(os.path.join(a.images, "*.jpg")))[:30]
frames = []
for f in files:
    im = Image.open(f).convert("RGB")
    im.thumbnail((640, 480))
    b = io.BytesIO()
    im.save(b, "JPEG", quality=70)
    frames.append(b.getvalue())


class H(BaseHTTPRequestHandler):
    def log_message(self, *x):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        i = 0
        try:
            while frames:
                jpg = frames[(i // 8) % len(frames)]
                i += 1
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg))
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
                time.sleep(0.066)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


if not a.no_vision:   # --no-vision 時 1181 留給 vision_node.py 的串流
    httpd = ThreadingHTTPServer(("0.0.0.0", 1181), H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

# ---- NetworkTables ----
inst = ntcore.NetworkTableInstance.getDefault()
inst.startServer()
sd = inst.getTable("SmartDashboard")
vt = inst.getTable("Vision")
pose_t = inst.getTable("Pose")
# 模擬 SendableChooser「Auto/Start Side」：儀表板寫 selected，機器人回報 active
chooser = inst.getTable("SmartDashboard/Auto/Start Side")
chooser.getStringTopic(".type").publish().set("String Chooser")
chooser.getStringArrayTopic("options").publish().set(["左起始框", "右起始框"])
chooser.getStringTopic("default").publish().set("左起始框")
side_sel = chooser.getStringTopic("selected").subscribe("")
side_active = chooser.getStringTopic("active").publish()
# 模擬 SendableChooser「Auto/Strategy」（射擊策略）
strat_opts = ["上半場 27 顆（放棄紅燈格）", "下半場 9 顆 每孔 1 顆（剩下手動補）", "下半場 18 顆 紅燈格 + 連線"]
strat_chooser = inst.getTable("SmartDashboard/Auto/Strategy")
strat_chooser.getStringTopic(".type").publish().set("String Chooser")
strat_chooser.getStringArrayTopic("options").publish().set(strat_opts)
strat_chooser.getStringTopic("default").publish().set(strat_opts[0])
strat_sel = strat_chooser.getStringTopic("selected").subscribe("")
strat_active = strat_chooser.getStringTopic("active").publish()
# 模擬填表工具：Tune/Shooter RPS 開機設一次預設值（之後由儀表板改），Tune/Hood Target Deg 讓仰角走過去
_tune_rps_pub = sd_tbl = inst.getTable("SmartDashboard")
_rps_pub = sd_tbl.getDoubleTopic("Tune/Shooter RPS").publish(); _rps_pub.set(75.0)
_hood_pub = sd_tbl.getDoubleTopic("Tune/Hood Target Deg").publish(); _hood_pub.set(-1.0)
tune_rps = sd_tbl.getDoubleTopic("Tune/Shooter RPS").subscribe(75.0)
tune_hood = sd_tbl.getDoubleTopic("Tune/Hood Target Deg").subscribe(-1.0)
hood_deg_state = [20.0]
pubs = {}


def put(table, key, kind, val):
    k = (id(table), key)
    if k not in pubs:
        topic = {"d": table.getDoubleTopic, "b": table.getBooleanTopic, "s": table.getStringTopic,
                 "i": table.getIntegerTopic, "da": table.getDoubleArrayTopic}[kind](key)
        pubs[k] = topic.publish()
    pubs[k].set(val)


nan = float("nan")
fid = 0
t0 = time.time()
print("mock robot 已啟動：NT 5810、MJPEG http://localhost:1181/stream.mjpg（Ctrl+C 結束）")
while True:
    t = time.time() - t0
    cycle = t % 30.0
    shooting = cycle > 6
    top_t, bot_t = (60 * 0.85, 60 * 1.15) if shooting else (0.0, 0.0)
    ramp = min(1.0, max(0.0, (cycle - 6) / 2.0)) if shooting else 0.0
    top_a, bot_a = top_t * ramp, bot_t * ramp
    ho = 0.1 * min(1.0, ramp)
    balls = int(max(0, min(27, (cycle - 9) / 0.9))) if shooting else 0
    ready_sh = shooting and ramp >= 1.0

    side_active.set(side_sel.get() or "左起始框")
    strat_active.set(strat_sel.get() or strat_opts[0])
    sgn = -1.0 if side_active and side_sel.get() == "右起始框" else 1.0
    # 場地圖：0~6 秒從起始框靠中線的前角 (0.93, ±1.57) 斜向開到射擊點 (3.5, 0)，之後原地左右轉著瞄準
    if cycle < 6:
        k = cycle / 6.0
        px, py, pdeg = 0.93 + (3.5 - 0.93) * k, sgn * 1.57 * (1 - k), 0.0
    else:
        px, py, pdeg = 3.5, 0.0, 8.0 * math.sin((cycle - 6) / 2.0)
    put(pose_t, "robotPose", "da", [px, py, pdeg - 90.0])   # 發布的是底盤程式車頭 = 相機朝向 - 90
    locked = cycle > 7
    holes = []
    for i in range(12):
        holes += [7.5 + 0.04, (1 - i % 3) * 0.90 - 0.03] if locked else [nan, nan]   # 假裝視覺量到板子偏了幾公分
    put(sd, "Vision/Locked", "b", locked); put(sd, "Vision/Board Holes", "da", holes)

    put(sd, "Robot/Enabled", "b", True)
    put(sd, "Robot/Mode", "s", "自主" if cycle < 28 else "遙控")
    put(sd, "Robot/Battery V", "d", 12.3 - 0.3 * math.sin(t / 3) - (0.8 if shooting else 0))
    put(sd, "Robot/Match Time", "d", max(0.0, 180 - t % 180))

    put(sd, "Shooter/TopTargetRPS", "d", top_t); put(sd, "Shooter/Top Actual RPS", "d", top_a)
    put(sd, "Shooter/BottomTargetRPS", "d", bot_t); put(sd, "Shooter/Bottom Actual RPS", "d", bot_a)
    put(sd, "Shooter/At Target Velocity", "b", ready_sh)
    tgt = tune_hood.get()
    if tgt >= 0:
        hood_deg_state[0] += max(-2.0, min(2.0, tgt - hood_deg_state[0]))   # 每 50 ms 最多走 2 度
        put(sd, "Hood/Target (Rotations)", "d", tgt / 360.0)
    hd = hood_deg_state[0] if tgt >= 0 else ho * 360
    put(sd, "Hood/Position (Degrees)", "d", hd); put(sd, "Hood/Position (Rotations)", "d", hd / 360.0)
    put(sd, "Shooter/Top Actual RPS", "d", tune_rps.get() * 0.85); put(sd, "Shooter/Bottom Actual RPS", "d", tune_rps.get() * 1.15)
    put(sd, "Hood/At Target", "b", ramp >= 1.0)
    put(sd, "Feeder/Position (Rotations)", "d", balls / 6.0)
    put(sd, "Orbit/Velocity (RPS)", "d", 18.0 if (shooting and int(t * 2) % 2 == 0) else 0.0)
    put(sd, "Orbit/Current (A)", "d", 12.0 + 4 * math.sin(t))
    put(sd, "AutoShoot/Balls Fired", "d", balls)
    put(sd, "AutoShoot/Target Index", "d", 1 if balls < 3 else 3 + (balls - 3) // 3)
    put(sd, "AutoShoot/Hood Rot", "d", 0.1)
    put(sd, "AutoShoot/Shooter RPS", "d", 60.0); put(sd, "AutoShoot/Distance m", "d", 2.6)
    if a.no_vision:
        time.sleep(0.05)
        continue
    put(sd, "Vision/Connected", "b", True); put(sd, "Vision/Lit Small Col", "d", 1)

    bearing = [nan] * 12; dist = [nan] * 12
    for i in range(12):
        col, row = i % 3, i // 3
        bearing[i] = (col - 1) * 12.0 + 1.5 * math.sin(t + i)
        dist[i] = 2.6 + 0.02 * row
    bearing[11] = nan; dist[11] = nan   # 假裝有一個孔沒看到
    put(vt, "bearing_deg", "da", bearing); put(vt, "dist_m", "da", dist)
    put(vt, "grid_ok", "b", True); put(vt, "lit_small", "i", 1); put(vt, "n_big", "i", 8); put(vt, "n_small", "i", 3)
    put(vt, "fps", "d", 29.7); put(vt, "red_score", "da", [0.0, 0.11, 0.01])
    fid += 1
    put(vt, "frame_id", "i", fid)
    time.sleep(0.05)
