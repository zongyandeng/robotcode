"""Orbbec Gemini 2 VL 測試：同時顯示彩色 + 深度畫面，可即時調整相機參數。

執行:  .venv\\Scripts\\python.exe depth_test.py
按鍵:  q / ESC 離開
       s     存目前畫面 (color / depth png)
       l     雷射開關
       a     彩色自動曝光開關
       滑鼠移到深度畫面上會顯示該點距離 (公尺)；視窗上方滑桿可調參數
"""
import time
import cv2
import numpy as np
from pyorbbecsdk import (Config, Context, OBFormat, OBPropertyID, OBSensorType,
                         Pipeline)

P = OBPropertyID
MIN_MM, MAX_MM = 200, 4000      # 深度顯示範圍 (mm)，Gemini 2 VL 約 0.15~5 m

mouse_xy = [None]


def on_mouse(event, x, y, *_):
    mouse_xy[0] = (x, y)


def frame_to_bgr(f):
    w, h, fmt = f.get_width(), f.get_height(), f.get_format()
    d = np.asanyarray(f.get_data())
    if fmt == OBFormat.MJPG:
        return cv2.imdecode(d, cv2.IMREAD_COLOR)
    if fmt == OBFormat.RGB:
        return cv2.cvtColor(d.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
    if fmt == OBFormat.BGR:
        return d.reshape(h, w, 3)
    if fmt == OBFormat.YUYV:
        return cv2.cvtColor(d.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUY2)
    if fmt == OBFormat.UYVY:
        return cv2.cvtColor(d.reshape(h, w, 2), cv2.COLOR_YUV2BGR_UYVY)
    return None


def add_prop_slider(dev, win, name, pid):
    """若相機支援該屬性，就在視窗加一條滑桿。"""
    try:
        if not dev.is_property_supported(pid, 3):   # 3 = READ_WRITE
            return
        r = dev.get_int_property_range(pid)
        lo, hi, cur, step = r.min, r.max, r.cur, max(r.step, 1)
        cv2.createTrackbar(name, win, cur - lo, (hi - lo) // step or 1,
                           lambda v, lo=lo, step=step, pid=pid:
                           dev.set_int_property(pid, lo + v * step))
        print(f"  滑桿 {name}: {lo}~{hi} (目前 {cur})")
    except Exception as e:
        print(f"  (略過 {name}: {e})")


def toggle(dev, pid, label):
    try:
        v = not dev.get_bool_property(pid)
        dev.set_bool_property(pid, v)
        print(f"{label}: {'ON' if v else 'OFF'}")
    except Exception as e:
        print(f"{label} 不支援: {e}")


def main():
    ctx = Context()
    if ctx.query_devices().get_count() == 0:
        print("找不到 Orbbec 相機，請確認 USB 3 連接。")
        return
    pipe = Pipeline()
    dev = pipe.get_device()
    info = dev.get_device_info()
    print(f"裝置: {info.get_name()}  FW: {info.get_firmware_version()}  "
          f"SN: {info.get_serial_number()}")

    cfg = Config()
    for sensor in (OBSensorType.COLOR_SENSOR, OBSensorType.DEPTH_SENSOR):
        pl = pipe.get_stream_profile_list(sensor)
        prof = pl.get_default_video_stream_profile()
        print(f"{sensor.name}: {prof.get_width()}x{prof.get_height()} "
              f"@{prof.get_fps()}fps {prof.get_format().name}")
        cfg.enable_stream(prof)
    pipe.start(cfg)

    win = "Gemini2 (color | depth)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    print("可調參數滑桿:")
    add_prop_slider(dev, win, "ColorExposure", P.OB_PROP_COLOR_EXPOSURE_INT)
    add_prop_slider(dev, win, "ColorGain", P.OB_PROP_COLOR_GAIN_INT)
    add_prop_slider(dev, win, "DepthExposure", P.OB_PROP_DEPTH_EXPOSURE_INT)
    add_prop_slider(dev, win, "DepthGain", P.OB_PROP_DEPTH_GAIN_INT)
    add_prop_slider(dev, win, "LaserPower", P.OB_PROP_LASER_POWER_LEVEL_CONTROL_INT)

    n, t0, fps = 0, time.time(), 0.0
    try:
        while True:
            fs = pipe.wait_for_frames(200)
            if fs is None:
                continue
            cf, df = fs.get_color_frame(), fs.get_depth_frame()
            if cf is None or df is None:
                continue
            color = frame_to_bgr(cf)
            if color is None:
                continue
            dh, dw = df.get_height(), df.get_width()
            depth = np.frombuffer(df.get_data(), dtype=np.uint16).reshape(dh, dw)
            depth_mm = depth.astype(np.float32) * df.get_depth_scale()

            vis = np.clip((depth_mm - MIN_MM) / (MAX_MM - MIN_MM), 0, 1)
            vis = cv2.applyColorMap((vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
            vis[depth == 0] = 0                       # 無效點塗黑

            # 深度圖縮放成與彩色同高再併排
            vis = cv2.resize(vis, (int(dw * color.shape[0] / dh), color.shape[0]))
            sx, sy = dw / vis.shape[1], dh / vis.shape[0]

            # 中心點 / 滑鼠點距離
            pts = [(dw // 2, dh // 2)]
            if mouse_xy[0] and mouse_xy[0][0] >= color.shape[1]:
                mx = int((mouse_xy[0][0] - color.shape[1]) * sx)
                my = int(mouse_xy[0][1] * sy)
                if 0 <= mx < dw and 0 <= my < dh:
                    pts.append((mx, my))
            for (px, py) in pts:
                patch = depth_mm[max(py - 2, 0):py + 3, max(px - 2, 0):px + 3]
                patch = patch[patch > 0]
                dist = float(np.median(patch)) / 1000 if patch.size else 0
                cx, cy = int(px / sx), int(py / sy)
                cv2.drawMarker(vis, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 16, 2)
                cv2.putText(vis, f"{dist:.2f} m" if dist else "no data",
                            (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)

            n += 1
            if time.time() - t0 >= 1:
                fps, n, t0 = n, 0, time.time()
            out = np.hstack([color, vis])
            cv2.putText(out, f"{fps:.0f} fps", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow(win, out)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            elif k == ord('s'):
                ts = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"color_{ts}.png", color)
                cv2.imwrite(f"depth_{ts}.png", depth)   # 16-bit 原始深度
                print("已存檔", ts)
            elif k == ord('l'):
                toggle(dev, P.OB_PROP_LASER_BOOL, "雷射")
            elif k == ord('a'):
                toggle(dev, P.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, "彩色自動曝光")
    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
