# -*- coding: utf-8 -*-
"""用 best.pt 預測 val（或任意資料夾），輸出 mask 疊圖與每張數量統計。
python predict_val.py --src ../dataset/images/val --out ../runs/predict_val"""
import argparse, os
from ultralytics import YOLO
ap = argparse.ArgumentParser()
ap.add_argument("--src", default="../dataset/images/val")
ap.add_argument("--out", default="../runs/predict_val")
ap.add_argument("--w", default="../runs/yolo11n_seg_v1/weights/best.pt")
a = ap.parse_args()
m = YOLO(a.w)
os.makedirs(a.out, exist_ok=True)
for r in m.predict(a.src, imgsz=640, conf=0.25, stream=True, verbose=False):
    cls = r.boxes.cls.int().tolist() if r.boxes is not None else []
    print(os.path.basename(r.path), "big", cls.count(0), "small", cls.count(1), flush=True)
    r.save(os.path.join(a.out, os.path.basename(r.path)), boxes=False, labels=False)
