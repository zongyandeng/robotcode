# -*- coding: utf-8 -*-
"""用 best.pt 在 val 上算混淆矩陣（IoU 0.45、conf 0.25 ultralytics 預設的混淆矩陣設定）。"""
import os, json
from ultralytics import YOLO
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
m = YOLO(os.path.join(ROOT, "runs/yolo11n_seg_v1/weights/best.pt"))
r = m.val(data=os.path.join(ROOT, "dataset/data.yaml"), imgsz=640, plots=True, project=os.path.join(ROOT, "runs"), name="val_confusion", exist_ok=True, verbose=False)
cm = r.confusion_matrix.matrix
print("NAMES", list(r.names.values()) + ["background"])
print("MATRIX_JSON", json.dumps(cm.astype(int).tolist()))
