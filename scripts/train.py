# -*- coding: utf-8 -*-
"""YOLO11n-seg 訓練（hole_big / hole_small）。驗證集不擴增（ultralytics 本來就只擴增 train）。
用法（在 ~/TEL_YOLO/ 底下，venv_yolo）：
  YOLO_CONFIG_DIR=~/TEL_YOLO/.yolo_config python scripts/train.py"""
import os
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = YOLO(os.path.join(ROOT, "weights", "yolo11n-seg.pt"))
model.train(
    data=os.path.join(ROOT, "dataset", "data.yaml"),
    epochs=100, patience=10, imgsz=640, batch=12, workers=4, seed=42, deterministic=True,
    project=os.path.join(ROOT, "runs"), name="yolo11n_seg_v1", exist_ok=True,
    # 擴增：旋轉 ±5°、亮度/飽和度大、色相極小、輕微透視、不翻轉
    degrees=5.0, hsv_h=0.005, hsv_s=0.5, hsv_v=0.5, perspective=0.0005,
    translate=0.1, scale=0.4, fliplr=0.0, flipud=0.0, mosaic=0.5, close_mosaic=5, mixup=0.0,
)
