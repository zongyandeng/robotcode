# -*- coding: utf-8 -*-
"""重新產生 tests/expected/robotcam_onnx.jsonl（實拍照片的「正確答案」）。
只在確定新結果比較好時才跑（例如改了排格子邏輯、換了模型），跑完用 git diff 看哪些照片的結果變了。

    .venv_vision\\Scripts\\python.exe tests\\make_expected.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "integration", "orin"))
import vision_node as vn  # noqa: E402

MODEL = os.path.join(ROOT, "runs", "yolo11n_seg_v1", "export", "best.onnx")
IMAGES = os.path.join(ROOT, "data_robotcam")
OUT = os.path.join(ROOT, "tests", "expected", "robotcam_onnx.jsonl")

a = vn.parse_args(["--model", MODEL, "--source", "images", "--no-nt", "--row-anchor", ""])
pipeline = vn.VisionPipeline(vn.load_model(MODEL), a, vn.RowAnchor(None, a.cam_y_ref))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    for name, img, depth, K in vn.image_frames(IMAGES, a.hfov_deg):
        d = json.loads(vn.json_result(name, pipeline.process(img, depth, K)[0]))
        d.pop("latency_ms", None)
        f.write(json.dumps(d, ensure_ascii=False) + "\n")
print("寫入", OUT)
