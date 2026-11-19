# -*- coding: utf-8 -*-
"""找出驗證集中「預測了但沒對到標註」的 mask（誤報），並輸出疊圖與清單。
在 ~/TEL_YOLO/scripts 底下用 venv_yolo 執行：python find_fp.py"""
import json, os
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "dataset/images/val")
LAB = os.path.join(ROOT, "dataset/labels/val")
OUT = os.path.join(ROOT, "runs/fp_check")
os.makedirs(OUT, exist_ok=True)
NAMES = ["hole_big", "hole_small"]
IOU_T = 0.45
CONF = 0.25


def gt_masks(stem, W, H):
    res = []
    for line in open(os.path.join(LAB, stem + ".txt")):
        p = line.split()
        if len(p) < 7:
            continue
        c = int(p[0]); v = list(map(float, p[1:]))
        poly = [(v[i] * W, v[i + 1] * H) for i in range(0, len(v), 2)]
        m = Image.new("L", (W, H), 0)
        ImageDraw.Draw(m).polygon(poly, fill=1)
        res.append((c, np.array(m, bool)))
    return res


def iou(a, b):
    i = (a & b).sum()
    u = (a | b).sum()
    return float(i / u) if u else 0.0


model = YOLO(os.path.join(ROOT, "runs/yolo11n_seg_v1/weights/best.pt"))
rows, fps = [], []
for fn in sorted(os.listdir(IMG)):
    stem = os.path.splitext(fn)[0]
    im = Image.open(os.path.join(IMG, fn)).convert("RGB")
    W, H = im.size
    gts = gt_masks(stem, W, H)
    r = model.predict(os.path.join(IMG, fn), imgsz=640, conf=CONF, retina_masks=True, verbose=False)[0]
    preds = []
    if r.masks is not None:
        for k in range(len(r.masks.data)):
            m = r.masks.data[k].cpu().numpy() > 0.5
            if m.shape != (H, W):
                m = np.array(Image.fromarray(m.astype(np.uint8)).resize((W, H), Image.NEAREST), bool)
            preds.append((int(r.boxes.cls[k]), float(r.boxes.conf[k]), m))
    # 依信心由高到低，貪婪配對
    order = sorted(range(len(preds)), key=lambda k: -preds[k][1])
    used = set(); matched = {}
    for k in order:
        best, bi = 0.0, -1
        for j, (c, g) in enumerate(gts):
            if j in used:
                continue
            v = iou(preds[k][2], g)
            if v > best:
                best, bi = v, j
        if bi >= 0 and best >= IOU_T:
            used.add(bi); matched[k] = (bi, best)
    unmatched = [k for k in range(len(preds)) if k not in matched]
    missed = [j for j in range(len(gts)) if j not in used]
    rows.append((fn, len(gts), len(preds), len(unmatched), len(missed)))
    for k in unmatched:
        c, cf, m = preds[k]
        ys, xs = np.nonzero(m)
        best_any = max([iou(m, g) for _, g in gts] or [0.0])
        overlap_gt = max([(m & g).sum() / max(m.sum(), 1) for _, g in gts] or [0.0])
        info = dict(image=fn, cls=NAMES[c], conf=round(cf, 3), area=int(m.sum()),
                    bbox=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                    best_iou_with_any_gt=round(best_any, 3), frac_inside_gt=round(float(overlap_gt), 3))
        fps.append(info)
        # 疊圖：所有標註（白框輪廓）、所有預測（藍/綠）、這個誤報（紅）
        base = np.array(im).astype(np.float32)
        for c2, cf2, m2 in preds:
            col = np.array([80, 120, 255] if c2 == 0 else [60, 220, 120], np.float32)
            base[m2] = base[m2] * 0.55 + col * 0.45
        base[m] = base[m] * 0.3 + np.array([255, 40, 40], np.float32) * 0.7
        o = Image.fromarray(base.astype(np.uint8))
        d = ImageDraw.Draw(o)
        x0, y0, x1, y1 = info["bbox"]
        d.rectangle([x0 - 6, y0 - 6, x1 + 6, y1 + 6], outline=(255, 255, 0), width=5)
        o.thumbnail((700, 940)); o.save(os.path.join(OUT, f"{stem}_FP{len(fps)}_full.jpg"), quality=85)
        pad = 80
        crop = o0 = Image.fromarray(base.astype(np.uint8)).crop(
            (max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, W), min(y1 + pad, H)))
        crop.thumbnail((500, 500)); crop.save(os.path.join(OUT, f"{stem}_FP{len(fps)}_zoom.jpg"), quality=88)
        info["files"] = [f"{stem}_FP{len(fps)}_full.jpg", f"{stem}_FP{len(fps)}_zoom.jpg"]

json.dump(fps, open(os.path.join(OUT, "fp.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
tot = lambda i: sum(r[i] for r in rows)
print("images", len(rows), "GT", tot(1), "pred", tot(2), "FP(unmatched pred)", tot(3), "missed GT", tot(4))
for r in rows:
    if r[3] or r[4]:
        print(r)
for f in fps:
    print(f)
