# -*- coding: utf-8 -*-
"""masks_r2.json -> YOLO seg 資料集（hole_big / hole_small）。
規則：只用 prompt 0（circular hole）；巢狀重複（小 mask 完全在大 mask 內）保留內層；
      最上排 3 個（中心 y 最小）= hole_small，其餘 = hole_big；另以面積檢查一致性。
用法：python build_dataset.py --masks masks_r2.json --splits ../splits --out ../dataset"""
import argparse, json, os, shutil
import numpy as np
from PIL import Image, ImageDraw


def rasterize(poly, W, H):
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).polygon([tuple(p) for p in poly], fill=1)
    return np.array(m, bool)


def dedup(ins, W, H):
    ms = [rasterize(i["poly"], W, H) for i in ins]
    drop = set()
    for a in range(len(ins)):
        for b in range(len(ins)):
            if a == b or a in drop or b in drop:
                continue
            if ms[a].sum() < ms[b].sum() and (ms[a] & ms[b]).sum() / max(ms[a].sum(), 1) >= 0.95:
                drop.add(b)  # a 在 b 內：丟掉外層 b
    return [i for k, i in enumerate(ins) if k not in drop], len(drop)


def drop_tiny(ins):
    """面積 < 中位數 10% 的碎片（例如邊緣誤判）直接丟掉"""
    med = float(np.median([i["area"] for i in ins]))
    keep = [i for i in ins if i["area"] >= 0.1 * med]
    return keep, len(ins) - len(keep)


def classify(ins):
    """面積為主、位置為輔，回傳 (類別list, 小孔是否都落在最上排)"""
    med = float(np.median([i["area"] for i in ins]))
    by_y = sorted(range(len(ins)), key=lambda k: ins[k]["centroid"][1])
    top3 = set(by_y[:3])
    # 明顯小的 = 小孔；最上排 3 個且面積 < 0.75 × 中位數的也算小孔（涵蓋「範圍含 LED 環」而偏大的小孔 mask）
    cls = [1 if (i["area"] < 0.5 * med or (k in top3 and i["area"] < 0.75 * med)) else 0 for k, i in enumerate(ins)]
    return cls, top3 >= {k for k, c in enumerate(cls) if c} and sum(cls) >= 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", default="masks_r2.json")
    ap.add_argument("--splits", default="../splits")
    ap.add_argument("--raw", default="../raw")
    ap.add_argument("--out", default="../dataset")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    d = json.load(open(a.masks, encoding="utf-8"))
    rep, labels = {"not12": [], "disagree": [], "dedup_total": 0, "n_small": {}}, {}
    for n, im in d["images"].items():
        W, H = im["w"], im["h"]
        ins = [i for i in im["instances"] if i["prompt"] == 0]
        ins, nd = dedup(ins, W, H)
        rep["dedup_total"] += nd
        ins, nt = drop_tiny(ins)
        rep["tiny_dropped"] = rep.get("tiny_dropped", 0) + nt
        if nt:
            rep.setdefault("tiny_images", []).append(n)
        cls, agree = classify(ins)
        if len(ins) != 12:
            rep["not12"].append((n, len(ins)))
        if not agree:
            rep["disagree"].append(n)
        rep["n_small"][n] = int(sum(cls))
        lines = []
        for c, i in zip(cls, ins):
            pts = " ".join(f"{x / W:.6f} {y / H:.6f}" for x, y in i["poly"])
            lines.append(f"{c} {pts}")
        labels[n] = lines
    print("dedup 丟掉的外層 mask 數:", rep["dedup_total"])
    print("碎片 mask 丟掉:", rep.get("tiny_dropped", 0), rep.get("tiny_images"))
    print("去重後不是 12 個的圖:", rep["not12"])
    import collections
    print("每張小孔數分布:", dict(collections.Counter(rep["n_small"].values())))
    print("位置/面積分類不一致的圖:", len(rep["disagree"]), rep["disagree"][:10])
    if a.dry:
        return
    for sp in ("train", "val"):
        names = [l.strip() for l in open(os.path.join(a.splits, f"{sp}.txt"), encoding="utf-8") if l.strip()]
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(a.out, sub, sp), exist_ok=True)
        for n in names:
            shutil.copy2(os.path.join(a.raw, n), os.path.join(a.out, "images", sp, n))
            with open(os.path.join(a.out, "labels", sp, os.path.splitext(n)[0] + ".txt"), "w") as f:
                f.write("\n".join(labels[n]) + "\n")
        print(sp, len(names))
    with open(os.path.join(a.out, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(a.out)}\ntrain: images/train\nval: images/val\nnames:\n  0: hole_big\n  1: hole_small\n")
    json.dump(rep, open(os.path.join(a.out, "build_report.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
