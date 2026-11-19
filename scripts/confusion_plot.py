# -*- coding: utf-8 -*-
"""畫混淆矩陣（Predicted 為列、True 為欄，Blues 色階＋色條），數字取自 find_fp.py 的逐孔配對結果。
python confusion_plot.py"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
labels = ["hole_big", "hole_small", "background"]
# 列 = 預測，欄 = 真實。大孔 216 全中、小孔 71 全中；4 個預測沒對到任何標註（真實＝背景）。
M = np.array([[216, 0, 4],
              [0, 71, 0],
              [0, 0, 0]])

fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=150)
im = ax.imshow(M, cmap="Blues", vmin=0, vmax=max(M.max(), 1) * 1.15)
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(labels, rotation=90); ax.set_yticklabels(labels)
ax.set_xlabel("True"); ax.set_ylabel("Predicted")
ax.set_title("Confusion Matrix: mask IoU>=0.45 (conf 0.25)")
for i in range(3):
    for j in range(3):
        v = M[i, j]
        txt = "-" if (i == 2 and j == 2) else str(v)
        ax.text(j, i, txt, ha="center", va="center", fontsize=15,
                color="white" if v > M.max() * 0.55 else "black")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
out = os.path.join(ROOT, "runs", "confusion_matrix_iou045.png")
fig.savefig(out)
print(out)
