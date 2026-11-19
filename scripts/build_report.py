# -*- coding: utf-8 -*-
"""依 crack_v5_r3_aug_neg 範本版型，產生 yolo11n_seg_v1 的 HTML 報告（圖片以 data URI 內嵌）。"""
import base64, io, os
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED = os.path.join(ROOT, "runs", "predict_val")


def img(n):
    p = os.path.join(PRED, f"LINE_ALBUM_場地照片_261013_{n}.jpg")
    im = Image.open(p).convert("RGB")
    im.thumbnail((380, 520))
    b = io.BytesIO()
    im.save(b, "JPEG", quality=78)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def fig(n, cap):
    return f'<figure><img alt="驗證圖 _{n} 的預測疊圖" src="{img(n)}"><figcaption>{cap}</figcaption></figure>'


def data_uri(path, maxw=None, q=82):
    im = Image.open(path).convert("RGB")
    if maxw:
        im.thumbnail((maxw, maxw * 2))
    b = io.BytesIO()
    im.save(b, "JPEG" if path.lower().endswith(("jpg", "jpeg")) else "PNG", quality=q)
    mime = "jpeg" if path.lower().endswith(("jpg", "jpeg")) else "png"
    return f"data:image/{mime};base64," + base64.b64encode(b.getvalue()).decode()


FP = os.path.join(ROOT, "runs", "fp_check")
CM_URI = data_uri(os.path.join(ROOT, "runs", "confusion_matrix_iou045.png"))
fp_items = [(10, 1, "0.32", "畫面左下角，地上的塑膠袋與雜物"), (19, 2, "0.48", "畫面下緣，地上的塑膠袋"),
            (6, 3, "0.85", "畫面右下角，地上的袋子"), (94, 4, "0.26", "畫面右緣，爬梯子的人的腿")]
G_FP = ""
for n, k, cf, where in fp_items:
    f = f"LINE_ALBUM_場地照片_261013_{n}_FP{k}_full.jpg"
    z = f"LINE_ALBUM_場地照片_261013_{n}_FP{k}_zoom.jpg"
    G_FP += (f'<figure><img alt="驗證圖 _{n} 的誤報位置（黃框）" src="{data_uri(os.path.join(FP, f), 380)}"><figcaption>_{n}　信心 {cf}　{where}（黃框）</figcaption></figure>'
             f'<figure><img alt="驗證圖 _{n} 誤報放大" src="{data_uri(os.path.join(FP, z), 380)}"><figcaption>_{n}　誤報放大（紅色）</figcaption></figure>')

ok = [(13, "_13　大 9／小 3"), (14, "_14　大 9／小 3"), (29, "_29　大 9／小 3"),
      (32, "_32　大 9／小 3"), (35, "_35　大 9／小 3"), (42, "_42　大 9／小 3")]
extra = [(6, "_6　大 10／小 3"), (10, "_10　大 10／小 3"), (19, "_19　大 10／小 3"), (94, "_94　大 10／小 3")]
occl = [(70, "_70　大 9／小 2（第三個小孔被人擋住，標註也是 2 個）")]
G_OK = "".join(fig(n, c) for n, c in ok)
G_EX = "".join(fig(n, c) for n, c in occl)

CSS = """:root{--bg:#f1f3f2;--surface:#fafbfa;--ink:#161d1c;--mute:#5a6664;--rule:#d3dad8;--acc:#0d6e66;--tp:#1e6b45;--fp:#a3372c;--fn:#8a5a10;--hi:#fff1c9}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#0f1615;--surface:#152020;--ink:#e2ebe9;--mute:#94a6a3;--rule:#26373a;--acc:#4dc2b6;--tp:#6fcf9b;--fp:#e5776b;--fn:#e0a84a;--hi:#3a3216;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0f1615;--surface:#152020;--ink:#e2ebe9;--mute:#94a6a3;--rule:#26373a;--acc:#4dc2b6;--tp:#6fcf9b;--fp:#e5776b;--fn:#e0a84a;--hi:#3a3216;color-scheme:dark}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font:15px/1.7 "Noto Sans TC","IBM Plex Sans",system-ui,sans-serif;margin:0;padding-inline:16px;padding-block:32px 64px}
main{max-width:940px;margin:0 auto}
h1{font-size:clamp(28px,5vw,40px);line-height:1.2;margin:0 0 6px;text-wrap:balance;font-family:"IBM Plex Mono",ui-monospace,Consolas,monospace}
h2{font-size:20px;margin:44px 0 10px}
p{margin:0 0 12px;max-width:72ch}
.lead{color:var(--mute);margin-bottom:22px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.kpi{background:var(--surface);border:1px solid var(--rule);padding:14px 16px;min-width:0}
.kpi .v{font:600 32px/1.1 "IBM Plex Mono",monospace;color:var(--acc)}
.kpi .t{font-size:13px;color:var(--mute);margin-top:6px}
.tbl{overflow-x:auto;background:var(--surface);border:1px solid var(--rule)}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{padding:8px 12px;text-align:left;border-bottom:1px solid var(--rule);vertical-align:top}
thead th{font-size:12px;color:var(--mute);font-weight:500;white-space:nowrap}
tr:last-child td{border-bottom:0}
td.n{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums;font-family:"IBM Plex Mono",monospace}
td.best{background:var(--hi);font-weight:600}
.note{border-left:3px solid var(--fn);padding:2px 0 2px 14px;margin:18px 0 0}
.foot{font-size:12.5px;color:var(--mute);margin:10px 0 0}
.cmwrap{max-width:560px}
table.cm{table-layout:fixed}
table.cm th,table.cm td{border:1px solid var(--rule);text-align:center;padding:10px}
table.cm td{font:600 24px "IBM Plex Mono",monospace}
table.cm small{display:block;font:400 11.5px/1.4 "Noto Sans TC",sans-serif;color:var(--mute);margin-top:2px}
.tp{color:var(--tp)}.fp{color:var(--fp)}.fn{color:var(--fn)}.na{color:var(--mute)}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px}
figure{margin:0;background:var(--surface);border:1px solid var(--rule);min-width:0}
figure img{width:100%;height:auto;display:block}
figcaption{padding:6px 10px;font-size:12px;color:var(--mute)}
ul{padding-left:20px;margin:0 0 12px;max-width:72ch}
li{margin:4px 0}
footer{margin-top:44px;color:var(--mute);font-size:12.5px}
"""

BODY = """<h1>yolo11n_seg_v1</h1>
<p class="lead">九宮格孔洞偵測：YOLO11n-seg，兩個類別 hole_big（直徑 40 cm 大孔）、hole_small（直徑 20 cm 小孔）。資料集 104 張（SAM 3.1 標註，已清理巢狀重複 mask），train 80 張／valid 24 張（固定種子 42，valid 不擴增）；訓練 640、batch 12、100 epochs（早停 patience 10，第 55 個 epoch 停，最佳在第 51 個），起點為 COCO 預訓練權重。亮燈判斷不在這個模型裡。</p>

<div class="kpis">
<div class="kpi"><div class="v">0.765</div><div class="t">mask mAP50-95（valid 24 張，兩類平均）<br>最佳 epoch 51 的曲線值 0.768</div></div>
<div class="kpi"><div class="v">0.995</div><div class="t">mask mAP50（兩類平均）<br>精確率 0.997／召回率 1.000</div></div>
<div class="kpi"><div class="v">4／291</div><div class="t">預測的 291 個 mask 有 4 個誤報（不在孔上）；287 個標註孔漏報 0、類別判錯 0</div></div>
</div>
<div class="kpis" style="margin-top:12px">
<div class="kpi"><div class="v">0.864</div><div class="t">hole_big　mask mAP50-95<br>216 個實例，mAP50 0.995</div></div>
<div class="kpi"><div class="v">0.666</div><div class="t">hole_small　mask mAP50-95<br>71 個實例，mAP50 0.995</div></div>
</div>
<p class="note"><b>怎麼算：</b>valid 24 張共 287 個標註孔（大 216、小 71），標註來自 SAM，不是人工逐個畫的。mAP 為 mask 的 IoU 門檻 0.5（mAP50）與 0.5～0.95（mAP50-95）。混淆矩陣與預測數量統計都用信心門檻 0.25，預測與標註的 mask IoU 達 0.45 才算配對成功。<b>valid 與 train 拍自同一天、同一場地，相鄰照片很像，所有數字偏樂觀，還沒有機器人相機的實拍驗證。</b></p>

<h2>1　指標（valid 24 張，mask）</h2>
<div class="tbl"><table><thead><tr><th>類別</th><th>實例</th><th>精確率</th><th>召回率</th><th>mAP50</th><th>mAP50-95</th></tr></thead><tbody>
<tr><td>全部</td><td class="n">287</td><td class="n">0.997</td><td class="n">1.000</td><td class="n">0.995</td><td class="n">0.765</td></tr>
<tr><td>hole_big</td><td class="n">216</td><td class="n">0.994</td><td class="n">1.000</td><td class="n">0.995</td><td class="n best">0.864</td></tr>
<tr><td>hole_small</td><td class="n">71</td><td class="n">0.999</td><td class="n">1.000</td><td class="n">0.995</td><td class="n">0.666</td></tr></tbody></table></div>
<p class="foot">mAP50 已接近上限，差別在 mask 貼合程度。小孔像素少、邊緣誤差佔比高，且小孔的 SAM 標註範圍本身不一致，所以 mAP50-95 較低。推論 1.1 ms／張（實驗室 AMD 7900 XTX，不代表 Jetson）。</p>

<h2>2　混淆矩陣（實例層級，conf 0.25，valid 287 個孔）</h2>
<p>列是預測、欄是真實。「True＝background」那欄是誤報（預測了但不在任何標註孔上），「Predicted＝background」那列是漏報。</p>
<div class="cmwrap"><img alt="混淆矩陣：大孔 216 全中、小孔 71 全中，4 個誤報" src="@@CM@@" style="width:100%;height:auto;display:block;border:1px solid var(--rule);background:#fff"></div>
<p class="foot">大孔與小孔沒有互相混淆，287 個標註孔都被找到，另有 4 個預測不在任何孔上，全部被當成 hole_big。ultralytics 驗證流程內建的混淆矩陣只列出 1 個誤報，我用逐個 mask 的配對重算得到 4 個，並實際看過這 4 個的位置（第 5 節），採用後者。</p>

<h2>3　每張圖偵測到幾個孔（valid 24 張，門檻 0.25）</h2>
<p>理想是每張 9 個大孔、最多 3 個小孔。</p>
<div class="tbl"><table><thead><tr><th>偵測數量</th><th>張數</th><th>圖片</th></tr></thead><tbody>
<tr><td>大孔 9 個</td><td class="n">20</td><td>—</td></tr>
<tr><td>大孔 10 個（多 1 個）</td><td class="n fp">4</td><td>_6、_10、_19、_94</td></tr>
<tr><td>小孔 3 個</td><td class="n">23</td><td>—</td></tr>
<tr><td>小孔 2 個</td><td class="n">1</td><td>_70（第三個小孔被爬梯子的人擋住，標註也是 2 個）</td></tr></tbody></table></div>
<p class="foot">這 4 張多出來的大孔都是誤報，位置見第 5 節。</p>

<h2>4　預測的樣子（6 張，大孔藍紫、小孔綠）</h2>
<div class="gallery">@@OK@@</div>

<h2>5　4 個誤報在哪裡</h2>
<p>4 個誤報全部落在畫面邊緣的雜物上，不在九宮格板子裡：地上的塑膠袋與袋子、爬梯子的人的腿。信心 0.26、0.32、0.48、0.85，所以只靠提高信心門檻擋不掉最高的那個（0.85）。</p>
<div class="gallery">@@FP@@</div>
<h3 style="font-size:16px;margin:24px 0 8px">另一張：小孔只有 2 個</h3>
<div class="gallery">@@EX@@</div>
<p class="foot">_70 的第三個小孔被爬梯子的人擋住，標註本身也只有 2 個，不是漏報。</p>
<p>可行的處理方式（尚未實作）：只保留位置落在 9＋3 孔格局內的偵測；或把板子外框當作區域，丟掉區域外的 mask。</p>

<h2>6　限制</h2>
<ul>
<li>valid 與 train 同一天、同一場地拍攝，結果主要證明流程可行，不代表機器人上的表現。</li>
<li>現有照片是人站著用手機拍的，與機器人相機（低位、距離約 2.5 m）視角不同，還沒有實拍驗證集。</li>
<li>小孔 mask mAP50-95 只有 0.666，標註範圍不一致會影響中心點的精度。</li>
<li>板子外的雜物（袋子、人的腿）會被誤判成大孔，valid 有 4 個，需要用位置過濾或補負樣本。</li>
<li>valid 只有 24 張，指標容易持平，patience 10 在第 55 個 epoch 就停了。</li>
<li>模型只回答「哪 3 個位置是小孔」，哪一個亮紅燈還沒做（計畫用 OpenCV 紅色比例判斷）。</li>
<li>資料量小：train 80 張，光線、遮擋、角度變化有限。</li>
</ul>

<footer>遠端 AMD 7900XTX（ROCm）；訓練腳本 train.py、資料集 build_dataset.py，權重 runs/yolo11n_seg_v1/weights/best.pt（6 MB）。擴增為線上即時進行：旋轉 ±5°、hsv_v 0.5、hsv_s 0.5、hsv_h 0.005、透視 0.0005、不翻轉、mosaic 0.5（最後 5 個 epoch 關閉）。</footer>
"""

html = (
    '<title>yolo11n_seg_v1</title>\n'
    '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Noto+Sans+TC:wght@400;500;700&display=swap">\n'
    "<style>" + CSS + "</style>\n<main>\n"
    + BODY.replace("@@OK@@", G_OK).replace("@@EX@@", G_EX).replace("@@FP@@", G_FP).replace("@@CM@@", CM_URI)
    + "</main>\n"
)
out = os.path.join(ROOT, "report", "yolo11n_seg_v1_report.html")
open(out, "w", encoding="utf-8").write(html)
print(out, len(html) // 1024, "KB")
