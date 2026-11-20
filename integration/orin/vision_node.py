# -*- coding: utf-8 -*-
"""Orin Nano 視覺節點：USB 相機／RealSense + YOLO11n-seg -> 12 個孔的 3D 位置 + 亮燈小孔 -> NetworkTables。

孔的編號（Java 端用同一套）：index = row * 3 + col
    row 0 = 最上排 3 個小孔（col 0..2 由左到右）
    row 1..3 = 大孔三排（由上到下），col 0..2 由左到右
座標約定（相機在底盤上，面向前方）：
    bearing_deg  水平偏角，正 = 目標在相機右側
    pitch_deg    垂直仰角，正 = 目標在相機上方
    dist_m       相機到目標的水平距離（公尺）
    height_m     目標高於相機光心的高度（公尺）
缺資料的孔以 NaN 表示。

用法：
  # 在 Orin 上，接 RealSense，發布到 roboRIO（IP 與隊號之後再設定）
  python vision_node.py --model best.engine --source realsense --team 1234
  python vision_node.py --model best.engine --source realsense --server 10.12.34.2
  # 離線測試（任何電腦，不需 roboRIO／RealSense）：對照片跑一遍，印 JSON，並輸出疊圖
  python vision_node.py --model best.pt --source images --path ./imgs --no-nt --debug-dir ./dbg

相機校正值（焦距、距離修正）放在同資料夾的 calibration.json，命令列參數可以覆蓋。
"""
import argparse
import collections
import glob
import json
import math
import os
import threading
import time

import cv2
import numpy as np

NAN = float("nan")
BIG, SMALL = "hole_big", "hole_small"
HERE = os.path.dirname(os.path.abspath(__file__))

# calibration.json 不存在、或少了某個欄位時用的預設值
CALIB_DEFAULTS = dict(fx=1057.0, dist_scale=1.1905, dist_offset=-0.6872, hfov_deg=69.0, camera_pitch_deg=0.0)


# ----------------------------------------------------------------------------
# 偵測 -> 孔清單
# ----------------------------------------------------------------------------
class Hole:
    """一個偵測到的孔。遮罩只存孔周圍的小區塊（不是整張畫面），後面的環狀區域運算也只在這塊裡做，
    Orin 的 CPU 才跑得動（整張畫面做大核心膨脹會讓幀率掉到個位數）。
    mask 是區塊內的遮罩，區塊左上角在原圖的 (x0, y0)；區塊比孔的外框往外多留 pad 像素。"""

    def __init__(self, cls, poly, shape):
        self.cls = cls
        self.poly = poly.astype(np.float32)
        h, w = shape[:2]
        bx, by, bw, bh = cv2.boundingRect(self.poly.astype(np.int32))
        pad = int(max(bw, bh) * 0.9) + 4            # 夠放下 red_fraction / hole_depth 的外環（0.7 倍孔徑）
        self.x0, self.y0 = max(bx - pad, 0), max(by - pad, 0)
        x1, y1 = min(bx + bw + pad, w), min(by + bh + pad, h)
        m = np.zeros((max(y1 - self.y0, 1), max(x1 - self.x0, 1)), np.uint8)
        cv2.fillPoly(m, [(self.poly - [self.x0, self.y0]).astype(np.int32)], 1)
        self.mask = m
        mo = cv2.moments(m, binaryImage=True)        # m00 = 像素數，m10/m00、m01/m00 = 像素中心
        self.area = float(mo["m00"])
        self.cx = mo["m10"] / mo["m00"] + self.x0 if mo["m00"] else NAN
        self.cy = mo["m01"] / mo["m00"] + self.y0 if mo["m00"] else NAN
        self.diam = math.sqrt(4.0 * self.area / math.pi)  # 等效直徑（像素）
        self.row = -1
        self.col = -1

    def crop(self, img):
        """原圖中對應 mask 的區塊。"""
        return img[self.y0:self.y0 + self.mask.shape[0], self.x0:self.x0 + self.mask.shape[1]]


def detect_holes(model, frame, imgsz, conf):
    if hasattr(model, "predict_polys"):   # trt_seg.SegModel（TensorRT / ONNX Runtime，不需要 PyTorch）
        raw = model.predict_polys(frame, conf)
    else:
        r = model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
        if r.masks is None:
            return []
        raw = [(r.names[int(r.boxes.cls[k])], np.asarray(poly))
               for k, poly in enumerate(r.masks.xy) if len(poly) >= 3]
    holes = [Hole(cls, poly, frame.shape) for cls, poly in raw]
    return [h for h in holes if h.area > 20]


# ----------------------------------------------------------------------------
# 把孔排成 3x4 格（容忍板子有點歪）
# ----------------------------------------------------------------------------
def _gap_cluster(vals, thr):
    """1D 依間隔分群：排序後相鄰差距 > thr 就切開。回傳每個值的群編號（由小到大 0,1,2...）與群數。"""
    order = np.argsort(vals)
    lab = np.zeros(len(vals), int)
    g = 0
    for k in range(1, len(order)):
        if vals[order[k]] - vals[order[k - 1]] > thr:
            g += 1
        lab[order[k]] = g
    return lab, g + 1


# 板子實際尺寸（2026 規則書：大孔直徑 40 cm、小孔 20 cm；欄中心距 90 cm；
# 小孔中心到第一排大孔中心 50 cm；大孔排與排中心距 60 cm）
BOARD_COL_M = 0.90
BOARD_SMALL_TO_ROW1_M = 0.50
BOARD_ROW_M = 0.60
BIG_DIAM_M = 0.40


def assign_grid(holes):
    """依位置為孔填 row/col（row 0 = 小孔排，1~3 = 大孔由上到下；col 0~2 由左到右）。
    不假設 12 個孔都在畫面裡（相機仰角大時最下面一排常常看不到），也容許透視造成的欄線傾斜、偶爾多一個誤判：
      1. 排：所有孔依垂直間隔分排（搜尋板子在畫面中的旋轉角）；小孔為主的那排是第 0 排，
         大孔排用實際間距（小孔→第 1 排 50 cm、排與排 60 cm）換算是第幾排；沒看到小孔就假設最上面是第 1 排
      2. 欄：剛好 3 個孔的排，由左到右編 0、1、2，並用這些排擬合三條欄線（x 隨 y 線性變化，容許透視）；
         其他排的孔對到最近的欄線
    回傳 grid_ok：至少 4 個孔有唯一的格子（夠用來 solvePnP）。對不上的孔維持 -1、不採用。"""
    for h in holes:
        h.row = h.col = -1
    good = [h for h in holes if np.isfinite(h.cx)]
    bigs = [h for h in good if h.cls == BIG]
    smalls = [h for h in good if h.cls == SMALL]
    if len(good) < 4:
        return False
    dbig = float(np.median([h.diam for h in bigs])) if bigs else 2.0 * float(np.median([h.diam for h in smalls]))
    pts = np.array([[h.cx, h.cy] for h in good])
    # 1) 找讓「排」最整齊的旋轉角
    #    範圍 ±15 度（車頭轉向斜看板子時，透視會讓排看起來傾斜 6 度以上）；每多一群加罰分。
    #    不加罰分時，孔在畫面上很大（仰角低、只剩兩排）、角度轉到 ±15 度附近，同一排相鄰的孔旋轉後高度差會超過門檻，
    #    6 個孔被拆成 6 個「單孔排」，每群誤差都是 0 反而得分最好 -> 找不到 3 孔排、grid_ok 失敗
    #    （瞄左下、右下孔時沒有影像資料）。
    best = None
    group_penalty = (0.25 * dbig) ** 2
    for deg in np.arange(-15, 15.1, 1.0):
        t = math.radians(deg)
        c, s = math.cos(t), math.sin(t)
        yr = -pts[:, 0] * s + pts[:, 1] * c
        lab, k = _gap_cluster(yr, dbig * 0.6)
        sse = sum(float(((yr[lab == g] - yr[lab == g].mean()) ** 2).sum()) for g in range(k))
        score = sse + k * group_penalty
        if best is None or score < best[0]:
            best = (score, deg, lab, k)
    _, deg, rlab, nrow = best
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    xr = pts[:, 0] * c + pts[:, 1] * s
    yr = -pts[:, 0] * s + pts[:, 1] * c
    rows = []                                   # (平均 y, [孔在 good 中的索引], 是否小孔排)
    for g in range(nrow):
        idx = [i for i in range(len(good)) if rlab[i] == g]
        n_small = sum(good[i].cls == SMALL for i in idx)
        rows.append((float(np.mean(yr[idx])), idx, n_small * 2 > len(idx)))
    rows.sort(key=lambda r: r[0])
    small_rows = [r for r in rows if r[2]]
    big_rows = [r for r in rows if not r[2]]
    # 2) 排號
    row_no = {}
    if len(small_rows) > 1:                     # 小孔不會有兩排：分群失敗
        return False
    big_y = [r[0] for r in big_rows]
    pitch = float(np.median(np.diff(big_y))) if len(big_y) >= 2 else dbig * BOARD_ROW_M / BIG_DIAM_M
    if small_rows:
        row_no[id(small_rows[0])] = 0
        first = small_rows[0][0] + pitch * BOARD_SMALL_TO_ROW1_M / BOARD_ROW_M
        for r in big_rows:
            row_no[id(r)] = int(round((r[0] - first) / pitch)) + 1
    else:
        for k_, r in enumerate(big_rows):
            row_no[id(r)] = k_ + 1
    # 3) 欄線：用剛好 3 個孔的排擬合 x = a + b*y（每欄一條）
    full = [r for r in rows if len(r[1]) == 3]
    if not full:
        return False
    col_pts = [[], [], []]
    for r in full:
        for col, i in enumerate(sorted(r[1], key=lambda i: xr[i])):
            col_pts[col].append((yr[i], xr[i]))
    lines = []
    for cp in col_pts:
        cp = np.array(cp)
        if len(cp) >= 2 and np.ptp(cp[:, 0]) > dbig:
            b_, a_ = np.polyfit(cp[:, 0], cp[:, 1], 1)
        else:
            b_, a_ = 0.0, float(cp[:, 1].mean())
        lines.append((a_, b_))
    # 4) 指定 row/col
    for r in rows:
        rn = row_no[id(r)]
        if not (0 <= rn <= 3):
            continue
        used = {}
        for i in r[1]:
            pred = [a_ + b_ * yr[i] for a_, b_ in lines]
            col = int(np.argmin([abs(xr[i] - p) for p in pred]))
            if abs(xr[i] - pred[col]) > dbig * 0.9:     # 離最近的欄線太遠：誤判，不採用
                continue
            if col in used:                              # 同一格兩個孔：留比較接近欄線的
                j = used[col]
                if abs(xr[i] - pred[col]) >= abs(xr[j] - pred[col]):
                    continue
                good[j].row = good[j].col = -1
            used[col] = i
            good[i].row, good[i].col = rn, col
    cells = [(h.row, h.col) for h in good if h.row >= 0]
    return len(cells) >= 4 and len(set(cells)) == len(cells)


# ----------------------------------------------------------------------------
# 孔外圍的環狀區域（亮燈判斷與深度都用）
# ----------------------------------------------------------------------------
def ring_mask(hole, inner, outer):
    """孔區塊內的環狀遮罩（bool）：孔邊緣往外 inner*d 到 outer*d（d = 孔的等效直徑）。"""
    d = max(hole.diam, 6.0)
    k_in = max(int(inner * d), 1)
    k_out = max(int(outer * d), k_in + 2)
    ker = lambda k: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    return (cv2.dilate(hole.mask, ker(k_out)) - cv2.dilate(hole.mask, ker(k_in))).astype(bool)


def red_fraction(frame_bgr, hole, inner=0.10, outer=0.70):
    """環狀區域內紅色像素的比例。"""
    ring = ring_mask(hole, inner, outer)
    if ring.sum() < 20:
        return 0.0
    hsv = cv2.cvtColor(hole.crop(frame_bgr), cv2.COLOR_BGR2HSV)   # 只轉孔周圍的區塊
    hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = ((hh <= 10) | (hh >= 168)) & (ss >= 90) & (vv >= 90)
    return float(red[ring].mean())


def pick_lit(frame_bgr, holes, min_score=0.04, max_second_ratio=0.5):
    """回傳 (lit_col 或 -1, [col0,col1,col2 的分數])。
    最高分要達 min_score，且第二高不能超過最高的 max_second_ratio 倍，否則視為不確定（-1）。
    比賽實際只會亮一個；練習場常同時亮兩個，會得到 -1。"""
    scores = [0.0, 0.0, 0.0]
    for h in holes:
        if h.cls == SMALL and 0 <= h.col <= 2:
            scores[h.col] = max(scores[h.col], red_fraction(frame_bgr, h))
    order = sorted(range(3), key=lambda i: -scores[i])
    top, second = scores[order[0]], scores[order[1]]
    if top >= min_score and second <= max_second_ratio * top:
        return order[0], scores
    return -1, scores


# ----------------------------------------------------------------------------
# 3D：孔中心的像素 + 孔周圍「板子上」的深度（不能取孔內，孔後面是遠處的牆）
# ----------------------------------------------------------------------------
def hole_depth(depth_m, hole, inner=0.15, outer=0.55):
    v = hole.crop(depth_m)[ring_mask(hole, inner, outer)]
    v = v[(v > 0.2) & (v < 8.0)]
    if len(v) < 15:
        return NAN
    return float(np.median(v))


def board_point(index):
    """孔 index 在板子座標的位置（公尺）：X 往右、Y 往上、Z 從板面指向機器人；原點 = 中間那欄的小孔中心。"""
    row, col = divmod(index, 3)
    y = 0.0 if row == 0 else -(BOARD_SMALL_TO_ROW1_M + (row - 1) * BOARD_ROW_M)
    return np.array([(col - 1) * BOARD_COL_M, y, 0.0])


def board_pnp(holes, K, max_err_ratio=0.15):
    """用看到的孔中心 + 已知的板子尺寸解相機位姿（solvePnP），再算出「全部 12 個孔」相對相機的
    水平偏角、仰角、水平距離、高度（世界座標，不受相機仰角或隨仰角機構轉動影響）。
    沒看到的孔（例如仰角大時最下面一排）也算得出來。
    需要至少 4 個已編號的孔、重投影誤差夠小；失敗回傳 None。"""
    seen = [h for h in holes if h.row >= 0 and h.col >= 0]
    if len(seen) < 4:
        return None
    obj = np.array([board_point(h.row * 3 + h.col) for h in seen], np.float64)
    img = np.array([[h.cx, h.cy] for h in seen], np.float64)
    fx, fy, ppx, ppy = K
    cam = np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]], np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, cam, None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    proj, _ = cv2.projectPoints(obj, rvec, tvec, cam, None)
    err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)))
    if err > max_err_ratio * float(np.median([h.diam for h in seen])):
        return None
    R, _ = cv2.Rodrigues(rvec)
    C = (-R.T @ tvec).reshape(3)            # 相機在板子座標的位置
    if C[2] <= 0:                           # 相機跑到板子後面 = 解錯
        return None
    a = R.T @ np.array([0.0, 0.0, 1.0])     # 相機光軸在板子座標的方向
    out = []
    elev = []                               # 孔相對「相機光軸（＝畫面中央準心）」的上下角度，往上為正
    for i in range(12):
        vc = R @ board_point(i) + tvec.reshape(3)   # 孔在相機座標（x 右、y 下、z 前）
        elev.append(math.degrees(math.atan2(-vc[1], vc[2])))
        v = board_point(i) - C
        horiz = math.hypot(v[0], v[2])
        bearing = math.degrees(math.atan2(a[0] * v[2] - a[2] * v[0], a[0] * v[0] + a[2] * v[2]))   # 往右為正
        pitch = math.degrees(math.atan2(v[1], horiz))
        out.append((bearing, pitch, horiz, float(v[1])))
    return out, err, float(C[2]), elev, float(C[1])   # C[1] = 鏡頭在板子座標的高度（小孔排中心為 0，往上為正）


def to_target(hole, depth_m, K, cam_pitch_deg):
    """回傳 (bearing_deg, pitch_deg, dist_m, height_m)。K = (fx, fy, ppx, ppy)。"""
    fx, fy, ppx, ppy = K
    if depth_m is None:
        z = 1.0  # 沒有深度：只算角度
        dist_known = False
    else:
        z = hole_depth(depth_m, hole)
        dist_known = not math.isnan(z)
        if not dist_known:
            z = 1.0
    x = (hole.cx - ppx) / fx * z   # 右為正
    y = (hole.cy - ppy) / fy * z   # 下為正
    # 相機若向上仰 cam_pitch_deg，把點轉回水平座標
    p = math.radians(cam_pitch_deg)
    y2 = y * math.cos(p) + z * math.sin(p) * -1.0
    z2 = z * math.cos(p) + y * math.sin(p)
    bearing = math.degrees(math.atan2(x, z2))
    horiz = math.hypot(x, z2)
    pitch = math.degrees(math.atan2(-y2, horiz))
    if not dist_known:
        return bearing, pitch, NAN, NAN
    return bearing, pitch, horiz, -y2


def pixel_elev(h, K):
    """孔中心相對準心（光軸）的上下角度，往上為正（度），只用像素。"""
    return math.degrees(math.atan2(K[3] - h.cy, K[1]))


# ----------------------------------------------------------------------------
# 看不到小孔時決定排號
# ----------------------------------------------------------------------------
class RowAnchor:
    """沒看到小孔時決定排號。

    assign_grid 沒看到小孔時只能假設「最上面那排大孔是第 1 排」；仰角放低瞄最下排時，小孔和第 1 排都出了畫面，
    第 2、3 排會被當成第 1、2 排，第 3 排的孔就被算到板子下方不存在的位置，瞄準跟著往下掉。
    鏡頭固定在機器人上，它在板子座標的高度（solvePnP 的 C[1]）幾乎不變（仰角機構轉動只差幾公分），
    認錯一排就差 60 cm。所以：看得到小孔時記下鏡頭高度當基準（指數平均，存檔，重開機還在）；
    看不到小孔時分別假設最上面是第 1、2、3 排各解一次 PnP，選鏡頭高度最接近基準的。"""

    def __init__(self, path, default=None):
        self.path = path
        self.ref = default
        self._saved = 0.0
        # 每排的「畫面上下角度（相對準心，往上為正）+ 仰角機構角度」。鏡頭裝在仰角機構上，仰角轉 1 度畫面就上下移 1 度，
        # 機器人沒移動時這個值固定。看得到小孔（排號確定）時記下；看不到小孔時用「記下的值 - 目前仰角」預測每排在畫面的位置。
        # 只用像素與仰角讀數，不經過 PnP（只看到兩排大孔、擠在畫面邊緣時，PnP 的鏡頭高度會錯一公尺以上）。
        self.row_ang = {}
        if path is None:
            return
        try:
            with open(path) as f:
                self.ref = float(json.load(f)["cam_y"])
        except (OSError, ValueError, KeyError, TypeError):
            pass
        if self.ref is not None and not math.isfinite(self.ref):
            self.ref = None     # 存檔裡是 NaN（舊版會存進去）：當作沒有基準，重新學

    def learn_rows(self, holes, K, hood_deg):
        rows = {}
        for h in holes:
            if h.row >= 0 and h.col >= 0:
                rows.setdefault(h.row, []).append(pixel_elev(h, K))
        for r, v in rows.items():
            self.row_ang[r] = float(np.mean(v)) + hood_deg

    def learn(self, cam_y):
        if not math.isfinite(cam_y):
            return              # 一個 NaN 進到指數平均就永遠是 NaN，基準會失效
        self.ref = cam_y if self.ref is None else 0.9 * self.ref + 0.1 * cam_y
        now = time.time()
        if self.path is not None and now - self._saved > 5.0:
            self._saved = now
            try:
                with open(self.path, "w") as f:
                    json.dump({"cam_y": self.ref}, f)
            except OSError:
                pass


def solve_board(holes, K, anchor, hood_deg=None, hood_still=True):
    """solvePnP，沒看到小孔時用 RowAnchor 決定排號（會改 holes 的 row）。
    回傳 (pnp 結果或 None, 排號平移量, 判斷方式)。判斷方式：small = 看得到小孔、hood = 用仰角預測、camY = 用鏡頭高度。"""
    if any(h.row == 0 for h in holes):                  # 有小孔：排號可靠，順便更新基準
        pnp = board_pnp(holes, K)
        if pnp is not None and anchor is not None:
            anchor.learn(pnp[4])
            if hood_deg is not None and hood_still:      # 仰角在動時不學（畫面與讀數不同步）
                anchor.learn_rows(holes, K, hood_deg)
        return pnp, 0, "small"
    base = [(h, h.row) for h in holes if h.row >= 1]
    if not base:
        return None, 0, "none"
    top = max(r for _, r in base)
    # 優先：用仰角預測每排在畫面的位置（相鄰兩排差約 7 度，仰角讀數誤差 1 度多，很好分）
    if anchor is not None and hood_deg is not None and anchor.row_ang:
        best = None
        for shift in range(0, 4 - top):
            errs = [abs(pixel_elev(h, K) - (anchor.row_ang[r + shift] - hood_deg))
                    for h, r in base if (r + shift) in anchor.row_ang]
            if len(errs) == len(base):                  # 每個孔的排都有記錄才比
                e = float(np.mean(errs))
                if best is None or e < best[0]:
                    best = (e, shift)
        if best is not None:
            for h, r in base:
                h.row = r + best[1]
            pnp = board_pnp(holes, K)
            if pnp is not None:
                return pnp, best[1], "hood"
            for h, r in base:
                h.row = r
    # 退而求其次：PnP 的鏡頭高度（沒有仰角讀數、或還沒看過小孔時）
    cands = []
    for shift in range(0, 4 - top):                     # 平移後最下面那排不能超過第 3 排
        for h, r in base:
            h.row = r + shift
        pnp = board_pnp(holes, K)
        if pnp is not None:
            cands.append((shift, pnp))
    if not cands:
        for h, r in base:
            h.row = r
        return None, 0, "none"
    if anchor is not None and anchor.ref is not None:
        shift, pnp = min(cands, key=lambda c: abs(c[1][4] - anchor.ref))
    else:
        shift, pnp = cands[0]                           # 還沒有基準：沿用舊規則（最上面是第 1 排）
    for h, r in base:
        h.row = r + shift
    return pnp, shift, "camY"


# ----------------------------------------------------------------------------
# 一幀 -> 結果 dict
# ----------------------------------------------------------------------------
class VisionPipeline:
    """偵測 -> 排格子 -> 亮燈 -> 位姿 -> 12 個孔的目標值。跨幀的狀態（仰角歷史、排號基準）都放在這裡。"""

    def __init__(self, model, args, anchor=None):
        self.model = model
        self.imgsz, self.conf = args.imgsz, args.conf
        self.red_min, self.red_second_ratio = args.red_min, args.red_second_ratio
        self.camera_pitch_deg = args.camera_pitch_deg
        # 經驗修正（廣角鏡頭變形讓近距離被高估）：實際 = dist_scale * 讀數 + dist_offset，高度依同比例縮放
        self.dist_scale, self.dist_offset = args.dist_scale, args.dist_offset
        self.anchor = anchor
        self._hood_hist = collections.deque()          # (時間, 仰角)

    def _hood_still(self, t, hood_deg):
        """仰角是否靜止（最近 0.3 秒內變化 < 0.3 度）：只有靜止時才學每排的位置。
        仰角快速移動時（約 100 度/秒），畫面和仰角讀數差 0.1 秒就差 10 度，學到的位置會錯一排（2026-11-17 孔 10 瞄到第二排）"""
        hist = self._hood_hist
        hist.append((t, hood_deg))
        while hist and t - hist[0][0] > 0.3:
            hist.popleft()
        return len(hist) >= 3 and t - hist[0][0] > 0.2 and max(h for _, h in hist) - min(h for _, h in hist) < 0.3

    def process(self, frame_bgr, depth_m, K, hood_deg=None, t_cap=None):
        """hood_deg：拍這一幀當下的仰角讀數（度，沒有就 None）。t_cap：拍照時間（time.time()）。"""
        holes = detect_holes(self.model, frame_bgr, self.imgsz, self.conf)
        grid_ok = assign_grid(holes)
        lit, scores = pick_lit(frame_bgr, holes, self.red_min, self.red_second_ratio)
        n = 12
        out = dict(bearing=[NAN] * n, pitch=[NAN] * n, dist=[NAN] * n, height=[NAN] * n, elev=[NAN] * n)
        hood_still = self._hood_still(t_cap if t_cap is not None else time.time(), hood_deg) if hood_deg is not None else False
        pnp, row_shift, row_method = (solve_board(holes, K, self.anchor, hood_deg, hood_still)
                                      if (depth_m is None and grid_ok) else (None, 0, "none"))
        out["row_shift"] = row_shift
        out["row_method"] = row_method
        out["hood_deg"] = hood_deg
        out["cam_y"] = pnp[4] if pnp is not None else NAN
        out["anchor_ref"] = self.anchor.ref if self.anchor is not None else None
        if pnp is not None:
            # 沒有深度相機：用板子尺寸解位姿，12 個孔（含畫面外的）都有偏角、仰角、距離、高度
            targets, err, board_dist, elev, _ = pnp
            out["elev"] = [float(e) for e in elev]
            corr = lambda d: self.dist_scale * d + self.dist_offset
            for i, (b, p, d, ht) in enumerate(targets):
                dc = corr(d)
                out["bearing"][i], out["pitch"][i], out["dist"][i], out["height"][i] = b, p, dc, ht * dc / d
            board_dist = corr(board_dist)
        elif depth_m is None:
            # 沒有深度相機又解不出位姿：編號很可能錯了（例如分排失敗），寧可這一幀不給目標，也不要讓機器人瞄錯孔
            grid_ok = False
            err, board_dist = NAN, NAN
        else:
            for h in holes:
                if h.row < 0 or h.col < 0:
                    continue
                i = h.row * 3 + h.col
                b, p, d, ht = to_target(h, depth_m, K, self.camera_pitch_deg)
                out["bearing"][i], out["pitch"][i], out["dist"][i], out["height"][i] = b, p, d, ht
                out["elev"][i] = pixel_elev(h, K)   # 相對準心，往上為正
            err, board_dist = NAN, NAN
        out["pnp_err_px"], out["board_dist_m"] = err, board_dist
        out.update(grid_ok=bool(grid_ok), lit_small=int(lit), red_score=[float(s) for s in scores],
                   n_big=sum(h.cls == BIG for h in holes), n_small=sum(h.cls == SMALL for h in holes))
        # 拍照到算完的時間（毫秒）：roboRIO 可用「收到時間 - latency_ms」對齊里程計／仰角
        out["latency_ms"] = (time.time() - t_cap) * 1000.0 if t_cap is not None else NAN
        return out, holes


# ----------------------------------------------------------------------------
# 疊圖
# ----------------------------------------------------------------------------
def draw_aim(img, holes, res):
    """串流畫面正中央的準心，以及離準心最近那個孔的距離／偏角／仰角（實測比對、校正用）。"""
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.line(img, (cx - 25, cy), (cx + 25, cy), (0, 255, 0), 2)
    cv2.line(img, (cx, cy - 25), (cx, cy + 25), (0, 255, 0), 2)
    cv2.circle(img, (cx, cy), 6, (0, 255, 0), 1)
    lines = []
    if res.get("board_dist_m") == res.get("board_dist_m"):      # 不是 NaN
        lines.append(f"board {res['board_dist_m']:.2f} m  pnp err {res['pnp_err_px']:.1f}px")
    seen = [x for x in holes if x.row >= 0 and np.isfinite(x.cx)]
    if seen:
        near = min(seen, key=lambda x: (x.cx - cx) ** 2 + (x.cy - cy) ** 2)
        i = near.row * 3 + near.col
        b, p, d = res["bearing"][i], res["pitch"][i], res["dist"][i]
        e = res.get("elev", [NAN] * 12)[i]
        cv2.line(img, (cx, cy), (int(near.cx), int(near.cy)), (0, 255, 0), 1)
        lines.append(f"#{i}  dist {d:.2f} m  bearing {b:+.1f}  pitch {p:+.1f}")
        lines.append(f"elev {e:+.1f} (+ = hole above cross)")
    if res.get("cam_y") == res.get("cam_y"):                  # 不是 NaN
        hd = res.get("hood_deg")
        lines.append(f"rows by {res.get('row_method', '?')} shift {res.get('row_shift', 0)}  hood "
                     + (f"{hd:.1f}" if hd is not None else "?") + f"  camY {res['cam_y']:+.2f}")
    for k, t in enumerate(lines):
        cv2.putText(img, t, (cx + 32, cy + 22 + 24 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(img, t, (cx + 32, cy + 22 + 24 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def draw_debug(frame, holes, result):
    img = frame.copy()
    for h in holes:
        col = (255, 120, 60) if h.cls == BIG else (80, 220, 120)
        cv2.polylines(img, [h.poly.astype(np.int32)], True, col, 2)
        if h.row >= 0:
            cv2.putText(img, str(h.row * 3 + h.col), (int(h.cx) - 8, int(h.cy) + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    lit = result["lit_small"]
    for h in holes:
        if h.cls == SMALL and h.col == lit:
            cv2.circle(img, (int(h.cx), int(h.cy)), int(h.diam * 0.9), (0, 0, 255), 3)
    txt = f"grid_ok={result['grid_ok']} big={result['n_big']} small={result['n_small']} lit={lit} red={['%.2f' % s for s in result['red_score']]}"
    cv2.putText(img, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return img


# ----------------------------------------------------------------------------
# NetworkTables 發布
# ----------------------------------------------------------------------------
# /Vision/ 底下發布的 topic：(名稱, 型別, 結果 dict 的欄位)。兩種實作（pyntcore / nt4_lite）共用這張表，
# 依這個順序寫入，frame_id 永遠最後（Java 端以 frame_id 變動當作「新的一幀」）。
VISION_TOPICS = [
    ("grid_ok", "boolean", "grid_ok"),
    ("lit_small", "int", "lit_small"),
    ("n_big", "int", "n_big"),
    ("n_small", "int", "n_small"),
    ("fps", "double", "fps"),
    ("red_score", "double[]", "red_score"),
    ("bearing_deg", "double[]", "bearing"),
    ("pitch_deg", "double[]", "pitch"),
    ("dist_m", "double[]", "dist"),
    ("height_m", "double[]", "height"),
    ("elev_deg", "double[]", "elev"),            # 孔相對準心的上下角度（往上為正）
    ("latency_ms", "double", "latency_ms"),      # 拍照到發布的延遲（毫秒）
]
HOOD_TOPIC = "/SmartDashboard/Hood/Position (Degrees)"   # roboRIO 發布的仰角機構角度（度），用來判斷看不到小孔時的排號


def _server_of(team, server):
    if server:
        return server
    if not team:
        raise SystemExit("需要 --team 或 --server")
    t = int(team)
    return f"10.{t // 100}.{t % 100}.2"


class _SinkBase:
    """publish()：先寫資料，最後才更新 frame_id。子類別提供 _pub(名稱, 型別) 與 _flush()。"""

    def _setup(self):
        self._pubs = [(self._pub(name, typ), key) for name, typ, key in VISION_TOPICS]
        self.frame_id = self._pub("frame_id", "int")
        self._n = 0

    def publish(self, r, fps):
        r = dict(r, fps=fps)
        for pub, key in self._pubs:
            pub.set(r[key])
        self._n += 1
        self.frame_id.set(self._n)
        self._flush()


class NTSinkLite(_SinkBase):
    """沒有 pyntcore 時用的版本（Orin 的 Python 3.10 + aarch64 沒有 pyntcore 套件），走 nt4_lite.py。
    發布的 topic 名稱與型別和 NTSink 完全相同，roboRIO 端看不出差別。"""

    def __init__(self, team, server, name="orin-vision"):
        from nt4_lite import NT4Lite
        self.nt = NT4Lite(_server_of(team, server), name)
        self._setup()
        self._enable = self.nt.subscribe("/Vision/enable", default=True)
        self._hood = self.nt.subscribe(HOOD_TOPIC, default=None)

    def _pub(self, name, typ):
        return self.nt.publish("/Vision/" + name, typ)

    def _flush(self):
        self.nt.flush()

    def hood_deg(self):
        v = self._hood.get()
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None

    def enabled(self):
        return bool(self._enable.get())


class NTSink(_SinkBase):
    def __init__(self, team, server, name="orin-vision"):
        import ntcore
        self.inst = ntcore.NetworkTableInstance.getDefault()
        self.inst.startClient4(name)
        if server:
            self.inst.setServer(server)
        elif team:
            self.inst.setServerTeam(int(team))
        else:
            raise SystemExit("需要 --team 或 --server")
        self._table = self.inst.getTable("Vision")
        self._setup()
        # roboRIO 發布的開關：false = 不拍照、不跑辨識（自主開場移動期間關閉，到定點才開）
        # 沒有人發布時預設 True，所以沒接 roboRIO 也會照常運作
        self._enable = self._table.getBooleanTopic("enable").subscribe(True)
        self._hood = self.inst.getDoubleTopic(HOOD_TOPIC).subscribe(NAN)

    def _pub(self, name, typ):
        t = self._table
        getter = {"boolean": t.getBooleanTopic, "int": t.getIntegerTopic, "double": t.getDoubleTopic,
                  "double[]": t.getDoubleArrayTopic}[typ]
        return getter(name).publish()

    def _flush(self):
        self.inst.flush()

    def hood_deg(self):
        v = self._hood.get()
        return float(v) if math.isfinite(v) else None

    def enabled(self):
        return bool(self._enable.get())


def make_nt_sink(team, server):
    """有 pyntcore 就用官方的 NTSink，沒有（或設 VISION_NT_LITE=1）就用純 Python 的 NTSinkLite。"""
    if os.environ.get("VISION_NT_LITE") != "1":
        try:
            import ntcore  # noqa: F401
            return NTSink(team, server)
        except ImportError:
            pass
    print("pyntcore 不可用，改用 nt4_lite（純 Python NT4）")
    return NTSinkLite(team, server)


# ----------------------------------------------------------------------------
# MJPEG 串流（給儀表板的相機畫面；瀏覽器直接用 <img src="http://orin:1181/stream.mjpg"> 就能看）
# ----------------------------------------------------------------------------
class MJPEGServer:
    """width > 0 時先縮到這個寬度再編碼（儀表板顯示的尺寸遠小於 1280，縮小後編碼與傳輸都省很多）。"""

    def __init__(self, port, quality=70, width=640):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        self.latest = None
        self.quality, self.width = quality, width
        self.cond = threading.Condition()
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if not self.path.startswith("/stream.mjpg"):
                    self.send_response(404); self.end_headers(); return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                last = None
                try:
                    while True:
                        with outer.cond:     # 等下一張（最多 1 秒，順便偵測連線是否還在）
                            outer.cond.wait_for(lambda: outer.latest is not last, timeout=1.0)
                            jpg = outer.latest
                        if jpg is None or jpg is last:
                            continue
                        head = b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg)
                        self.wfile.write(head)
                        self.wfile.write(jpg)
                        self.wfile.write(b"\r\n")
                        last = jpg
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

        self.httpd = ThreadingHTTPServer(("0.0.0.0", port), H)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def update(self, bgr):
        h, w = bgr.shape[:2]
        if self.width and w > self.width:
            bgr = cv2.resize(bgr, (self.width, int(round(h * self.width / w))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if ok:
            with self.cond:
                self.latest = buf.tobytes()
                self.cond.notify_all()


# ----------------------------------------------------------------------------
# 影像來源：每一幀 yield (BGR, 深度或 None, K, 拍照時間, 拍照當下的仰角或 None)
# ----------------------------------------------------------------------------
def realsense_frames(w=640, h=480, fps=30, hood_fn=None):
    import pyrealsense2 as rs
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
    cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
    prof = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    scale = prof.get_device().first_depth_sensor().get_depth_scale()
    intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = (intr.fx, intr.fy, intr.ppx, intr.ppy)
    # 固定曝光很重要：LED 環亮度會讓自動曝光飽和成白色，紅色資訊就消失了
    try:
        cs = prof.get_device().first_color_sensor()
        cs.set_option(rs.option.enable_auto_exposure, 0)
        cs.set_option(rs.option.exposure, float(os.environ.get("RS_EXPOSURE", 80)))
        cs.set_option(rs.option.gain, float(os.environ.get("RS_GAIN", 32)))
    except Exception as e:  # 型號不支援時不要中斷
        print("曝光設定失敗：", e)
    try:
        while True:
            fs = align.process(pipe.wait_for_frames())
            t_cap = time.time()
            hood = hood_fn() if hood_fn else None
            c, d = fs.get_color_frame(), fs.get_depth_frame()
            if not c or not d:
                continue
            yield np.asanyarray(c.get_data()), np.asanyarray(d.get_data()).astype(np.float32) * scale, K, t_cap, hood
    finally:
        pipe.stop()


class ExposureControl:
    """USB 相機曝光控制（--exposure）：
      soft  ：程式自己調（預設）。有偵測到孔時只看「板子區域」的亮度，沒偵測到就看畫面中央；
              目標亮度 --ae-target（0~255），曝光上限 --ae-max-exp（避免拖慢幀率），
              區域內過曝的像素太多也會往下調，避免 LED 環變成一片白、紅色資訊消失。
      auto  ：交給相機內建的自動曝光。
      數字  ：固定曝光（Linux 單位是 exposure_time_absolute，多數相機 1 = 0.1 ms）。比賽要鎖死時用。
    Linux（Orin）用 v4l2-ctl 直接設定；Windows 只支援 auto 與固定值（soft 會改成 auto）。
    UVC 相機通電期間會記住上一個程式設的曝光，所以啟動時一定要重新設定，不然可能沿用到很暗的值。"""

    def __init__(self, mode="soft", target=110.0, max_exp=330, min_exp=1, every=4):
        self.mode, self.target, self.max_exp, self.min_exp, self.every = mode, target, max_exp, min_exp, every
        self.exp = None
        self.dev = None
        self.cap = None
        self._n = 0

    def _v4l2(self, **ctrls):
        import subprocess
        args = ",".join(f"{k}={v}" for k, v in ctrls.items())
        r = subprocess.run(["v4l2-ctl", "-d", self.dev, "-c", args], capture_output=True, text=True)
        if r.returncode != 0:
            print("v4l2-ctl 設定失敗：", args, r.stderr.strip())

    def attach(self, cap, index):
        self.cap = cap
        linux = os.name != "nt"
        self.dev = os.path.realpath(index) if isinstance(index, str) else f"/dev/video{index}"
        if self.mode == "soft" and not linux:
            print("Windows 不支援 soft 曝光，改用相機內建自動曝光")
            self.mode = "auto"
        if linux:
            # 白平衡交給相機自動（別的程式可能把它鎖在手動，例如舊的 AprilTag 系統鎖在 3500K，畫面會偏藍綠）
            self._v4l2(white_balance_automatic=1)
        if self.mode == "auto":
            if linux:
                self._v4l2(auto_exposure=3)               # Aperture Priority = 相機自動
            else:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        elif self.mode == "soft":
            self.exp = 157                                 # 這顆相機的出廠預設，從這裡開始調
            # 手動模式 + 關掉「暗時自動降幀率」，幀率才穩定
            self._v4l2(auto_exposure=1, exposure_dynamic_framerate=0)
            self._v4l2(exposure_time_absolute=self.exp)
        else:
            self.exp = int(float(self.mode))
            if linux:
                self._v4l2(auto_exposure=1, exposure_dynamic_framerate=0)
                self._v4l2(exposure_time_absolute=self.exp)
            else:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                cap.set(cv2.CAP_PROP_EXPOSURE, self.exp)
        print(f"曝光模式：{self.mode}" + (f"，起始曝光 {self.exp}" if self.exp is not None else ""))

    def desc(self):
        return f"exp {self.mode}" + (f" {self.exp}" if self.mode == "soft" else "")

    def update(self, img, holes):
        """soft 模式：每 every 幀依測光區域調一次曝光。"""
        if self.mode != "soft":
            return
        self._n += 1
        if self._n % self.every:
            return
        h, w = img.shape[:2]
        pts = [(hh.cx, hh.cy, hh.diam) for hh in holes if np.isfinite(hh.cx)]
        if len(pts) >= 3:      # 板子區域：所有孔的外框再往外擴 1.5 個孔徑
            d = max(p[2] for p in pts) * 1.5
            x0, x1 = int(max(min(p[0] for p in pts) - d, 0)), int(min(max(p[0] for p in pts) + d, w))
            y0, y1 = int(max(min(p[1] for p in pts) - d, 0)), int(min(max(p[1] for p in pts) + d, h))
        else:                  # 沒看到板子：畫面中央 60%
            x0, x1, y0, y1 = int(w * 0.2), int(w * 0.8), int(h * 0.2), int(h * 0.8)
        roi = cv2.cvtColor(img[y0:y1:2, x0:x1:2], cv2.COLOR_BGR2GRAY)
        if roi.size == 0:
            return
        mean = float(roi.mean())
        sat = float((roi >= 250).mean())
        ratio = self.target / max(mean, 1.0)
        if sat > 0.02:         # 過曝像素超過 2%：不管平均亮度，先往下調
            ratio = min(ratio, 0.8)
        if 0.88 < ratio < 1.12:
            return
        new = int(round(min(max(self.exp * min(max(ratio, 0.6), 1.6), self.min_exp), self.max_exp)))
        if new != self.exp:
            self.exp = new
            self._v4l2(exposure_time_absolute=new)


class FrameGrabber:
    """另一條執行緒一直讀相機，只留最新的一幀（含拍照時間、拍照當下的仰角讀數）。
    主迴圈拿到的永遠是最新畫面，不會因為辨識比相機慢而處理到舊畫面；仰角也和畫面同一時間讀，排號判斷才準。"""

    def __init__(self, cap, hood_fn=None):
        self.cap, self.hood_fn = cap, hood_fn
        self.cond = threading.Condition()
        self.item, self.seq = None, 0
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self.running:
            ok, img = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            t_cap = time.time()
            hood = self.hood_fn() if self.hood_fn else None
            with self.cond:
                self.item, self.seq = (img, t_cap, hood), self.seq + 1
                self.cond.notify_all()

    def frames(self):
        last = 0
        while True:
            with self.cond:
                self.cond.wait_for(lambda: self.seq != last)
                last = self.seq
                item = self.item
            yield item

    def stop(self):
        self.running = False


def uvc_frames(index, w, h, hfov_deg, rotate180=False, exposure=None, mirror=False, fx1280=None, hood_fn=None):
    """一般攝影機（OpenCV 開得到的都行，例如 Orbbec 的彩色 UVC）。只有彩色、沒有深度：
    孔的偏角／仰角照算，距離與高度為 NaN（solvePnP 會補上）。內參用 --fx 或水平視角估。exposure：ExposureControl（可省略）。"""
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L2
    if isinstance(index, str):      # 固定路徑（例如 /dev/v4l/by-id/...-video-index0），重插 USB 後編號變了也找得到
        index = os.path.realpath(index)
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        raise RuntimeError(f"開不了攝影機 index={index}")
    # 先要求 MJPEG：多數 USB 相機在 1280x720 用未壓縮格式只有 10 fps，MJPEG 才有 30 fps
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # 緩衝區至少要 2 個：只有 1 個時，程式把緩衝區還回驅動之前相機拍好的下一幀沒地方放，會被丟掉，
    # 幀率剛好砍半（2026-11-20 在 Orin 實測：1 個 = 15 fps，2 個以上 = 30 fps）。
    # 取像執行緒（FrameGrabber）一直把幀讀走，排隊中不會累積舊畫面，所以 2 個不會增加延遲。
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    if exposure is not None:
        exposure.attach(cap, index)
    grab = FrameGrabber(cap, hood_fn)
    K = None
    try:
        for img, t_cap, hood in grab.frames():
            if rotate180:
                img = cv2.rotate(img, cv2.ROTATE_180)
            if mirror:
                # 相機輸出左右相反時翻回來。一定要在辨識前做：否則偏角正負號、左中右欄編號都會反過來
                img = cv2.flip(img, 1)
            ih, iw = img.shape[:2]
            if K is None or K[2] != iw / 2 or K[3] != ih / 2:
                # 焦距：有校正值（--fx，以 1280 寬為準）就用，否則用水平視角估
                fx = fx1280 * iw / 1280.0 if fx1280 else (iw / 2) / math.tan(math.radians(hfov_deg) / 2)
                K = (fx, fx, iw / 2, ih / 2)
            yield img, None, K, t_cap, hood
    finally:
        grab.stop()
        cap.release()


def image_frames(path, hfov_deg):
    files = sorted(glob.glob(os.path.join(path, "*.jpg")) + glob.glob(os.path.join(path, "*.png")))
    for f in files:
        img = cv2.imdecode(np.fromfile(f, np.uint8), cv2.IMREAD_COLOR)  # imread 在 Windows 讀不了中文檔名
        if img is None:
            continue
        h, w = img.shape[:2]
        fx = (w / 2) / math.tan(math.radians(hfov_deg) / 2)
        yield os.path.basename(f), img, None, (fx, fx, w / 2, h / 2)


class Watchdog:
    """看門狗：主迴圈每收到一幀就 beat()；太久沒有新畫面（相機卡住、USB 斷線、辨識卡死）就結束程式，
    交給 systemd（Restart=always）重新啟動。第一幀前給較長的 startup 秒數（載入 TensorRT 引擎要時間）。"""

    def __init__(self, timeout, startup=30.0, poll=1.0):
        self.timeout, self.startup, self.poll = timeout, max(startup, timeout), poll
        self.t0 = time.time()
        self.t = None
        threading.Thread(target=self._run, daemon=True).start()

    def beat(self):
        self.t = time.time()

    def _run(self):
        while True:
            time.sleep(self.poll)
            last = self.t if self.t is not None else self.t0
            limit = self.timeout if self.t is not None else self.startup
            idle = time.time() - last
            if idle > limit:
                print(f"[watchdog] {idle:.0f} 秒沒有新畫面，結束程式讓 systemd 重啟", flush=True)
                os._exit(3)


# ----------------------------------------------------------------------------
# 主程式
# ----------------------------------------------------------------------------
def load_calibration(path):
    """讀 calibration.json（以 _ 開頭的欄位是說明，略過）；檔案不存在就回傳空 dict。"""
    try:
        with open(path, encoding="utf-8") as f:
            return {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    except FileNotFoundError:
        return {}


def json_result(name, res):
    """images 模式印出的一行 JSON（NaN 改成 null、陣列取 3 位小數）。"""
    def clean(v):
        if isinstance(v, list):
            return [None if (isinstance(x, float) and math.isnan(x)) else round(x, 3) for x in v]
        return v
    return json.dumps({"image": name, **{k: clean(v) for k, v in res.items()}}, ensure_ascii=False)


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", choices=["realsense", "images", "uvc"], default="realsense")
    ap.add_argument("--path", help="images 模式的資料夾")
    ap.add_argument("--cam-index", type=lambda v: int(v) if v.isdigit() else v, default=0,
                    help="uvc 模式的攝影機：編號，或固定路徑如 /dev/v4l/by-id/...-video-index0（建議，編號會因重插 USB 而改變）")
    ap.add_argument("--watchdog", type=float, default=10.0,
                    help="看門狗：超過這麼多秒沒有新畫面就結束程式（交給 systemd 重啟）；0 = 關閉。只在 uvc／realsense 即時模式有效")
    ap.add_argument("--cam-size", default="1280x800", help="uvc 模式的解析度，寬x高")
    ap.add_argument("--rotate180", action="store_true", help="相機倒裝時把畫面轉 180 度")
    ap.add_argument("--mirror", action="store_true", help="相機畫面左右相反時，左右鏡像翻轉回來")
    ap.add_argument("--exposure", default="soft",
                    help="uvc 模式的曝光：soft = 程式依板子亮度自動調（預設）、auto = 相機內建自動、數字 = 固定曝光值")
    ap.add_argument("--ae-target", type=float, default=110.0, help="soft 曝光的目標亮度（0~255，板子區域的平均）")
    ap.add_argument("--ae-max-exp", type=int, default=330,
                    help="soft 曝光的上限（Linux exposure_time_absolute，330 ≈ 33 ms，再高會掉到 30 fps 以下）")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--team", help="隊號（連 roboRIO）")
    ap.add_argument("--server", help="直接指定 roboRIO IP")
    ap.add_argument("--no-nt", action="store_true")
    ap.add_argument("--ignore-enable", action="store_true", help="忽略 roboRIO 的 Vision/enable 開關，永遠拍照辨識（桌上測試用）")
    ap.add_argument("--calib", default=os.path.join(HERE, "calibration.json"),
                    help="相機校正檔（焦距、距離修正等）。下面幾個參數有給就覆蓋檔案裡的值")
    ap.add_argument("--camera-pitch-deg", type=float, default=None, help="相機向上仰的角度（只用在有深度的 RealSense）")
    ap.add_argument("--hfov-deg", type=float, default=None, help="水平視角（images 模式，或沒有 --fx 時估焦距）")
    ap.add_argument("--dist-scale", type=float, default=None, help="距離經驗修正：實際 = scale * PnP 讀數 + offset（見 calibration.json）")
    ap.add_argument("--dist-offset", type=float, default=None, help="距離經驗修正的偏移（公尺）")
    ap.add_argument("--fx", type=float, default=None, help="uvc 相機焦距（像素，以 1280 寬為準）；0 = 改用 --hfov-deg 估")
    ap.add_argument("--red-min", type=float, default=0.04)
    ap.add_argument("--red-second-ratio", type=float, default=0.5)
    ap.add_argument("--debug-dir")
    ap.add_argument("--mjpeg-port", type=int, default=0, help="大於 0 就在這個 port 提供疊圖的 MJPEG 串流（儀表板用，建議 1181）")
    ap.add_argument("--mjpeg-width", type=int, default=640, help="串流畫面縮到這個寬度再編碼；0 = 原尺寸")
    ap.add_argument("--cam-y-ref", type=float, default=None,
                    help="鏡頭在板子座標的高度基準（公尺，小孔排中心為 0）。通常不用給：看到小孔時會自動學並存到 row_anchor.json")
    ap.add_argument("--row-anchor", default=os.path.join(HERE, "row_anchor.json"),
                    help="排號基準的存檔位置；給空字串就不讀也不存（離線測試用）")
    a = ap.parse_args(argv)
    calib = {**CALIB_DEFAULTS, **load_calibration(a.calib)}
    for k in CALIB_DEFAULTS:
        if getattr(a, k) is None:
            setattr(a, k, float(calib[k]))
    return a


def load_model(path):
    if path.endswith((".engine", ".onnx")):
        # TensorRT engine / ONNX 走 trt_seg（不 import PyTorch；Orin 的 JetPack 6.0 沒有可用的 GPU 版 PyTorch）
        from trt_seg import SegModel
        return SegModel(path)
    from ultralytics import YOLO
    return YOLO(path, task="segment")


def main(argv=None):
    a = parse_args(argv)
    anchor = RowAnchor(a.row_anchor or None, a.cam_y_ref)
    pipeline = VisionPipeline(load_model(a.model), a, anchor)
    sink = None if a.no_nt else make_nt_sink(a.team, a.server)
    hood_fn = sink.hood_deg if sink is not None else None
    if a.debug_dir:
        os.makedirs(a.debug_dir, exist_ok=True)

    if a.source == "images":
        for name, img, depth, K in image_frames(a.path, a.hfov_deg):
            res, holes = pipeline.process(img, depth, K)
            print(json_result(name, res))
            if a.debug_dir:
                cv2.imwrite(os.path.join(a.debug_dir, name), draw_debug(img, holes, res))
        return

    stream = MJPEGServer(a.mjpeg_port, width=a.mjpeg_width) if a.mjpeg_port > 0 else None
    t_stream = 0.0
    t_dbg = 0.0
    t_prev, fps = time.time(), 0.0
    was_enabled = True
    expo = None
    if a.source == "uvc":
        cw, ch = (int(v) for v in a.cam_size.lower().split("x"))
        expo = ExposureControl(a.exposure, a.ae_target, a.ae_max_exp)
        frames = uvc_frames(a.cam_index, cw, ch, a.hfov_deg, a.rotate180, expo, a.mirror, a.fx or None, hood_fn)
    else:
        frames = realsense_frames(hood_fn=hood_fn)
    wd = Watchdog(a.watchdog) if a.watchdog > 0 else None
    for img, depth, K, t_cap, hood in frames:
        if wd:
            wd.beat()
        # roboRIO 說還不用看（機器人還在移動）：丟掉這一幀，不跑辨識也不發布
        # 相機照常取流（不然緩衝會塞滿），重新開啟時馬上就有最新的畫面
        enabled = a.ignore_enable or sink is None or sink.enabled()
        if not enabled:
            was_enabled = False
            continue
        if not was_enabled:      # 剛被重新開啟：重算 FPS 的起點
            t_prev, fps, was_enabled = time.time(), 0.0, True
        res, holes = pipeline.process(img, depth, K, hood, t_cap)
        if expo is not None:
            expo.update(img, holes)
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3)
        t_prev = now
        if sink:
            sink.publish(res, fps)
        want_stream = stream and now - t_stream > 1 / 15   # 串流最多 15 fps，省 CPU
        want_dbg = a.debug_dir and now - t_dbg > 1.0         # 除錯圖每秒存一次就好，存檔很花時間
        if want_stream or want_dbg:
            vis = draw_debug(img, holes, res)
            if expo is not None:
                cv2.putText(vis, expo.desc(), (10, vis.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            draw_aim(vis, holes, res)
            if want_stream:
                stream.update(vis)
                t_stream = now
            if want_dbg:
                cv2.imwrite(os.path.join(a.debug_dir, "latest.jpg"), vis)
                cv2.imwrite(os.path.join(a.debug_dir, "latest_raw.jpg"), img)   # 沒疊圖的原始畫面（校正焦距用）
                t_dbg = now


if __name__ == "__main__":
    main()
