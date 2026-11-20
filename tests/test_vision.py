# -*- coding: utf-8 -*-
"""vision_node 的回歸測試（不需要 pytest，用內建 unittest）：

    .venv_vision\\Scripts\\python.exe -m unittest discover -s tests -v

1. 合成板子：用已知的相機位姿把 12 個孔投影到畫面上，檢查排格子、solvePnP、看不到小孔時的排號推斷。
   不需要模型或照片，任何電腦都能跑。
2. 實拍照片：data_robotcam/ 與 ONNX 模型都在時，整條流程的輸出要和 tests/expected/robotcam_onnx.jsonl 一致
   （改了排格子或位姿的邏輯、結果「應該」改變時，用 tests/make_expected.py 重新產生）。
"""
import json
import math
import os
import sys
import unittest

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "integration", "orin"))
import vision_node as vn  # noqa: E402

W, H = 1280, 800
FX = 1057.0
K = (FX, FX, W / 2, H / 2)
CAM = np.array([[FX, 0, W / 2], [0, FX, H / 2], [0, 0, 1]], np.float64)


def camera_pose(dist=4.0, height=-0.6, side=0.0, yaw_deg=0.0, pitch_deg=0.0):
    """相機在板子座標 (side, height, dist)，光軸朝板子（-Z），可再左右轉 yaw、上下仰 pitch。回傳 rvec, tvec。"""
    base = np.diag([1.0, -1.0, -1.0])                     # 板子 X 右 Y 上 Z 朝相機 -> 相機 x 右 y 下 z 前
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    Ry = np.array([[math.cos(y), 0, -math.sin(y)], [0, 1, 0], [math.sin(y), 0, math.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(p), math.sin(p)], [0, -math.sin(p), math.cos(p)]])
    R = Rx @ Ry @ base
    C = np.array([side, height, dist])
    rvec, _ = cv2.Rodrigues(R)
    return rvec, (-R @ C).reshape(3, 1)


def synth_holes(rvec, tvec, indices=range(12)):
    """把孔投影成多邊形，做成 Hole（大孔直徑 40 cm、小孔 20 cm）。回傳 (holes, 各孔真正的 index)。"""
    holes, truth = [], []
    ang = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    for i in indices:
        r = 0.10 if i < 3 else 0.20
        c = vn.board_point(i)
        pts3 = np.stack([c[0] + r * np.cos(ang), c[1] + r * np.sin(ang), np.zeros_like(ang)], 1)
        img, _ = cv2.projectPoints(pts3, rvec, tvec, CAM, None)
        poly = img.reshape(-1, 2)
        if poly[:, 0].min() < 0 or poly[:, 1].min() < 0 or poly[:, 0].max() >= W or poly[:, 1].max() >= H:
            continue                                           # 出了畫面
        holes.append(vn.Hole(vn.SMALL if i < 3 else vn.BIG, poly, (H, W)))
        truth.append(i)
    return holes, truth


def indices_of(holes):
    return [h.row * 3 + h.col if h.row >= 0 else -1 for h in holes]


class SyntheticBoardTest(unittest.TestCase):
    def test_full_board_grid(self):
        holes, truth = synth_holes(*camera_pose())
        self.assertEqual(len(holes), 12)
        self.assertTrue(vn.assign_grid(holes))
        self.assertEqual(indices_of(holes), truth)

    def test_skewed_view_grid(self):
        """車頭斜看板子（側移 1 m、轉 8 度朝向板子）：透視讓排傾斜，仍要排對。"""
        holes, truth = synth_holes(*camera_pose(dist=3.5, side=1.0, yaw_deg=-8.0))
        self.assertEqual(len(holes), 12)
        self.assertTrue(vn.assign_grid(holes))
        self.assertEqual(indices_of(holes), truth)

    def test_pnp_distance_and_bearing(self):
        rvec, tvec = camera_pose(dist=4.0, height=-0.6, side=0.3)
        holes, _ = synth_holes(rvec, tvec)
        vn.assign_grid(holes)
        pnp = vn.board_pnp(holes, K)
        self.assertIsNotNone(pnp)
        targets, err, board_dist, elev, cam_y = pnp
        self.assertLess(err, 1.0)                              # 合成資料：重投影誤差應該很小
        self.assertAlmostEqual(board_dist, 4.0, delta=0.03)
        self.assertAlmostEqual(cam_y, -0.6, delta=0.03)
        # 中欄小孔（index 1）在相機左邊 0.3 m、前方 4 m：偏角約 -4.3 度
        self.assertAlmostEqual(targets[1][0], -math.degrees(math.atan2(0.3, 4.0)), delta=0.3)

    def test_no_small_row_uses_anchor(self):
        """看不到小孔和第 1 排（仰角放低瞄最下排）：排號要靠鏡頭高度基準推回來，不能把第 2、3 排當成第 1、2 排。"""
        rvec, tvec = camera_pose(dist=3.0, height=-0.6, pitch_deg=-12.0)
        holes, truth = synth_holes(rvec, tvec, indices=range(6, 12))
        self.assertTrue(vn.assign_grid(holes))
        anchor = vn.RowAnchor(None, default=-0.6)               # 之前看到小孔時學到的鏡頭高度
        pnp, shift, method = vn.solve_board(holes, K, anchor)
        self.assertIsNotNone(pnp)
        self.assertEqual(method, "camY")
        self.assertEqual(shift, 1)
        self.assertEqual(indices_of(holes), truth)

    def test_too_few_holes(self):
        holes, _ = synth_holes(*camera_pose(), indices=[3, 4, 5])
        self.assertFalse(vn.assign_grid(holes))


class PipelineTest(unittest.TestCase):
    def test_hood_still_detection(self):
        class A:  # 只需要 VisionPipeline 讀的參數
            imgsz, conf, red_min, red_second_ratio = 640, 0.4, 0.04, 0.5
            camera_pitch_deg, dist_scale, dist_offset = 0.0, 1.0, 0.0
        p = vn.VisionPipeline(None, A())
        self.assertFalse(any(p._hood_still(t, 20.0) for t in (0.0, 0.1)))   # 樣本不夠
        self.assertTrue(p._hood_still(0.25, 20.1))                           # 0.25 秒內只動 0.1 度
        self.assertFalse(p._hood_still(0.3, 25.0))                           # 突然動了 5 度


class RecordedImagesTest(unittest.TestCase):
    IMAGES = os.path.join(ROOT, "data_robotcam")
    MODEL = os.path.join(ROOT, "runs", "yolo11n_seg_v1", "export", "best.onnx")
    EXPECTED = os.path.join(ROOT, "tests", "expected", "robotcam_onnx.jsonl")

    @unittest.skipUnless(os.path.isdir(IMAGES) and os.path.exists(MODEL), "沒有 data_robotcam/ 或 best.onnx（不在 repo 裡）")
    def test_matches_expected(self):
        a = vn.parse_args(["--model", self.MODEL, "--source", "images", "--no-nt", "--row-anchor", ""])
        pipeline = vn.VisionPipeline(vn.load_model(self.MODEL), a, vn.RowAnchor(None, a.cam_y_ref))
        with open(self.EXPECTED, encoding="utf-8") as f:
            expected = {d["image"]: d for d in map(json.loads, f)}
        n = 0
        for name, img, depth, K_ in vn.image_frames(self.IMAGES, a.hfov_deg):
            got = json.loads(vn.json_result(name, pipeline.process(img, depth, K_)[0]))
            got.pop("latency_ms", None)
            exp = expected[name]
            with self.subTest(image=name):
                for k in ("grid_ok", "lit_small", "n_big", "n_small", "row_method", "row_shift"):
                    self.assertEqual(got[k], exp[k], k)
                for k in ("bearing", "pitch", "dist", "elev"):
                    for x, y in zip(got[k], exp[k]):
                        if x is None or y is None:
                            self.assertEqual(x, y, k)
                        else:
                            self.assertAlmostEqual(x, y, delta=0.01, msg=k)
            n += 1
        self.assertEqual(n, len(expected))


if __name__ == "__main__":
    unittest.main()
