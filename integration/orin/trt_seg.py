# -*- coding: utf-8 -*-
"""YOLO11-seg 推論，不需要 GPU 版 PyTorch：
  - .engine：TensorRT（Orin 上用；CUDA 記憶體用 ctypes 呼叫 libcudart，不需 pycuda / cuda-python）
  - .onnx  ：ONNX Runtime（筆電上驗證前後處理用）

前處理（letterbox）與後處理（NMS、遮罩）照 Ultralytics 的預設做法：
letterbox 到 imgsz x imgsz、灰邊 114；conf 門檻後各類別分開做 NMS (IoU 0.7)；
遮罩 = sigmoid(係數 @ protos)，裁到框內、放大回原圖，取 0.5 門檻後的最大輪廓當多邊形。

類別名稱：讀與模型同名的 .json（例如 best.engine -> best.json：{"names": {"0": "hole_big", ...}}），
沒有就用 ONNX 的 metadata，再沒有就用 {0: hole_big, 1: hole_small}。
"""
import ctypes
import json
import os

import cv2
import numpy as np

DEFAULT_NAMES = {0: "hole_big", 1: "hole_small"}


def _load_names(model_path, fallback=None):
    j = os.path.splitext(model_path)[0] + ".json"
    if os.path.exists(j):
        with open(j, encoding="utf-8") as f:
            return {int(k): v for k, v in json.load(f)["names"].items()}
    return fallback or DEFAULT_NAMES


# ---------------------------------------------------------------------------- backends
class ORTBackend:
    def __init__(self, path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.imgsz = self.sess.get_inputs()[0].shape[2]
        meta = self.sess.get_modelmeta().custom_metadata_map
        fb = None
        if "names" in meta:
            import ast
            fb = {int(k): v for k, v in ast.literal_eval(meta["names"]).items()}
        self.names = _load_names(path, fb)

    def infer(self, blob):
        out0, out1 = self.sess.run(None, {self.inp: blob})
        return out0, out1


class TRTBackend:
    def __init__(self, path):
        import tensorrt as trt
        self.cudart = _load_cudart()
        logger = trt.Logger(trt.Logger.WARNING)
        with open(path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("TensorRT engine 載入失敗（engine 要在同一台 Orin、同版 TensorRT 上建）")
        self.ctx = self.engine.create_execution_context()
        self.names = _load_names(path)
        self.bufs = {}   # name -> (host ndarray, device ptr, is_input)
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            host = np.empty(shape, dtype=dtype)
            ptr = ctypes.c_void_p()
            _check(self.cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(host.nbytes)), "cudaMalloc")
            self.ctx.set_tensor_address(name, ptr.value)
            is_in = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            self.bufs[name] = (host, ptr, is_in)
            if is_in:
                self.inp = name
                self.imgsz = shape[2]
        outs = [n for n, (_, _, i) in self.bufs.items() if not i]
        # output0 是 (1, 4+nc+32, N)，output1 是 (1, 32, h, w) 的 protos
        self.out_det = next(n for n in outs if len(self.bufs[n][0].shape) == 3)
        self.out_proto = next(n for n in outs if len(self.bufs[n][0].shape) == 4)
        self.stream = ctypes.c_void_p()
        _check(self.cudart.cudaStreamCreate(ctypes.byref(self.stream)), "cudaStreamCreate")

    def infer(self, blob):
        host, ptr, _ = self.bufs[self.inp]
        np.copyto(host, blob.astype(host.dtype, copy=False))
        _check(self.cudart.cudaMemcpyAsync(ptr, host.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(host.nbytes),
                                           1, self.stream), "H2D")
        if not self.ctx.execute_async_v3(self.stream.value):
            raise RuntimeError("TensorRT execute_async_v3 失敗")
        for n in (self.out_det, self.out_proto):
            h, p, _ = self.bufs[n]
            _check(self.cudart.cudaMemcpyAsync(h.ctypes.data_as(ctypes.c_void_p), p, ctypes.c_size_t(h.nbytes),
                                               2, self.stream), "D2H")
        _check(self.cudart.cudaStreamSynchronize(self.stream), "sync")
        return (self.bufs[self.out_det][0].astype(np.float32, copy=False),
                self.bufs[self.out_proto][0].astype(np.float32, copy=False))


def _load_cudart():
    for name in ("libcudart.so.12", "/usr/local/cuda/lib64/libcudart.so.12", "libcudart.so"):
        try:
            lib = ctypes.CDLL(name)
            lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
            lib.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
            lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
            lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
            return lib
        except OSError:
            continue
    raise OSError("找不到 libcudart.so.12（確認 /usr/local/cuda/lib64 存在）")


def _check(err, what):
    if err != 0:
        raise RuntimeError(f"CUDA {what} 失敗，錯誤碼 {err}")


# ---------------------------------------------------------------------------- 模型
class SegModel:
    """predict_polys(frame) -> [(類別名稱, 多邊形 Nx2 float32（原圖座標）), ...]"""

    def __init__(self, path, iou=0.7, mask_thr=0.5):
        self.backend = TRTBackend(path) if path.endswith(".engine") else ORTBackend(path)
        self.names = self.backend.names
        self.imgsz = self.backend.imgsz
        self.iou, self.mask_thr = iou, mask_thr

    def _letterbox(self, img):
        h, w = img.shape[:2]
        s = self.imgsz
        r = min(s / h, s / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        dw, dh = (s - nw) / 2, (s - nh) / 2
        top, left = int(round(dh - 0.1)), int(round(dw - 0.1))
        if (nw, nh) != (w, h):
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        out = np.full((s, s, 3), 114, np.uint8)
        out[top:top + nh, left:left + nw] = img
        return out, r, left, top, nw, nh

    def predict_polys(self, frame, conf=0.4):
        lb, r, left, top, nw, nh = self._letterbox(frame)
        blob = np.ascontiguousarray(lb[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        det, protos = self.backend.infer(blob)
        det = det[0].T                          # (N, 4+nc+32)
        nc = len(self.names)
        scores_all = det[:, 4:4 + nc]
        cls = scores_all.argmax(1)
        scores = scores_all[np.arange(len(cls)), cls]
        keep = scores > conf
        if not keep.any():
            return []
        det, cls, scores = det[keep], cls[keep], scores[keep]
        cx, cy, bw, bh = det[:, 0], det[:, 1], det[:, 2], det[:, 3]
        x1, y1 = cx - bw / 2, cy - bh / 2
        boxes_xywh = np.stack([x1, y1, bw, bh], 1)
        idx = cv2.dnn.NMSBoxesBatched(boxes_xywh.tolist(), scores.tolist(), cls.tolist(), conf, self.iou)
        idx = np.array(idx).reshape(-1)
        if idx.size == 0:
            return []
        det, cls, boxes_xywh = det[idx], cls[idx], boxes_xywh[idx]

        p = protos[0]                           # (32, mh, mw)
        c, mh, mw = p.shape
        coef = det[:, 4 + nc:]
        s = self.imgsz
        sx, sy = mw / s, mh / s
        # protos 到 letterbox 通常剛好放大整數倍（160 -> 640 = 4 倍）：這時只在「框附近的小區塊」算 sigmoid、放大、找輪廓，
        # 結果與整張 640x640 放大相同（雙線性放大的取樣位置一致），但每個孔只處理自己那一小塊，CPU 省很多
        up_k = s // mw if (s % mw == 0 and s % mh == 0 and s // mw == s // mh) else 0
        H, W = frame.shape[:2]
        polys = []
        for k in range(len(idx)):
            bx, by, bw_, bh_ = boxes_xywh[k]
            # 在 protos 尺度裁到框內，與 Ultralytics crop_mask 相同：保留 x1 <= 像素索引 < x2
            xa, ya = max(int(np.ceil(bx * sx)), 0), max(int(np.ceil(by * sy)), 0)
            xb, yb = min(int(np.ceil((bx + bw_) * sx)), mw), min(int(np.ceil((by + bh_) * sy)), mh)
            if xb <= xa or yb <= ya:
                continue
            if up_k:
                # 區塊外圍多留 1 格 0（框外的值本來就是 0），碰到 protos 邊界時不留（cv2.resize 在邊界是複製最外圈，和整張放大一致）
                x0, y0 = max(xa - 1, 0), max(ya - 1, 0)
                x1, y1 = min(xb + 1, mw), min(yb + 1, mh)
                crop = np.zeros((y1 - y0, x1 - x0), np.float32)
                logit = coef[k] @ p[:, ya:yb, xa:xb].reshape(c, -1)
                crop[ya - y0:yb - y0, xa - x0:xb - x0] = (1.0 / (1.0 + np.exp(-logit))).reshape(yb - ya, xb - xa)
                up = cv2.resize(crop, ((x1 - x0) * up_k, (y1 - y0) * up_k), interpolation=cv2.INTER_LINEAR)
                off = np.array([x0 * up_k, y0 * up_k], np.float32)
            else:
                # 非整數倍（少見）：照原本做法整張放大
                crop = np.zeros((mh, mw), np.float32)
                logit = coef[k] @ p[:, ya:yb, xa:xb].reshape(c, -1)
                crop[ya:yb, xa:xb] = (1.0 / (1.0 + np.exp(-logit))).reshape(yb - ya, xb - xa)
                up = cv2.resize(crop, (s, s), interpolation=cv2.INTER_LINEAR)
                off = np.zeros(2, np.float32)
            # 門檻化後找輪廓，再把座標換回原圖（與 Ultralytics masks.xy 相同）
            mask = (up > self.mask_thr).astype(np.uint8)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            c_ = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32) + off
            poly = (c_ - np.array([left, top], np.float32)) / r
            poly[:, 0] = poly[:, 0].clip(0, W - 1)
            poly[:, 1] = poly[:, 1].clip(0, H - 1)
            if len(poly) >= 3:
                polys.append((self.names.get(int(cls[k]), str(int(cls[k]))), poly))
        return polys
