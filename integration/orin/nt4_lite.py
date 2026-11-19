# -*- coding: utf-8 -*-
"""極簡的 NetworkTables 4 客戶端（純 Python），給裝不了 pyntcore 的環境用。

為什麼需要：Orin 上 NVIDIA 的 PyTorch 只有 Python 3.10 版，而 RobotPy 的 pyntcore 沒有出
Python 3.10 + aarch64 的套件。這個檔案只用 websocket-client 和 msgpack，照 NT4 協定
（WPILib networktables4.adoc）實作 vision_node.py 用得到的部分：
  - 發布 boolean / int / double / double[] 等 topic
  - 訂閱 topic 並讀最新值（例如 Vision/enable）
  - 與伺服器對時（送出的值要帶伺服器時間）
  - 斷線自動重連，重連後重新發布、重新訂閱，並補送最後的值

用法：
    nt = NT4Lite("10.94.27.2", "orin-vision")
    fid = nt.publish("/Vision/frame_id", "int")
    en = nt.subscribe("/Vision/enable", default=True)
    fid.set(1); nt.flush()      # flush 把這段時間 set 的值一次送出（同一個封包，順序不變）
    en.get()
"""
import json
import threading
import time

import msgpack
import websocket

TYPE_IDS = {"boolean": 0, "double": 1, "int": 2, "float": 3, "string": 4, "json": 4, "raw": 5,
            "boolean[]": 16, "double[]": 17, "int[]": 18, "float[]": 19, "string[]": 20}
_COERCE = {
    "boolean": bool, "double": float, "int": int, "float": float, "string": str,
    "boolean[]": lambda v: [bool(x) for x in v], "double[]": lambda v: [float(x) for x in v],
    "int[]": lambda v: [int(x) for x in v], "float[]": lambda v: [float(x) for x in v],
    "string[]": lambda v: [str(x) for x in v],
}
SUBPROTOCOLS = ["v4.1.networktables.first.wpi.edu", "networktables.first.wpi.edu"]


def _now_us():
    return int(time.monotonic() * 1e6)


class _Pub:
    def __init__(self, nt, name, typ, uid):
        self.nt, self.name, self.type, self.uid = nt, name, typ, uid
        self.value = None

    def set(self, v):
        self.value = _COERCE[self.type](v)
        with self.nt._lock:
            self.nt._pending[self.uid] = self.value


class _Sub:
    def __init__(self, name, default):
        self.name, self.default = name, default
        self.value = None

    def get(self):
        return self.default if self.value is None else self.value


class NT4Lite:
    def __init__(self, server, name="orin-vision", port=5810):
        self.url = f"ws://{server}:{port}/nt/{name}"
        self._pubs = []           # [_Pub]
        self._subs = {}           # name -> _Sub
        self._ids = {}            # 伺服器 topic id -> name
        self._pending = {}        # pubuid -> value（下次 flush 送出）
        self._lock = threading.Lock()
        self._ws = None
        self._offset_us = 0       # 伺服器時間 - 本機時間
        # 收到第一次對時回覆前不能送值：伺服器只接受時間戳比目前值新的更新，
        # 若先用本機時間送出，時間戳可能遠超過伺服器時間，之後正確時間的值都會被當成「比較舊」而丟掉
        self._synced = False
        self.connected = False
        threading.Thread(target=self._run, daemon=True).start()

    # ---------------------------------------------------------------- 對外 API
    def publish(self, name, typ):
        p = _Pub(self, name, typ, len(self._pubs) + 1)
        self._pubs.append(p)
        self._send_text([self._publish_msg(p)])
        return p

    def subscribe(self, name, default=None):
        s = self._subs.setdefault(name, _Sub(name, default))
        self._send_text([self._subscribe_msg()])
        return s

    def flush(self):
        if not (self.connected and self._synced):
            return                # 還沒連線或還沒對時：值留在 _pending，對時完成後一起送
        with self._lock:
            pending, self._pending = self._pending, {}
        if not pending:
            return
        ts = _now_us() + self._offset_us
        by_uid = {p.uid: p for p in self._pubs}
        buf = b"".join(msgpack.packb([uid, ts, TYPE_IDS[by_uid[uid].type], v]) for uid, v in pending.items())
        self._send_bin(buf)

    # ---------------------------------------------------------------- 內部
    def _publish_msg(self, p):
        return {"method": "publish", "params": {"name": p.name, "pubuid": p.uid, "type": p.type, "properties": {}}}

    def _subscribe_msg(self):
        # periodic 0.02 秒：預設 0.1 秒太慢（仰角讀數用來判斷排號，要跟畫面同步）
        return {"method": "subscribe", "params": {"topics": list(self._subs), "subuid": 1, "options": {"periodic": 0.02}}}

    def _send_text(self, msgs):
        ws = self._ws
        if ws is not None and self.connected:
            try:
                with self._lock:
                    ws.send(json.dumps(msgs))
            except Exception:
                pass

    def _send_bin(self, data):
        ws = self._ws
        if ws is None:
            return
        try:
            with self._lock:
                ws.send(data, opcode=websocket.ABNF.OPCODE_BINARY)
        except Exception:
            self.connected = False

    def _time_sync(self):
        self._send_bin(msgpack.packb([-1, 0, TYPE_IDS["int"], _now_us()]))

    def _run(self):
        while True:
            try:
                ws = websocket.create_connection(self.url, subprotocols=SUBPROTOCOLS, timeout=3)
                ws.settimeout(None)
                self._ws, self.connected, self._ids, self._synced = ws, True, {}, False
                self._time_sync()
                # 重新發布、訂閱，並補送最後的值
                msgs = [self._publish_msg(p) for p in self._pubs]
                if self._subs:
                    msgs.append(self._subscribe_msg())
                if msgs:
                    with self._lock:
                        ws.send(json.dumps(msgs))
                with self._lock:
                    for p in self._pubs:
                        if p.value is not None:
                            self._pending.setdefault(p.uid, p.value)
                print(f"[nt4_lite] 已連線 {self.url}", flush=True)   # 最後的值等對時回覆後才補送
                last_sync = time.monotonic()
                while True:
                    op, data = ws.recv_data(control_frame=False)
                    if op == websocket.ABNF.OPCODE_TEXT:
                        self._on_text(data)
                    elif op == websocket.ABNF.OPCODE_BINARY:
                        self._on_bin(data)
                    if time.monotonic() - last_sync > 5:   # 定期重新對時
                        self._time_sync()
                        last_sync = time.monotonic()
            except Exception as e:
                if self.connected:
                    print(f"[nt4_lite] 斷線：{e}，1 秒後重連", flush=True)
                self.connected = False
                try:
                    if self._ws is not None:
                        self._ws.close()
                except Exception:
                    pass
                self._ws = None
                time.sleep(1.0)

    def _on_text(self, data):
        try:
            msgs = json.loads(data)
        except ValueError:
            return
        for m in msgs if isinstance(msgs, list) else [msgs]:
            if m.get("method") == "announce":
                self._ids[m["params"]["id"]] = m["params"]["name"]
            elif m.get("method") == "unannounce":
                self._ids.pop(m["params"]["id"], None)

    def _on_bin(self, data):
        up = msgpack.Unpacker(raw=False, strict_map_key=False)
        up.feed(data)
        for msg in up:
            if not isinstance(msg, (list, tuple)) or len(msg) != 4:
                continue
            tid, ts, _typ, val = msg
            if tid == -1:          # 對時回覆：val = 我們送出的本機時間
                rtt = _now_us() - int(val)
                self._offset_us = int(ts) + rtt // 2 - _now_us()
                if not self._synced:
                    self._synced = True
                    self.flush()       # 補送連線前後累積的值
                continue
            name = self._ids.get(tid)
            s = self._subs.get(name) if name else None
            if s is not None:
                s.value = val
