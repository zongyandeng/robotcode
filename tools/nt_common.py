# -*- coding: utf-8 -*-
"""記錄器共用：隊號 -> roboRIO 位址、記錄檔位置。

隊號只寫在這裡一次（TEAM）；其他地方的預設位址：
  - integration/orin/run_vision.sh 的 --team
  - dashboard/dashboard.html 右上角的 roboRIO 欄位（預設值寫在 HTML，改過會記在瀏覽器）
"""
import argparse
import os
import time

TEAM = 9427
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, "logs")


def team_server(team):
    """隊號 -> roboRIO 的 IP（10.TE.AM.2）。"""
    t = int(team)
    return f"10.{t // 100}.{t % 100}.2"


def server_from_args(description):
    """命令列：可以直接給位址（舊用法 `python xxx_logger.py 10.94.27.2`），或用 --team；都沒給就用 TEAM。"""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("server", nargs="?", help=f"roboRIO 位址（預設由隊號推得：{team_server(TEAM)}）")
    ap.add_argument("--team", type=int, default=TEAM, help=f"隊號（預設 {TEAM}）")
    a = ap.parse_args()
    return a.server or team_server(a.team)


def log_path(prefix):
    """logs/<prefix>_<日期時間>.csv（資料夾不存在就建立）。"""
    os.makedirs(LOG_DIR, exist_ok=True)
    return os.path.join(LOG_DIR, time.strftime(f"{prefix}_%Y%m%d_%H%M%S.csv"))
