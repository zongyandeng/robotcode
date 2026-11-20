#!/bin/bash
# Orin 上啟動視覺節點（JetPack 6.0：TensorRT 引擎 + 純 Python NT4，不需要 PyTorch / pyntcore）
# 用法：./run_vision.sh [額外參數，會接在後面，例如 --ignore-enable 或 --rotate180]
cd "$(dirname "$0")"
# log 輪替：vision.service 把輸出附加到 vision.log，不處理會一直變大。超過 20 MB 時把舊內容存成 vision.log.1、清空 vision.log
# （systemd 以附加模式開檔，清空後會從頭繼續寫，不需要重開服務）
LOG="$PWD/vision.log"
if [ -f "$LOG" ] && [ "$(stat -c %s "$LOG")" -gt 20000000 ]; then
    cp "$LOG" "$LOG.1" && : > "$LOG"
fi
# TensorRT 的 Python 綁定是從 NVIDIA 官方 deb 解開到使用者目錄（不用 sudo）
export PYTHONPATH="$PWD/trt_py/root/usr/lib/python3.10/dist-packages:$PYTHONPATH"
exec python3 -u vision_node.py --model best.engine --source uvc --cam-index /dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._Adesso_CyberTrack_K1_SN0001-video-index0 --cam-size 1280x720 \
    --team 9427 --mjpeg-port 1181 --mirror --debug-dir /tmp/vision_dbg "$@"
# --mirror：這顆 USB 相機（Adesso CyberTrack K1）輸出的畫面左右相反，在辨識前翻回來
