# FRC 9427 視覺辨識與儀表板（TEL_YOLO）

FRC 9427 機器人的「上位」系統：用 **YOLO11n-seg** 辨識九宮格板上的 12 個孔與亮燈小孔，在 **Jetson Orin Nano** 上以 TensorRT 即時推論，把每個孔的角度與距離透過 **NetworkTables** 送給 roboRIO，並提供比賽用的瀏覽器儀表板。

> roboRIO 上的 Java 機器人程式（`Vision.java`、`AutoShoot.java`、`ShotTable.java` 等）在另一個 repo：[zongyandeng/robot-code](https://github.com/zongyandeng/robot-code)。

## 系統架構

```
USB 相機 ─► Orin Nano  integration/orin/vision_node.py
              YOLO11n-seg（TensorRT）→ 12 個孔排成 3x4 格 → solvePnP → 每孔偏角／仰角／距離、亮燈小孔
                 │  NetworkTables 4（/Vision/…）         │  MJPEG 串流（:1181）
                 ▼                                         ▼
            roboRIO（robot-code）  ◄──── NT4 ────►  筆電 dashboard/dashboard.html
              自主移動、瞄準、射擊                       狀態監看、調參、ShotTable 填表
```

孔的編號：`index = row * 3 + col`。row 0 = 最上排 3 個小孔，row 1~3 = 大孔由上到下；col 0~2 = 由左到右。

## 資料夾

| 路徑 | 內容 |
|---|---|
| [`integration/orin/`](integration/orin) | Orin 視覺節點：`vision_node.py`（主程式）、`trt_seg.py`（不需 PyTorch 的 TensorRT／ONNX 推論）、`nt4_lite.py`（純 Python NT4 客戶端）、`run_vision.sh`、`vision.service`（開機自動啟動） |
| [`dashboard/`](dashboard) | 比賽儀表板 `dashboard.html`（單一檔案、不需網路）、自主流程模擬器 `auto_sim.html`、沒有機器人時測試用的 `mock_robot.py`、一鍵開啟 DS＋儀表板的 `start_dashboard.ps1` |
| [`scripts/`](scripts) | 模型訓練流程：建資料集、訓練、驗證、混淆矩陣、找誤報、產生報告 |
| [`tools/`](tools) | NetworkTables 記錄器（`shoot_logger.py`、`auto_logger.py`）、SAM 標註審查工具、Orbbec 深度相機測試 |
| [`review/`](review) | 標註審查結果（`masks_r2.json` 是建資料集的來源） |
| [`report/`](report) | 模型訓練報告（HTML） |
| [`flowchart/`](flowchart) | 系統流程圖與產生腳本 |
| [`document/`](document) | 影片講稿等文件 |

詳細說明：
- **視覺整合、NT 表格式、Orin 安裝與校正** → [integration/README.md](integration/README.md)
- **儀表板使用方式** → [dashboard/README.md](dashboard/README.md)
- **新增機構的寫法（Java）** → [how_to_add_subsystem.md](how_to_add_subsystem.md)

## 快速開始

### 比賽時（筆電）

雙擊 `dashboard/開啟儀表板.bat`：自動開 Driver Station 並把儀表板擺在上方。roboRIO 預設 `10.94.27.2`，相機串流 `http://10.94.27.12:1181/stream.mjpg`。

### Orin 視覺節點

Orin 已設定開機自動啟動（systemd 使用者服務）。常用指令：

```bash
systemctl --user status vision       # 看狀態
tail -f ~/vision/vision.log          # 看 log
systemctl --user restart vision      # 換模型或改程式後重啟
```

手動執行（先 `systemctl --user stop vision`，否則會搶相機）：

```bash
~/vision/run_vision.sh --ignore-enable   # 桌上測試，不等 roboRIO 開啟
```

### 離線測試（任何電腦，不需要 roboRIO 或相機）

```bash
python integration/orin/vision_node.py --model best.pt --source images --path ./imgs --no-nt --debug-dir ./dbg
```

每張照片印出一行 JSON，並在 `./dbg` 輸出疊圖。

### 沒有機器人時測試儀表板

```powershell
cd dashboard
..\.venv_nt\Scripts\python.exe mock_robot.py
```

儀表板右上角的 roboRIO 位址改成 `127.0.0.1`，相機網址填 `http://127.0.0.1:1181/stream.mjpg`。

### 記錄射擊數據

```powershell
cd tools
..\.venv_vision\Scripts\python.exe shoot_logger.py    # Ctrl+C 結束，輸出 shoot_log_<時間>.csv
```

## 訓練模型

資料集、權重與訓練輸出沒有放進 repo（太大），需另外備份。流程：

1. `scripts/build_dataset.py`：`review/data/masks_r2.json` → YOLO seg 資料集（`hole_big` / `hole_small`）
2. `scripts/train.py`：從 `yolo11n-seg.pt` 訓練，輸出到 `runs/yolo11n_seg_v1/`
3. `scripts/predict_val.py`、`confusion.py`、`find_fp.py`：驗證與找誤報
4. 匯出到 Orin：在筆電匯出 ONNX，再到 Orin 上用 `trtexec` 建 engine（engine 必須在 Orin 本機建）

```bash
yolo export model=best.pt format=onnx imgsz=640 opset=17 simplify=True
# 在 Orin 上：
/usr/src/tensorrt/bin/trtexec --onnx=best.onnx --saveEngine=best.engine --fp16
```

## 環境

| 環境 | 用途 | 主要套件 |
|---|---|---|
| Orin Nano（JetPack 6.0、Python 3.10） | 視覺節點 | TensorRT 8.6、OpenCV、numpy、websocket-client、msgpack（不需 PyTorch／pyntcore） |
| 筆電 `.venv_vision` | 訓練、離線測試、記錄器 | ultralytics、opencv-python、pyntcore |
| 筆電 `.venv_nt` | `mock_robot.py` | pyntcore、pillow |

## 不在 repo 裡的檔案

以下檔案只在本機，請另外備份：`data/`、`data_robotcam/`（原始照片）、`dataset/`、`runs/`（訓練輸出與權重）、`*.pt` / `*.onnx` / `*.engine`（模型）、`document/` 內的 PDF／Word／影片、`組合圖.stp`（CAD）、`tools/*.csv`（記錄檔）。
