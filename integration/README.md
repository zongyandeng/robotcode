# 視覺整合：Orin Nano ⇄ roboRIO

## 分工

```
RealSense ─► Orin Nano (integration/orin/vision_node.py)
              YOLO11n-seg(TensorRT) → 12 個孔 → 3D 位置、亮燈小孔
                      │ NetworkTables  表名 "Vision"（Orin 是 client）
                      ▼
            roboRIO (robot_code)
              Vision.java      讀 NT，環狀緩衝＋中位數
              AutoShoot.java   轉塔／仰角／發射輪就位後才送球
              ShotTable.java   距離 → 仰角、轉速（待實測填表）
```

## 孔的編號

`index = row * 3 + col`。row 0 = 最上排 3 個小孔；row 1~3 = 大孔由上到下；col 0~2 = 由左到右。

## NetworkTables "Vision" 表

| 名稱 | 型態 | 說明 |
|---|---|---|
| `frame_id` | int | 每處理完一幀 +1（最後才寫，roboRIO 以它判斷新資料） |
| `bearing_deg` | double[12] | 水平偏角，正 = 目標在相機右側；缺的孔為 NaN |
| `pitch_deg` | double[12] | 仰角，正 = 目標在相機上方 |
| `dist_m` | double[12] | 水平距離（深度取孔周圍的板子，不取孔內） |
| `height_m` | double[12] | 目標高於相機光心的高度 |
| `lit_small` | int | 亮燈小孔欄位 0~2，認不出來為 -1 |
| `red_score` | double[3] | 三個小孔外圍的紅色像素比例（除錯用） |
| `grid_ok` | bool | 12 個孔是否都排進格子 |
| `n_big`, `n_small`, `fps` | | 偵測數量與幀率 |
| `enable` | bool | **roboRIO → Orin**。false = Orin 不拍照、不跑辨識。自主開場移動時關閉，到定點才開；沒人發布時預設 true |

## 自主開場（robot_code：`AutoDrive` + `RobotContainer.autoShootCommand`）

場地座標（自己定義）：原點 = 機器人作業區後緣中心，+X 朝九宮格，+Y 朝左，單位公尺。依規則圖，起始框在作業區後方左右兩角（中心約 X=0.6、Y=±1.65），九宮格板面在 X=7.5、Y=0。

流程：儀表板選 `Auto/Start Side`（左／右起始框）→ Start 鍵 → 關相機、里程計設到起始框 → 斜向直線開到射擊定點（`AutoConstants.kShootX/kShootY`，車頭全程朝九宮格）→ 煞車停穩 0.3 秒 → **這時才開相機**（`Vision/enable = true`）→ 等視覺上線 → 煞車固定、`AutoShoot` 射擊。整段限時 60 秒（移動時間算在內）。

桌上測試不接 roboRIO：`vision_node.py` 沒收到 `enable` 時預設為 true；要強制忽略加 `--ignore-enable`。

## Orin 上要準備

1. JetPack 版本確認後，安裝 NVIDIA 提供的 PyTorch（Jetson 版 wheel）、`ultralytics`、`pyrealsense2`（Jetson 上通常要從原始碼編譯 librealsense）、`pyntcore`。
2. 在 **Orin 本機**匯出 TensorRT engine（不能在別台電腦匯出再拷貝）：
   `yolo export model=best.pt format=engine half=True imgsz=640`
3. 執行（隊號 9427，roboRIO 的 IP 為 `10.94.27.2`）：
   `python vision_node.py --model best.engine --source realsense --team 9427 --debug-dir /tmp/dbg`

離線測試（任何電腦、不需要 roboRIO 或 RealSense）：
`python vision_node.py --model best.pt --source images --path ./imgs --no-nt --debug-dir ./dbg`

## 實機前必須校準的東西

- **`kTurretDirection`**（Constants.VisionConstants）：轉塔馬達正轉時砲口往右填 +1、往左填 -1，接反會越瞄越偏。
- **`kTurretZeroOffsetRot`**：砲口正對車頭時的轉塔讀數。
- 相機若有向上仰，用 `--camera-pitch-deg` 告訴 `vision_node.py`。
- **相機曝光要固定**（`RS_EXPOSURE`、`RS_GAIN` 環境變數）。LED 環在自動曝光下會飽和成白色，紅色資訊就消失。
- 紅燈門檻 `--red-min`（現在 0.04）：用練習場 104 張照片調的，機器人相機實拍後要重調。
- **`ShotTable`**：現在是佔位值，等同手動預設，要用實測數據填。

## Driver Station 的自主／遙控時間

在 Driver Station 的 Setup（齒輪）分頁有 Practice Mode 的時間設定，可以自訂自主與遙控的長度；這個設定只影響你們自己練習。比賽是否使用 FMS／官方 DS 目前未知，所以 `RobotContainer` 的 Start 鍵可以在 Teleop 模式下直接啟動同一套 60 秒自主流程，不依賴 DS 的 Autonomous 模式。`getAutonomousCommand()` 也回傳同一個流程。

## Orin 實際安裝方式（JetPack 6.0 / L4T 36.3，2026-11-11 實測可用）

這台 Orin 是 JetPack 6.0（CUDA 12.2、cuDNN 8.9、TensorRT 8.6.2），而且在機器人網路上**不能上網**、sudo 需要密碼。
NVIDIA 已下架 JetPack 6.0 版的 GPU PyTorch，RobotPy 也沒有 Python 3.10 + aarch64 的 pyntcore，所以改成**不需要 PyTorch 與 pyntcore** 的做法：

| 原本 | 改成 |
|---|---|
| Ultralytics + GPU PyTorch 推論 | `trt_seg.py`：TensorRT engine 直接推論，前後處理照 Ultralytics（已用 30 張照片比對：孔數、格子、亮燈全部一致，角度差 < 0.3°） |
| pyntcore | `nt4_lite.py`：純 Python NT4 客戶端（websocket-client + msgpack），topic 名稱與型別相同 |

Orin 上的檔案都在 `~/vision/`：`vision_node.py`、`trt_seg.py`、`nt4_lite.py`、`best.onnx`、`best.engine`、`best.json`（類別名稱）、`run_vision.sh`、
`trt_py/`（NVIDIA 官方 `python3-libnvinfer_8.6.2.3-1+cuda12.2_arm64.deb` 用 `dpkg-deb -x` 解開，不用 sudo）。
Python 套件用 `pip install --user --no-index` 從筆電下載好的 wheel 安裝（numpy 1.26.4、opencv-python 4.10、websocket-client、msgpack）。

- 重建 engine（換模型時）：在筆電 `yolo export model=best.pt format=onnx imgsz=640 opset=17 simplify=True`，把 `best.onnx` 複製到 Orin，
  `/usr/src/tensorrt/bin/trtexec --onnx=best.onnx --saveEngine=best.engine --fp16`（實測 GPU 約 5.6 ms／幀）。
- 啟動：`~/vision/run_vision.sh`（桌上測試加 `--ignore-enable`；相機倒裝加 `--rotate180`）。儀表板相機網址 `http://10.94.27.12:1181/stream.mjpg`。
- 一般 USB 相機沒有深度：`dist_m` 為 NaN，`AutoShoot` 改用里程計推算距離。
- **開機自動啟動**（已設定）：`~/.config/systemd/user/vision.service`（範本在 `orin/vision.service`）＋ `loginctl enable-linger william`，
  不需要 sudo。斷電重開後約 20 秒內 roboRIO 就收到視覺資料（已實測）。出錯或相機拔掉 2 秒後自動重啟。
  狀態 `systemctl --user status vision`、log `tail -f ~/vision/vision.log`、改程式後 `systemctl --user restart vision`；
  要手動跑 `run_vision.sh` 前先 `systemctl --user stop vision`，否則兩個程式會搶相機。
- **相機用固定路徑**：`run_vision.sh` 的 `--cam-index` 用 `/dev/v4l/by-id/usb-Sonix_..._Adesso_CyberTrack_K1_SN0001-video-index0`，
  重插 USB 後 `/dev/videoN` 編號變了也找得到（2026-11-17 相機掉線後改成 `/dev/video1`，原本寫死的 0 就打不開）。換相機要改這條路徑（`ls /dev/v4l/by-id/` 查）。
- **看門狗**（`--watchdog 秒數`，預設 10，0 = 關閉）：連續超過這麼多秒沒有新畫面（相機卡住、USB 斷線、辨識卡死），
  程式自己結束（exit 3），由 systemd 在 2 秒後重啟；log 會寫 `[watchdog] … 秒沒有新畫面`。啟動時第一幀前給 30 秒（載入引擎）。
  `enable=false` 暫停辨識時畫面照常進來，不會被誤判。**限制**：相機若一直回傳「同一張」凍結畫面，看門狗抓不到。
- **曝光**：`vision_node.py --exposure soft`（預設）由程式依「板子區域」亮度自動調整（`--ae-target` 目標亮度、`--ae-max-exp` 上限），
  也可用 `auto`（相機內建）或固定數字（比賽要鎖死時）。啟動時也會把白平衡設回自動。
- **舊的 AprilTag 系統已停用**（2026-11-11）：`frc9427-vision`、`frc9427-watchdog` 會開 `/dev/video0` 並把曝光鎖在 13、白平衡鎖在 3500K，
  跟這套視覺搶相機，所以用 `sudo systemctl disable --now frc9427-vision frc9427-watchdog` 停掉（檔案都在 `~/workspaces/`，`enable --now` 可恢復）。
  port 8080 的 `frc9427-fileserver` 還在跑，**不要在那個網頁按 Apply**，會把曝光改回手動值。
