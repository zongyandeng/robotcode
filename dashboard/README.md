# 機器人儀表板

單一檔案 `dashboard.html`，用 Chrome 或 Edge 直接開啟（雙擊即可，不需要網路）。瀏覽器經 WebSocket 直接連 roboRIO 的 NetworkTables（`ws://roboRIO:5810`），讀取：

- `/SmartDashboard/…`：Shooter、Hood、Feeder、Orbit、AutoShoot、Vision（roboRIO 整理後）、Robot（模式、啟用、電池、比賽時間）
- `/Vision/…`：Orin 發布的孔位置、亮燈、幀率

## 比賽畫面（Driver Station 在下、儀表板在上）

雙擊 `開啟儀表板.bat`（執行 `start_dashboard.ps1`）：

- Driver Station 沒開會自動開、最小化會還原，並移到螢幕最下方。DS 是系統管理員權限，要移動它時會跳一次 UAC，按「是」即可（DS 已經在最下方就不會跳）。
- 儀表板用 Edge 的 App 模式開（沒有網址列），填滿 DS 上方的空間。用獨立的 Edge 設定檔（`%LOCALAPPDATA%\FRC9427Dashboard`），roboRIO 位址與相機網址記在那裡，不影響平常用的 Edge。
- 儀表板已經開著時不會再開第二個，只會重新擺好位置。

## 畫面

- 上方：連線狀態、機器人模式與啟用、電池電壓、比賽時間；有問題時下面會跳警示（NT 斷線、視覺斷線、電壓低、格子排列失敗、看不出亮燈、Orbit 電流高）。
- 左：相機畫面（預留，貼上 Orin 的串流網址後連線）、發射條件（發射輪／仰角／視覺是否到位）、板子 12 個孔的位置與亮燈、射擊目標。
- 右：自主射擊進度（已射幾顆）、發射輪目標與實際轉速、仰角儀表（轉塔機構已拆除）、送球機構。

右上角可改 roboRIO 位址（預設 `10.94.27.2`，由隊號 9427 推得），並切換深／淺色。位址與相機網址會記在瀏覽器裡。

## 射擊策略（上／下半場）

場地卡片上方有「射擊策略」三顆按鈕，開賽前選好，按 Start 時機器人才讀（`Auto/Strategy` chooser，高亮的是機器人實際生效的）：

| 策略 | 球數 | 做法 |
|---|---|---|
| 上半場 27 顆 | 27 | 每個大孔 3 顆，由下往上，放棄紅燈格（原本的行為，預設） |
| 下半場 連線優先 | 18 | 9 孔先各 1 顆（一次湊齊斜 8／直 4／橫 2），再繞第二圈補到每孔 2 顆；放棄紅燈格 |
| 下半場 紅燈＋連線 | 18 | 先打亮燈小孔 2 顆（各 4 分），再連線優先；需要小孔排 `ShotTable` 量好才有意義 |

策略定義在 robot_code 的 `AutoShoot.Strategy`，要加新策略在那裡加一項，儀表板按鈕的 `data-strat` 要和它的 label 一致。

## 相機畫面

Orin 上的 `vision_node.py` 加 `--mjpeg-port 1181` 就會提供疊了偵測結果的 MJPEG 串流；儀表板的相機欄位填
`http://<Orin 的 IP>:1181/stream.mjpg` 後按「連線」。連不上時每 3 秒自動重試。

## 沒有機器人時測試

```
cd dashboard
..\.venv_nt\Scripts\python.exe mock_robot.py
```

`mock_robot.py` 在本機開 NetworkTables 伺服器與 MJPEG 串流，發布同名的假數值與 `data/` 裡的照片。把儀表板的 roboRIO 位址改成 `127.0.0.1`、相機網址填 `http://127.0.0.1:1181/stream.mjpg`。
（若用瀏覽器直接開 `file://`，有些環境不允許連本機，可用 `python -m http.server 8077` 從 `http://127.0.0.1:8077/dashboard.html` 開。）

## 還沒做的

- 各馬達的溫度、電流（目前只有 Orbit 電流）。要加的話在各子系統的 `periodic()` 多發布幾個數值即可。
- 底盤速度與姿態（`DriveState` 表是結構型別，需要另外解碼）。
