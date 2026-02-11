# Audio AI Web Platform

這是一個基於 Web 的音訊 AI 分析平台，專為海洋生物（如鯨豚）聲音偵測與標記而設計。系統提供音訊上傳、頻譜圖轉換、AI 自動偵測 (YOLOv8)、人工標記校正以及模型訓練功能。

## ✨ 主要功能

*   **音訊上傳與處理**：
    *   支援上傳音訊檔案 (.wav, .mp3)。
    *   自動將長音訊切割並轉換為頻譜圖 (Spectrogram)。
*   **AI 自動偵測**：
    *   整合 YOLOv8 分類模型，自動辨識頻譜圖中的聲音事件。
    *   "隨需載入" 模型機制，節省資源。
*   **標記與驗證系統**：
    *   提供 Web 介面檢視分析結果。
    *   支援人工標記與修正 AI 預測結果。
*   **模型訓練**：
    *   利用標記好的資料，直接在平台上一鍵啟動 YOLOv8 模型訓練。
    *   訓練完成後可檢視詳細的訓練報告 (Confusion Matrix, 預測結果圖等)。
*   **資料集管理**：
    *   支援打包下載 "圖片 + 標籤" 的資料集 (Zip 格式)，方便地端研究或備份。
*   **背景任務處理**：
    *   使用 Celery + Redis 處理耗時的音訊分析與模型訓練任務，確保網頁操作順暢。

## 🛠️ 技術棧

*   **Backend Framework**: Flask (Python)
*   **Database**: MySQL 8.0
*   **Task Queue**: Celery + Redis
*   **AI / ML**:
    *   Ultralytics YOLOv8 (Image Classification)
    *   Librosa (Audio Processing)
    *   PyTorch
*   **Containerization**: Docker & Docker Compose

## 🚀 快速開始

### 前置需求

請確保您的電腦已安裝：
*   [Docker](https://www.docker.com/products/docker-desktop)
*   [Docker Compose](https://docs.docker.com/compose/install/)

### 安裝與執行

1.  **Clone 專案**
    ```bash
    git clone <repository_url>
    cd audio-ai_web
    ```

2.  **啟動服務**
    使用 Docker Compose 一鍵啟動所有服務 (Web, Database, Redis, Worker)：
    ```bash
    docker-compose up --build
    ```

3.  **訪問應用程式**
    服務啟動後，請打開瀏覽器訪問：
    [http://localhost:5000](http://localhost:5000)

### 預設資料庫帳號
*   如果在 `docker-compose.yml` 中未修改，預設資料庫連線資訊如下：
    *   **Port**: 3306
    *   **User**: user
    *   **Password**: password
    *   **Database**: audio_db

## 📂 專案結構

```
.
├── app/
│   ├── static/               # 靜態資源目錄
│   │   ├── css/              # 網頁樣式表 (CSS)
│   │   ├── results/          # 分析結果儲存區 (頻譜圖、切割音訊、暫存檔)
│   │   ├── training_runs/    # 模型訓練產出 (權重檔 .pt、訓練圖表)
│   │   └── uploads/          # 原始上傳音訊備份
│   ├── templates/            # Flask HTML 模板
│   │   ├── base.html         # 基礎頁面佈局 (Navbar, Footer)
│   │   ├── index.html        # 首頁 (音訊上傳介面)
│   │   ├── history.html      # 歷史紀錄列表頁面
│   │   ├── result.html       # 單筆分析結果詳細頁面
│   │   ├── label.html        # 人工標記與修正介面
│   │   ├── training_status.html # 模型訓練任務狀態列表
│   │   └── training_report.html # 單次訓練詳細報告頁面
│   ├── __init__.py           # Flask App 初始化 (DB, Celery, Config 設定)
│   ├── ai_model.py           # AI 推論核心邏輯 (YOLOv8 模型載入與預測)
│   ├── audio_utils.py        # 音訊處理工具庫 (Librosa 頻譜圖生成、切割)
│   ├── main.py               # 主要路由 (Routes) 與視圖函式 (View Functions)
│   ├── models.py             # SQLAlchemy 資料庫模型定義 (Schema)
│   └── tasks.py              # Celery 背景任務邏輯 (非同步分析、訓練)
├── docker-compose.yml        # Docker 服務編排配置 (Web, DB, Redis, Worker)
├── Dockerfile                # Web Service 容器建置檔
└── requirements.txt          # Python 專案依賴套件列表
```

## 🤖 關於 AI 模型

系統預設會尋找 `app/models/best.pt` 作為推論模型。
1.  **初次使用**：如果您還沒有模型，推論功能將暫時無法使用，但您仍可使用標記功能。
2.  **訓練模型**：在平台上標記足夠資料後，至「訓練狀態」頁面啟動訓練。
3.  **部署新模型**：訓練成功後，請將訓練出的最佳模型檔案 (通常位於 static/training_results/...) 複製並重新命名為 `app/models/best.pt` 即可生效。
