## 1. 外部服務串接 API

### 上傳音訊檔案並觸發分析

- **端點 (Endpoint)**: `/api/v1/upload`
- **請求方法 (Method)**: `POST`
- **驗證 (Auth)**: 需要 Header `Authorization: Bearer <API_TOKEN>`
    - 預設 Token: `sk_test_1234567890abcdef` (可在環境變數 `YANG_SHENG_API_TOKEN` 中修改)
- **Content-Type**: `multipart/form-data`
- **請求參數 (Body)**:
    - `file` (File, 必填): 音訊檔案 (.wav 或 .mp3)
    - `point_id` (Integer, 必填): 錄音點/測站的 ID，必須是資料庫中 `PointInfo` 已存在的 ID。
    - `spec_type` (String, 選填): 頻譜圖類型 (預設: `yamnet_log_mel`)。可選：`yamnet_log_mel`, `stft`, `classic_demon`, `envelope_spectrum`。
    - `segment_duration` (Float, 選填): 音訊切割長度 (秒) (預設: `2.0`)。
    - `overlap` (Float, 選填): 切割時的重疊率 (%) (預設: `50.0`)。
    - `sample_rate` (String, 選填): 強制轉換的取樣率 (預設: `None`，保留原始)。可填：`4000`, `16000`, `32000`, `64000` 或 `None`。
    - `channels` (String, 選填): 聲道 (預設: `mono`)。可填：`mono` 或是 `stereo`。
    - `n_fft` (Integer, 選填): FFT Window Size，適用於 STFT (預設: `1024`)。
    - `window_overlap` (Float, 選填): 窗函數重疊率 (%)，適用於 STFT 等 (預設: `50.0`)。
    - `window_type` (String, 選填): 窗函數類型。適用於 STFT (預設: `hann`)。其他可選：`hamming`, `blackman`, `bartlett`, `kaiser`, `rectangular`。
    - `n_mels` (Integer, 選填): Mel 頻帶數量 (預設: `128`)。
    - `f_min` (Float, 選填): 頻譜圖最低頻率 (Hz) (預設: `0.0`)。
    - `f_max` (Float, 選填): 頻譜圖最高頻率 (Hz) (預設: `0.0`，表示使用 Nyquist 頻率)。
    - `power` (Float, 選填): 功率指數，適用於 STFT，`1.0` 為幅度，`2.0` 為功率 (預設: `2.0`)。
- **成功回應 (201 Created)**:
```json
{ "success": true, "upload_id": 123 }
```
- **錯誤回應**:
    - `401 Unauthorized`: 缺少或格式錯誤的 Authorization Header。
    - `403 Forbidden`: API Token 無效。
    - `400 Bad Request`: 未提供 `file` 檔案或未提供 `point_id`。
    - `404 Not Found`: 提供的 `point_id` 不存在。

---

## 2. 狀態查詢 API

查詢背景任務的進度。

### 查詢音訊分析進度

- **端點 (Endpoint)**: `/api/upload/<upload_id>/status`
- **請求方法 (Method)**: `GET`
- **URL 參數**:
    - `upload_id` (Integer): 音訊上傳紀錄的 ID。
- **成功回應 (200 OK)**:
```json
{ 
  "id": 123, 
  "status": "PROCESSING", // 可能的值: PENDING, PROCESSING, COMPLETED, ERROR 
  "progress": 45 // 進度百分比 0-100 
}
```

### 查詢模型訓練進度

- **端點 (Endpoint)**: `/api/training/<run_id>/status`
- **請求方法 (Method)**: `GET`
- **URL 參數**:
    - `run_id` (Integer): 訓練任務的 ID。
- **成功回應 (200 OK)**:
```json
{ 
  "id": 456, 
  "status": "TRAINING", // 可能的值: PENDING, TRAINING, SUCCESS, FAILED 
  "progress": 80 // 進度百分比 0-100 
}
```
