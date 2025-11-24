# app/ai_model.py
"""
此模組負責載入 AI 模型並執行推論 (Inference)。

功能說明：
1.  模型載入：採用「隨需載入」機制。只有在第一次呼叫 `run_inference` 時，
    才會嘗試從指定的路徑載入模型。這避免了在應用程式啟動時因模型問題導致的失敗。
2.  推論函式 (`run_inference`)：接收一張圖片的路徑，使用已載入的 YOLOv8 分類模型
    進行預測，並回傳最有可能的類別及其信心度。

部署說明：
- 當您透過本系統的訓練功能產生了一個滿意的模型後（通常是 `best.pt` 檔案），
  請將該檔案手動複製到 `app/models/` 資料夾下並命名為 `best.pt`。
- 如此一來，當系統在處理新的音訊時，`run_inference` 就能載入此模型，
  並為新產生的頻譜圖提供預測標籤建議。
"""

from ultralytics import YOLO
import torch
import os

# --- 全域設定 ---

# 偵測是否有可用的 GPU，否則使用 CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"AI Model will use device: {DEVICE}")

# 建立模型檔案的預期路徑
# 注意：您需要手動將訓練好的 `best.pt` 檔案放在這個路徑下
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best.pt")

# 全域模型變數，初始為 None
model = None

# --- 推論函式 ---

def run_inference(image_path):
    """
    對指定的圖片執行 YOLOv8 分類推論。
    
    Args:
        image_path (str): 輸入圖片的檔案路徑。
        
    Returns:
        list: 包含預測結果的列表。如果預測成功，列表將包含一個字典，
              格式為 {'label': '預測標籤', 'confidence': 0.99}。
              如果模型不存在或推論失敗，則回傳空列表 []。
    """
    global model

    # 隨需載入模型：如果模型尚未載入，且模型檔案存在，則進行載入
    if model is None and os.path.exists(MODEL_PATH):
        try:
            print(f"Attempting to load model from: {MODEL_PATH}")
            model = YOLO(MODEL_PATH)
            model.to(DEVICE)
            print("AI model loaded successfully on demand.")
        except Exception as e:
            print(f"Error loading AI model on demand: {e}")
            return [] # 載入失敗則返回空結果

    # 如果模型不存在，則跳過推論
    if model is None:
        # 註解掉此行 print 以避免在日誌中產生過多不必要的訊息
        # print("AI model is not available, skipping inference.")
        return []

    try:
        # 使用模型進行預測
        results = model(image_path, verbose=False) # verbose=False 避免在 console 印出過多資訊
        
        # 處理分類模型的預測結果
        if results:
            result = results[0] # 取得第一張圖片的結果
            if result.probs is not None:
                top1_index = result.probs.top1
                top1_confidence = result.probs.top1conf.item()
                label = model.names[top1_index]
                
                # 回傳格式化的結果
                return [{
                    "label": label,
                    "confidence": round(top1_confidence, 4)
                }]
        return []
    except Exception as e:
        print(f"Error during AI inference on '{image_path}': {e}")
        return []

# 應用程式啟動時，可以選擇性地檢查模型檔案是否存在
if not os.path.exists(MODEL_PATH):
    print(f"Warning: Model file not found at '{MODEL_PATH}'. Inference feature will be disabled.")

