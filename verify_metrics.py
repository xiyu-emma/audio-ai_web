
import os
import sys
import shutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

def test_metrics_calculation():
    print("=== 開始測試指標計算邏輯 ===")
    
    # 模擬資料
    class_names = ['Dolphin', 'Whale', 'Ship']
    print(f"模擬類別: {class_names}")
    
    # 模擬預測結果 (真實標籤, 預測標籤)
    # 0: Dolphin, 1: Whale, 2: Ship
    all_labels = [0, 0, 0, 1, 1, 1, 2, 2, 2, 0]
    all_preds  = [0, 0, 1, 1, 1, 2, 2, 2, 0, 0] 
    
    print(f"真實標籤: {all_labels}")
    print(f"預測標籤: {all_preds}")
    
    try:
        # 1. 計算詳細指標
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0, labels=list(range(len(class_names)))
        )
        
        per_class_list = []
        for i, name in enumerate(class_names):
            item = {
                'name': name,
                'precision': round(float(precision[i]), 3),
                'recall': round(float(recall[i]), 3),
                'f1-score': round(float(f1[i]), 3),
                'support': int(support[i])
            }
            per_class_list.append(item)
            print(f"類別 {name}: {item}")
            
        print("詳細指標計算... 成功 ✅")
        
        # 2. 生成混淆矩陣
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
        print(f"混淆矩陣:\n{cm}")
        
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        im = ax_cm.imshow(cm, interpolation='nearest', cmap='Blues')
        fig_cm.colorbar(im, ax=ax_cm)
        
        ax_cm.set_xticks(np.arange(len(class_names)))
        ax_cm.set_yticks(np.arange(len(class_names)))
        ax_cm.set_xticklabels(class_names, rotation=45, ha='right')
        ax_cm.set_yticklabels(class_names)
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax_cm.text(j, i, format(cm[i, j], 'd'),
                          ha="center", va="center",
                          color="white" if cm[i, j] > thresh else "black")
        
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_ylabel('True')
        ax_cm.set_title('Confusion Matrix (Manual)')
        
        output_file = 'test_confusion_matrix.png'
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        
        if os.path.exists(output_file):
            print(f"混淆矩陣圖表生成... 成功 ({output_file}) ✅")
            os.remove(output_file) # 清理
        else:
            print("混淆矩陣圖表生成... 失敗 ❌")
            
    except Exception as e:
        print(f"測試失敗 ❌: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        test_metrics_calculation()
    except ImportError as e:
        print(f"缺少依賴套件: {e}")
        print("請確保已安裝: numpy, matplotlib, scikit-learn")
