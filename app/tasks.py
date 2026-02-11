import os
import shutil
import random
import json
import time
import csv
from collections import defaultdict
import numpy as np

# AI/ML 函式庫
from ultralytics import YOLO
try:
    import tensorflow as tf
    from PIL import Image
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from . import celery, db
from .models import AudioInfo, Result, TrainingRun, Label, CetaceanInfo
from .audio_utils import process_large_audio
from flask import current_app

# --- 任務 1: 音訊處理 ---

@celery.task(name='app.tasks.process_audio_task', bind=True)
def process_audio_task(self, audio_id):
    """
    背景任務：處理上傳的音訊檔案，切割成片段並產生頻譜圖。
    """
    # 重要：在 task 開始時重置資料庫連接，避免 fork 進程共享連接問題
    db.session.remove()
    db.engine.dispose()
    
    try:
        # 使用原生 SQL 查詢必要資料，避免 ORM session 問題
        result = db.session.execute(
            db.text("SELECT file_path, result_path, params, fs FROM audio_info WHERE id = :id"),
            {"id": audio_id}
        ).fetchone()
        
        if not result:
            return
        
        upload_path = result[0]
        result_path = result[1]
        params_json = result[2]
        audio_fs = result[3]
        
        # 更新狀態為處理中
        db.session.execute(
            db.text("UPDATE audio_info SET status = 'PROCESSING', progress = 0 WHERE id = :id"),
            {"id": audio_id}
        )
        db.session.commit()

        result_dir = os.path.join(current_app.root_path, 'static', result_path)
        os.makedirs(result_dir, exist_ok=True)
        
        params = json.loads(params_json) if params_json else {}
        
        # 追蹤上次更新的進度，減少資料庫寫入頻率
        last_updated_progress = [0]

        def progress_callback(processed_count, total_count):
            """更新進度 - 使用原生 SQL 避免 session 衝突"""
            if total_count > 0:
                progress = int((processed_count / total_count) * 100)
                # 只在進度變化超過 10% 時才更新資料庫
                if progress - last_updated_progress[0] >= 10 or progress == 100:
                    try:
                        db.session.execute(
                            db.text("UPDATE audio_info SET progress = :progress WHERE id = :id"),
                            {"progress": progress, "id": audio_id}
                        )
                        db.session.commit()
                        last_updated_progress[0] = progress
                    except Exception as e:
                        db.session.rollback()
                        print(f"進度更新失敗 (ID: {audio_id}): {e}")

        # 組裝頻譜圖參數字典
        n_fft = int(params.get('n_fft', 1024))
        window_overlap = float(params.get('window_overlap', 50)) / 100.0
        hop_length = int(n_fft * (1 - window_overlap))
        if hop_length < 1:
            hop_length = 1
        
        spec_params = {
            'n_fft': n_fft,
            'hop_length': hop_length,
            'window_type': params.get('window_type', 'hann'),
            'n_mels': int(params.get('n_mels', 128)),
            'f_min': float(params.get('f_min', 0)),
            'f_max': float(params.get('f_max', 0)),
            'power': float(params.get('power', 2.0))
        }
        
        results_data = process_large_audio(
            filepath=upload_path,
            result_dir=result_dir,
            spec_type=params.get('spec_type', 'mel'),
            segment_duration=float(params.get('segment_duration', 2.0)),
            overlap_ratio=float(params.get('overlap', 50)) / 100.0,
            target_sr=int(params['sample_rate']) if params.get('sample_rate', 'None').isdigit() else None,
            is_mono=(params.get('channels', 'mono') == 'mono'),
            progress_callback=progress_callback,
            spec_params=spec_params
        )

        # 計算時間參數
        segment_duration = float(params.get('segment_duration', 2.0))
        overlap_ratio = float(params.get('overlap', 50)) / 100.0
        
        try:
            target_sr = int(params.get('sample_rate'))
        except (ValueError, TypeError):
            target_sr = audio_fs if audio_fs else 44100

        frame_length_samples = int(segment_duration * target_sr)
        hop_length_samples = int(frame_length_samples * (1 - overlap_ratio))

        # 雙表寫入迴圈
        for i, res_item in enumerate(results_data):
            new_result = Result(
                upload_id=audio_id,
                audio_filename=res_item['audio'],
                spectrogram_filename=res_item['display_spectrogram'],
                spectrogram_training_filename=res_item['training_spectrogram']
            )
            db.session.add(new_result)

            start_sample = i * hop_length_samples
            end_sample = start_sample + frame_length_samples
            
            new_cetacean = CetaceanInfo(
                audio_id=audio_id,
                start_sample=start_sample,
                end_sample=end_sample,
                event_duration=segment_duration,
                event_type=0,
                detect_type=2
            )
            db.session.add(new_cetacean)
        
        # 最終更新狀態
        db.session.execute(
            db.text("UPDATE audio_info SET status = 'COMPLETED', progress = 100 WHERE id = :id"),
            {"id": audio_id}
        )
        db.session.commit()

    except Exception as e:
        print(f"音訊處理任務 {audio_id} 失敗: {e}")
        db.session.rollback()
        try:
            db.session.execute(
                db.text("UPDATE audio_info SET status = 'FAILED' WHERE id = :id"),
                {"id": audio_id}
            )
            db.session.commit()
        except:
            db.session.rollback()
        raise
    finally:
        # 任務結束時清理 session
        db.session.remove()

# --- 任務 2: 模型訓練 ---

@celery.task(name='app.tasks.train_yolo_model')
def train_yolo_model(upload_ids, training_run_id, model_name='yolov8n-cls.pt', train_params=None):
    """
    背景任務：使用已標記的資料來訓練 YOLOv8 分類模型。
    支援自訂訓練參數 (epochs, batch_size, learning_rate, image_size)
    """
    # 解析訓練參數
    if train_params is None:
        train_params = {}
    epochs = train_params.get('epochs', 50)
    batch_size = train_params.get('batch_size', 16)
    learning_rate = train_params.get('learning_rate', 0.001)
    image_size = train_params.get('image_size', 224)
    training_run = TrainingRun.query.get(training_run_id)
    if not training_run: return

    try:
        training_run.status = 'RUNNING'
        training_run.progress = 5
        db.session.commit()

        # 1. 準備路徑與資料
        base_dir = os.path.join(current_app.root_path, 'static', 'training_runs', str(training_run_id))
        dataset_dir = os.path.join(base_dir, 'dataset')
        
        # 找出有標記的 Result (透過 CetaceanInfo 的 event_type 映射，或者假設 Result 也有 label 關聯)
        # 為了訓練，我們需要將 CetaceanInfo 的標記對應到 Result 的圖片
        # 這裡採用更嚴謹的配對方式
        
        # 先撈取所有相關的 CetaceanInfo (有標記的)
        labeled_cetaceans = CetaceanInfo.query.filter(
            CetaceanInfo.audio_id.in_(upload_ids),
            CetaceanInfo.event_type != 0
        ).order_by(CetaceanInfo.audio_id, CetaceanInfo.id).all()

        if not labeled_cetaceans:
            raise ValueError("找不到任何已標記的資料來進行訓練。")

        # 建立對照表以加速查找 Result
        # Map: (audio_id, index) -> Result
        # 注意：這依賴於 Result 與 CetaceanInfo 的 ID 順序一致性
        results_map = defaultdict(list)
        all_results = Result.query.filter(Result.upload_id.in_(upload_ids)).order_by(Result.upload_id, Result.id).all()
        for res in all_results:
            results_map[res.upload_id].append(res)
        
        # 為了計算 Index，我們也需要該 audio 所有的 cetacean
        all_cetaceans_map = defaultdict(list)
        all_cetaceans = CetaceanInfo.query.filter(CetaceanInfo.audio_id.in_(upload_ids)).order_by(CetaceanInfo.audio_id, CetaceanInfo.id).all()
        for c in all_cetaceans:
            all_cetaceans_map[c.audio_id].append(c)

        # 準備 Label Mapping (ID -> Name)
        label_map = {l.id: l.name for l in Label.query.all()}

        data_by_label = defaultdict(list)
        
        for cetacean in labeled_cetaceans:
            aid = cetacean.audio_id
            try:
                # 找出此 cetacean 在其 audio 序列中的 index
                idx = all_cetaceans_map[aid].index(cetacean)
                
                if idx < len(results_map[aid]):
                    result_item = results_map[aid][idx]
                    
                    # 決定標籤名稱 (資料夾名稱)
                    # 優先使用 Label 表的名稱，若無則使用 ID
                    label_name = label_map.get(cetacean.event_type, str(cetacean.event_type))
                    
                    data_by_label[label_name].append(result_item)
            except (ValueError, IndexError):
                continue

        total_val_images = 0
        
        # 建立資料集 (複製圖檔)
        for label_name, items in data_by_label.items():
            random.shuffle(items)
            if len(items) < 2:
                train_items, val_items = items, []
            else:
                split_index = int(len(items) * 0.8)
                train_items, val_items = items[:split_index], items[split_index:]
            
            total_val_images += len(val_items)

            for item_list, target_dir in [(train_items, os.path.join(dataset_dir, 'train')), (val_items, os.path.join(dataset_dir, 'val'))]:
                label_folder = os.path.join(target_dir, label_name)
                os.makedirs(label_folder, exist_ok=True)
                for item in item_list:
                    source_path = os.path.join(current_app.root_path, 'static', item.audio_info.result_path, item.spectrogram_training_filename)
                    if os.path.exists(source_path):
                        shutil.copy(source_path, label_folder)
        
        # 容錯：若無驗證集，複製訓練集充當
        if total_val_images == 0:
            src_train = os.path.join(dataset_dir, 'train')
            dst_val = os.path.join(dataset_dir, 'val')
            if os.path.exists(src_train):
                if os.path.exists(dst_val): shutil.rmtree(dst_val)
                shutil.copytree(src_train, dst_val)
        
        training_run.progress = 15
        db.session.commit()
        
        # 2. 開始訓練
        model = YOLO(model_name)
        
        def on_epoch_end_callback(trainer):
            current_epoch = trainer.epoch + 1
            total_epochs = trainer.epochs
            progress = 15 + int((current_epoch / total_epochs) * 80)
            if progress % 5 == 0 and progress != training_run.progress:
                training_run.progress = progress
                db.session.commit()

        model.add_callback("on_epoch_end", on_epoch_end_callback)
        
        # 執行訓練
        model.train(
            data=dataset_dir, 
            epochs=epochs, 
            batch=batch_size,
            lr0=learning_rate,
            imgsz=image_size,
            project=base_dir, 
            name='train_results',
            val=True
        )
        
        training_run.progress = 98
        db.session.commit()
        
        # --- 3. 取得詳細指標 ---
        accuracy_top1 = 0.0
        per_class_list = []
        
        train_results_dir = os.path.join(base_dir, 'train_results')
        best_model_path = os.path.join(train_results_dir, 'weights', 'best.pt')
        
        # 取得類別名稱 (從資料集目錄)
        train_dir = os.path.join(dataset_dir, 'train')
        val_dir = os.path.join(dataset_dir, 'val')
        class_names = []
        if os.path.exists(train_dir):
            class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        
        print(f"[YOLO 訓練] 偵測到 {len(class_names)} 個類別: {class_names}")
        
        if os.path.exists(best_model_path):
            try:
                # 載入最佳模型
                val_model = YOLO(best_model_path)
                
                # 使用 val() 取得 Top-1 準確率
                metrics = val_model.val(data=dataset_dir, verbose=False)
                if hasattr(metrics, 'top1'):
                    accuracy_top1 = metrics.top1
                    print(f"[YOLO 訓練] Top-1 準確率: {accuracy_top1}")
                
                # 手動計算詳細指標：對驗證集進行預測
                from sklearn.metrics import precision_recall_fscore_support, confusion_matrix as sklearn_cm
                from PIL import Image as PILImage
                import matplotlib.pyplot as plt
                
                all_preds = []
                all_labels = []
                
                # 遍歷驗證集
                if os.path.exists(val_dir):
                    for class_idx, class_name in enumerate(class_names):
                        class_dir = os.path.join(val_dir, class_name)
                        if os.path.exists(class_dir):
                            for img_file in os.listdir(class_dir):
                                img_path = os.path.join(class_dir, img_file)
                                if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                                    try:
                                        # 進行預測
                                        result = val_model.predict(img_path, verbose=False)
                                        if result and len(result) > 0:
                                            pred_class = result[0].probs.top1
                                            all_preds.append(pred_class)
                                            all_labels.append(class_idx)
                                    except Exception as pred_e:
                                        print(f"[YOLO 訓練] 預測圖片失敗 {img_file}: {pred_e}")
                
                print(f"[YOLO 訓練] 完成 {len(all_preds)} 張驗證圖片預測")
                
                if all_preds and all_labels:
                    # 計算每類別指標
                    precision, recall, f1, support = precision_recall_fscore_support(
                        all_labels, all_preds, average=None, zero_division=0, labels=list(range(len(class_names)))
                    )
                    
                    for i, name in enumerate(class_names):
                        per_class_list.append({
                            'name': name,
                            'precision': round(float(precision[i]), 3),
                            'recall': round(float(recall[i]), 3),
                            'f1-score': round(float(f1[i]), 3),
                            'support': int(support[i])
                        })
                    
                    print(f"[YOLO 訓練] 成功計算 {len(per_class_list)} 個類別的詳細指標")
                    
                    # 生成混淆矩陣圖
                    cm = sklearn_cm(all_labels, all_preds, labels=list(range(len(class_names))))
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
                    plt.tight_layout()
                    plt.savefig(os.path.join(train_results_dir, 'confusion_matrix_manual.png'))
                    plt.close()
                    
            except Exception as e:
                print(f"[YOLO 訓練] 計算詳細指標時發生錯誤: {e}")
                import traceback
                traceback.print_exc()

        # 備案：如果 val() 失敗，嘗試從 CSV 讀取 Top-1
        if accuracy_top1 == 0.0:
            csv_path = os.path.join(train_results_dir, 'results.csv')
            if os.path.exists(csv_path):
                try:
                    with open(csv_path, 'r') as f:
                        lines = f.readlines()
                        if len(lines) > 1:
                            last_line = lines[-1].strip()
                            keys = [k.strip() for k in lines[0].split(',')]
                            values = [float(v.strip()) for v in last_line.split(',')]
                            data_dict = dict(zip(keys, values))
                            if 'metrics/accuracy_top1' in data_dict:
                                accuracy_top1 = data_dict['metrics/accuracy_top1']
                                print(f"[YOLO 訓練] 從 CSV 讀取 Top-1: {accuracy_top1}")
                except Exception as csv_e:
                    print(f"[YOLO 訓練] 讀取 CSV 失敗: {csv_e}")
        
        # 備案：如果沒有詳細指標，至少提供類別名稱
        if not per_class_list and class_names:
            print("[YOLO 訓練] 使用備案：僅提供類別名稱")
            per_class_list = [{'name': name, 'precision': 0, 'recall': 0, 'f1-score': 0} for name in class_names]

        # 5. 儲存
        metrics_dict = {
            'accuracy_top1': round(float(accuracy_top1), 4),
            'per_class_list': per_class_list
        }
        training_run.metrics = json.dumps(metrics_dict)
        
        # 6. 等待圖檔生成
        expected_images = ['results.png', 'confusion_matrix.png']
        for _ in range(5):
            if all(os.path.exists(os.path.join(train_results_dir, img)) for img in expected_images):
                break
            time.sleep(1)

        training_run.results_path = os.path.join('training_runs', str(training_run_id), 'train_results')
        training_run.status = 'SUCCESS'
        training_run.progress = 100
        db.session.commit()
        print(f"--- [訓練任務 #{training_run_id}] 成功完成 ---")

    except Exception as e:
        print(f"!!! [訓練任務 #{training_run_id}] 失敗: {e}")
        if 'training_run' in locals() and training_run:
            training_run.status = 'FAILURE'
            training_run.progress = 100
            db.session.commit()
        raise

# --- 任務 2b: CNN 模型訓練 (PyTorch) ---

@celery.task(name='app.tasks.train_cnn_model')
def train_cnn_model(upload_ids, training_run_id, model_name='resnet18', train_params=None):
    """
    背景任務：使用 PyTorch 訓練 CNN 分類模型 (ResNet18, EfficientNet-B0)。
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import transforms, datasets, models
    import matplotlib.pyplot as plt
    
    # 解析訓練參數
    if train_params is None:
        train_params = {}
    epochs = train_params.get('epochs', 50)
    batch_size = train_params.get('batch_size', 16)
    learning_rate = train_params.get('learning_rate', 0.001)
    image_size = train_params.get('image_size', 224)
    
    training_run = TrainingRun.query.get(training_run_id)
    if not training_run:
        return
    
    try:
        training_run.status = 'RUNNING'
        training_run.progress = 5
        db.session.commit()
        
        # 1. 準備路徑與資料
        base_dir = os.path.join(current_app.root_path, 'static', 'training_runs', str(training_run_id))
        dataset_dir = os.path.join(base_dir, 'dataset')
        train_results_dir = os.path.join(base_dir, 'train_results')
        weights_dir = os.path.join(train_results_dir, 'weights')
        os.makedirs(weights_dir, exist_ok=True)
        
        # 找出有標記的 Result (與 YOLO 任務共用邏輯)
        labeled_cetaceans = CetaceanInfo.query.filter(
            CetaceanInfo.audio_id.in_(upload_ids),
            CetaceanInfo.event_type != 0
        ).order_by(CetaceanInfo.audio_id, CetaceanInfo.id).all()
        
        if not labeled_cetaceans:
            raise ValueError("找不到任何已標記的資料來進行訓練。")
        
        # 建立對照表
        results_map = defaultdict(list)
        all_results = Result.query.filter(Result.upload_id.in_(upload_ids)).order_by(Result.upload_id, Result.id).all()
        for res in all_results:
            results_map[res.upload_id].append(res)
        
        all_cetaceans_map = defaultdict(list)
        all_cetaceans = CetaceanInfo.query.filter(CetaceanInfo.audio_id.in_(upload_ids)).order_by(CetaceanInfo.audio_id, CetaceanInfo.id).all()
        for c in all_cetaceans:
            all_cetaceans_map[c.audio_id].append(c)
        
        label_map = {l.id: l.name for l in Label.query.all()}
        data_by_label = defaultdict(list)
        
        for cetacean in labeled_cetaceans:
            aid = cetacean.audio_id
            try:
                idx = all_cetaceans_map[aid].index(cetacean)
                if idx < len(results_map[aid]):
                    result_item = results_map[aid][idx]
                    label_name = label_map.get(cetacean.event_type, str(cetacean.event_type))
                    data_by_label[label_name].append(result_item)
            except (ValueError, IndexError):
                continue
        
        # 建立資料集 (複製圖檔)
        total_val_images = 0
        for label_name, items in data_by_label.items():
            random.shuffle(items)
            if len(items) < 2:
                train_items, val_items = items, []
            else:
                split_index = int(len(items) * 0.8)
                train_items, val_items = items[:split_index], items[split_index:]
            
            total_val_images += len(val_items)
            
            for item_list, target_dir in [(train_items, os.path.join(dataset_dir, 'train')), (val_items, os.path.join(dataset_dir, 'val'))]:
                label_folder = os.path.join(target_dir, label_name)
                os.makedirs(label_folder, exist_ok=True)
                for item in item_list:
                    source_path = os.path.join(current_app.root_path, 'static', item.audio_info.result_path, item.spectrogram_training_filename)
                    if os.path.exists(source_path):
                        shutil.copy(source_path, label_folder)
        
        # 容錯：若無驗證集，複製訓練集充當
        if total_val_images == 0:
            src_train = os.path.join(dataset_dir, 'train')
            dst_val = os.path.join(dataset_dir, 'val')
            if os.path.exists(src_train):
                if os.path.exists(dst_val):
                    shutil.rmtree(dst_val)
                shutil.copytree(src_train, dst_val)
        
        training_run.progress = 15
        db.session.commit()
        
        # 2. 設定 PyTorch DataLoader
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        train_dataset = datasets.ImageFolder(os.path.join(dataset_dir, 'train'), transform=transform)
        val_dataset = datasets.ImageFolder(os.path.join(dataset_dir, 'val'), transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        num_classes = len(train_dataset.classes)
        class_names = train_dataset.classes
        
        # 3. 載入預訓練模型
        if model_name == 'resnet18':
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        else:
            raise ValueError(f"不支援的模型: {model_name}")
        
        model = model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # 4. 訓練迴圈
        train_losses = []
        val_accuracies = []
        best_acc = 0.0
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            epoch_loss = running_loss / len(train_loader)
            train_losses.append(epoch_loss)
            
            # 驗證
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_acc = correct / total if total > 0 else 0
            val_accuracies.append(val_acc)
            
            # 儲存最佳模型 (包含 metadata)
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'classes': class_names,
                    'arch': model_name
                }, os.path.join(weights_dir, 'best.pt'))
            
            # 更新進度
            progress = 15 + int((epoch + 1) / epochs * 80)
            if progress % 5 == 0 and progress != training_run.progress:
                training_run.progress = progress
                db.session.commit()
        
        # 5. 儲存最終模型 (包含 metadata)
        torch.save({
            'model_state_dict': model.state_dict(),
            'classes': class_names,
            'arch': model_name
        }, os.path.join(weights_dir, 'last.pt'))
        
        # 6. 計算詳細分類指標 (Precision, Recall, F1-Score)
        from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
        
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # 計算每類別指標
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )
        
        per_class_list = []
        for i, name in enumerate(class_names):
            per_class_list.append({
                'name': name,
                'precision': round(float(precision[i]), 3),
                'recall': round(float(recall[i]), 3),
                'f1-score': round(float(f1[i]), 3),
                'support': int(support[i])
            })
        
        # 7. 生成混淆矩陣圖 (使用純 matplotlib)
        cm = confusion_matrix(all_labels, all_preds)
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        
        # 繪製熱力圖
        im = ax_cm.imshow(cm, interpolation='nearest', cmap='Blues')
        fig_cm.colorbar(im, ax=ax_cm)
        
        # 設定標籤
        ax_cm.set_xticks(np.arange(len(class_names)))
        ax_cm.set_yticks(np.arange(len(class_names)))
        ax_cm.set_xticklabels(class_names, rotation=45, ha='right')
        ax_cm.set_yticklabels(class_names)
        
        # 在每個格子中顯示數值
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax_cm.text(j, i, format(cm[i, j], 'd'),
                          ha="center", va="center",
                          color="white" if cm[i, j] > thresh else "black")
        
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_ylabel('True')
        ax_cm.set_title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(train_results_dir, 'confusion_matrix.png'))
        plt.close()
        
        # 8. 生成訓練結果圖
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(train_losses, label='Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        
        ax2.plot(val_accuracies, label='Validation Accuracy', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(train_results_dir, 'results.png'))
        plt.close()
        
        # 9. 儲存指標
        metrics_dict = {
            'accuracy_top1': round(best_acc, 4),
            'per_class_list': per_class_list
        }
        training_run.metrics = json.dumps(metrics_dict)
        training_run.results_path = os.path.join('training_runs', str(training_run_id), 'train_results')
        training_run.status = 'SUCCESS'
        training_run.progress = 100
        db.session.commit()
        
        print(f"--- [CNN 訓練任務 #{training_run_id}] 成功完成 (最佳準確率: {best_acc:.4f}) ---")
        
    except Exception as e:
        print(f"!!! [CNN 訓練任務 #{training_run_id}] 失敗: {e}")
        if 'training_run' in locals() and training_run:
            training_run.status = 'FAILURE'
            training_run.progress = 100
            db.session.commit()
        raise

# --- 任務 3: AI 自動標記 (完整版) ---

@celery.task(name='app.tasks.auto_label_task')
def auto_label_task(upload_id, model_path, model_type='yolo', classes_str=''):
    """
    背景任務：對 CetaceanInfo 進行自動標記。
    會更新 status 與 progress，支援前端動態監控。
    支援 YOLOv8, ResNet18, EfficientNet-B0
    
    Args:
        upload_id: 音檔 ID
        model_path: 模型檔案路徑
        model_type: 模型類型 ('yolo', 'resnet18', 'efficientnet_b0')
        classes_str: 使用者指定的類別 (逗號分隔，如 "90,91")
    """
    import torch
    
    # 解析使用者指定的類別
    user_specified_classes = []
    if classes_str and classes_str.strip():
        user_specified_classes = [int(c.strip()) for c in classes_str.split(',') if c.strip().isdigit()]
        if user_specified_classes:
            print(f"使用者指定的類別映射: {user_specified_classes}")
    
    if not os.path.exists(model_path): return
    
    audio_info = AudioInfo.query.get(upload_id)
    if not audio_info: return

    try:
        # 1. 更新狀態為 PROCESSING
        audio_info.status = 'PROCESSING'
        audio_info.progress = 0
        db.session.commit()

        # ---------------------------------------------------------
        # A. 模型初始化與載入
        # ---------------------------------------------------------
        model = None
        is_yolo = (model_type == 'yolo' or model_type.startswith('yolov8'))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cnn_labels_map = [] # Index -> ID

        if is_yolo:
            try:
                model = YOLO(model_path)
            except Exception as e:
                print(f"YOLO 模型載入失敗: {e}。嘗試使用 PyTorch 載入...")
                is_yolo = False # Fallback to generic PyTorch handling check

        if not is_yolo:
            # 載入 PyTorch 模型 (ResNet / EfficientNet)
            import torch.nn as nn
            from torchvision import transforms, models
            from PIL import Image as PILImage
            
            # 默認類別對應 (若無法從其他地方取得)
            # 假設如果 Label 表為空，則無法進行正確的 Name->ID 映射，但我們至少不能讓程式崩潰
            # 如果 Label 表有資料，則依照 Name 排序
            all_labels = sorted(Label.query.all(), key=lambda x: x.name)
            cnn_labels_map = [l.id for l in all_labels]
            
            # 先載入 checkpoint 以確定 num_classes
            try:
                checkpoint = torch.load(model_path, map_location=device)
                state_dict = checkpoint
                
                # Checkpoint 格式判斷
                if isinstance(checkpoint, dict) and 'classes' in checkpoint:
                    print("偵測到含 Metadata 的模型 checkpoint")
                    class_names = checkpoint['classes']
                    state_dict = checkpoint['model_state_dict']
                    
                    # 使用儲存的類別名稱來建立 Mapping
                    name_to_id = {l.name: l.id for l in Label.query.all()}
                    cnn_labels_map = [name_to_id.get(name, 0) for name in class_names]
                    num_classes = len(class_names)
                
                else:
                    # 舊版或是純 state_dict
                    # 嘗試從權重形狀推斷 num_classes
                    detected_classes = 0
                    if 'fc.weight' in state_dict: # ResNet
                        detected_classes = state_dict['fc.weight'].shape[0]
                    elif 'classifier.1.weight' in state_dict: # EfficientNet
                        detected_classes = state_dict['classifier.1.weight'].shape[0]
                    # 處理 module. 前綴的情況
                    elif 'module.fc.weight' in state_dict:
                        detected_classes = state_dict['module.fc.weight'].shape[0]
                    elif 'module.classifier.1.weight' in state_dict:
                        detected_classes = state_dict['module.classifier.1.weight'].shape[0]
                    
                    if detected_classes > 0:
                        print(f"從權重偵測到分類數量: {detected_classes}")
                        num_classes = detected_classes
                        
                        # 優先使用使用者指定的類別
                        if user_specified_classes and len(user_specified_classes) == num_classes:
                            cnn_labels_map = user_specified_classes
                            print(f"使用使用者指定的類別映射: {list(enumerate(cnn_labels_map))}")
                        elif user_specified_classes:
                            print(f"警告: 使用者指定的類別數量 ({len(user_specified_classes)}) 與模型輸出數 ({num_classes}) 不符")
                            # 嘗試使用使用者指定的 (可能導致錯誤，但尊重使用者意圖)
                            if len(user_specified_classes) >= num_classes:
                                cnn_labels_map = user_specified_classes[:num_classes]
                            else:
                                cnn_labels_map = user_specified_classes
                                for i in range(len(cnn_labels_map), num_classes):
                                    cnn_labels_map.append(0)
                            print(f"使用調整後的類別映射: {cnn_labels_map}")
                        else:
                            # 沒有使用者指定，嘗試從資料庫推斷
                            used_event_types_query = db.session.query(CetaceanInfo.event_type).filter(
                                CetaceanInfo.event_type != 0,
                                CetaceanInfo.event_type != None
                            ).distinct().all()
                            used_event_types = [str(et[0]) for et in used_event_types_query]
                            used_event_types_sorted = sorted(used_event_types)
                            
                            print(f"資料庫中已使用的 event_type: {used_event_types_sorted}")
                            
                            if len(used_event_types_sorted) == num_classes:
                                cnn_labels_map = [int(et) for et in used_event_types_sorted]
                                print(f"成功建立類別映射 (Index -> event_type): {list(enumerate(cnn_labels_map))}")
                            else:
                                print(f"警告: 模型輸出數 ({num_classes}) 與資料庫 event_type 數 ({len(used_event_types_sorted)}) 不符。")
                                print("請在表單中手動指定「訓練類別」(如：90,91)")
                                # 備案邏輯
                                if len(used_event_types_sorted) >= num_classes:
                                    cnn_labels_map = [int(et) for et in used_event_types_sorted[:num_classes]]
                                    print(f"使用前 {num_classes} 個 event_type 作為映射: {cnn_labels_map}")
                                elif used_event_types_sorted:
                                    # 有一些 event_type，但比模型輸出少
                                    cnn_labels_map = [int(et) for et in used_event_types_sorted]
                                    for i in range(len(cnn_labels_map), num_classes):
                                        cnn_labels_map.append(0)
                                    print(f"警告: 使用部分 event_type 並補 0: {cnn_labels_map}")
                                else:
                                    # 完全沒有 event_type 資料
                                    print("警告: 無法確定類別映射，預測結果可能不正確")
                                    cnn_labels_map = list(range(num_classes))
                    else:
                        # 無法偵測，使用 DB 數量
                        num_classes = len(cnn_labels_map)
                        if num_classes == 0:
                            print("警告: 無法決定分類數量，預設為 2 以避免崩潰")
                            num_classes = 2
                            cnn_labels_map = [90, 91]

                # 初始化架構
                if model_type == 'resnet18':
                    model = models.resnet18(weights=None)
                    num_ftrs = model.fc.in_features
                    model.fc = nn.Linear(num_ftrs, num_classes)
                elif model_type == 'efficientnet_b0':
                    model = models.efficientnet_b0(weights=None)
                    num_ftrs = model.classifier[1].in_features
                    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
                else:
                    # 預設 ResNet18
                    model = models.resnet18(weights=None)
                    num_ftrs = model.fc.in_features
                    model.fc = nn.Linear(num_ftrs, num_classes)
                
                # 載入權重
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('module.'):
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v
                        
                model.load_state_dict(new_state_dict)
                model.to(device)
                model.eval()
                
                # 定義預處理
                val_transforms = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                
            except Exception as e:
                print(f"PyTorch 模型載入失敗: {e}")
                raise e

        # ---------------------------------------------------------
        # B. 執行推論
        # ---------------------------------------------------------
        
        # 準備 Label Mapping (Class Name -> ID) for YOLO
        # YOLO 輸出 class name, PyTorch 輸出 index
        all_labels_obj_map = {label.name: label.id for label in Label.query.all()} # Name -> ID 

        results_list = Result.query.filter_by(upload_id=upload_id).order_by(Result.id).all()
        cetaceans_list = CetaceanInfo.query.filter_by(audio_id=upload_id).order_by(CetaceanInfo.id).all()
        
        total_items = len(results_list)
        count = 0

        # 逐一預測
        for i, (res_item, cetacean_item) in enumerate(zip(results_list, cetaceans_list)):
            
            # 定期更新進度
            if total_items > 0 and i % max(1, int(total_items * 0.05)) == 0:
                current_prog = int((i / total_items) * 100)
                if current_prog != audio_info.progress:
                    audio_info.progress = current_prog
                    db.session.commit()

            image_path = os.path.join(current_app.root_path, 'static', res_item.audio_info.result_path, res_item.spectrogram_training_filename)
            
            if not os.path.exists(image_path): continue
            
            predicted_id = 0
            
            try:
                if is_yolo:
                    # YOLO 推論
                    preds = model(image_path, verbose=False)
                    if preds and preds[0].probs:
                        top1_index = preds[0].probs.top1
                        label_name = model.names[top1_index]
                        
                        if label_name in all_labels_obj_map:
                            predicted_id = all_labels_obj_map[label_name]
                        elif str(label_name).isdigit():
                            predicted_id = int(label_name)
                else:
                    # CNN 推論
                    img = PILImage.open(image_path).convert('RGB')
                    img_t = val_transforms(img).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        outputs = model(img_t)
                        _, predicted = torch.max(outputs, 1)
                        pred_idx = predicted.item()
                        
                        if pred_idx < len(cnn_labels_map):
                            predicted_id = cnn_labels_map[pred_idx]
            
                # 寫入標記
                if predicted_id != 0:
                    cetacean_item.event_type = predicted_id
                    cetacean_item.detect_type = 1 # 標記為 AI 辨識
                    count += 1
                    
            except Exception as e:
                print(f"預測錯誤 (Index {i}): {e}")
                continue

        # 完成後更新狀態
        audio_info.progress = 100
        audio_info.status = 'COMPLETED'
        db.session.commit()
        
        print(f"自動標記完成，更新了 {count} 筆資料。")

    except Exception as e:
        print(f"自動標記任務失敗: {e}")
        audio_info.status = 'COMPLETED'
        db.session.commit()
        import traceback
        traceback.print_exc()

    finally:
        # 清理暫存模型檔
        if os.path.exists(model_path):
            try:
                os.remove(model_path)
                parent = os.path.dirname(model_path)
                if not os.listdir(parent):
                    os.rmdir(parent)
            except: pass


# --- 任務 3b: AI 自動標記 V2 (使用已知類別資訊) ---

@celery.task(name='app.tasks.auto_label_task_v2')
def auto_label_task_v2(upload_id, model_path, model_type='yolov8n-cls', classes_list=None):
    """
    背景任務：對 CetaceanInfo 進行自動標記 (V2)。
    使用從 TrainingRun 取得的已知類別資訊，無需猜測映射。
    
    Args:
        upload_id: 音檔 ID
        model_path: 模型檔案路徑 (來自訓練結果)
        model_type: 模型類型 ('yolov8n-cls', 'resnet18', 'efficientnet_b0')
        classes_list: 類別名稱列表，順序對應模型輸出 index (如 ['90', '91'])
    """
    import torch
    
    if not os.path.exists(model_path):
        print(f"模型檔案不存在: {model_path}")
        return
    
    audio_info = AudioInfo.query.get(upload_id)
    if not audio_info:
        print(f"找不到音檔記錄: {upload_id}")
        return

    try:
        # 更新狀態
        audio_info.status = 'PROCESSING'
        audio_info.progress = 0
        db.session.commit()
        
        # 建立類別映射 (class_name -> event_type ID)
        # classes_list 中的名稱通常就是 event_type 值
        classes_list = classes_list or []
        cnn_labels_map = []
        for name in classes_list:
            try:
                cnn_labels_map.append(int(name))
            except ValueError:
                cnn_labels_map.append(0)
        
        print(f"類別映射 (Index -> event_type): {list(enumerate(cnn_labels_map))}")
        
        # 判斷模型類型
        is_yolo = model_type.startswith('yolov8')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = None
        val_transforms = None
        
        if is_yolo:
            model = YOLO(model_path)
        else:
            # CNN 模型
            import torch.nn as nn
            from torchvision import transforms, models
            from PIL import Image as PILImage
            
            num_classes = len(cnn_labels_map) if cnn_labels_map else 2
            
            checkpoint = torch.load(model_path, map_location=device)
            state_dict = checkpoint
            
            # 處理新版 checkpoint
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            
            # 初始化模型
            if model_type == 'resnet18':
                model = models.resnet18(weights=None)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif model_type == 'efficientnet_b0':
                model = models.efficientnet_b0(weights=None)
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            else:
                model = models.resnet18(weights=None)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            
            # 載入權重
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[k[7:] if k.startswith('module.') else k] = v
            model.load_state_dict(new_state_dict)
            model.to(device)
            model.eval()
            
            val_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        # 執行推論
        results_list = Result.query.filter_by(upload_id=upload_id).order_by(Result.id).all()
        cetaceans_list = CetaceanInfo.query.filter_by(audio_id=upload_id).order_by(CetaceanInfo.id).all()
        
        total_items = len(results_list)
        count = 0

        for i, (res_item, cetacean_item) in enumerate(zip(results_list, cetaceans_list)):
            if total_items > 0 and i % max(1, int(total_items * 0.05)) == 0:
                audio_info.progress = int((i / total_items) * 100)
                db.session.commit()

            image_path = os.path.join(
                current_app.root_path, 'static', 
                res_item.audio_info.result_path, 
                res_item.spectrogram_training_filename
            )
            
            if not os.path.exists(image_path):
                continue
            
            predicted_id = 0
            
            try:
                if is_yolo:
                    preds = model(image_path, verbose=False)
                    if preds and preds[0].probs:
                        top1_index = preds[0].probs.top1
                        label_name = model.names[top1_index]
                        try:
                            predicted_id = int(label_name)
                        except ValueError:
                            predicted_id = 0
                else:
                    from PIL import Image as PILImage
                    img = PILImage.open(image_path).convert('RGB')
                    img_t = val_transforms(img).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        outputs = model(img_t)
                        _, predicted = torch.max(outputs, 1)
                        pred_idx = predicted.item()
                        
                        if pred_idx < len(cnn_labels_map):
                            predicted_id = cnn_labels_map[pred_idx]
                
                if predicted_id != 0:
                    cetacean_item.event_type = predicted_id
                    cetacean_item.detect_type = 1
                    count += 1
                    
            except Exception as e:
                print(f"預測錯誤 (Index {i}): {e}")
                continue

        audio_info.progress = 100
        audio_info.status = 'COMPLETED'
        db.session.commit()
        
        print(f"自動標記 V2 完成，更新了 {count} 筆資料。")

    except Exception as e:
        print(f"自動標記任務 V2 失敗: {e}")
        audio_info.status = 'COMPLETED'
        db.session.commit()
        import traceback
        traceback.print_exc()