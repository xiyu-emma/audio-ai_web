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
    audio_info = AudioInfo.query.get(audio_id)
    if not audio_info:
        return

    try:
        audio_info.status = 'PROCESSING'
        audio_info.progress = 0
        db.session.commit()

        upload_path = audio_info.file_path
        result_dir = os.path.join(current_app.root_path, 'static', audio_info.result_path)
        os.makedirs(result_dir, exist_ok=True)
        
        params = audio_info.get_params()

        def progress_callback(processed_count, total_count):
            if total_count > 0:
                progress = int((processed_count / total_count) * 100)
                if progress != audio_info.progress:
                    audio_info.progress = progress
                    db.session.commit()

        # 呼叫音訊處理工具
        results_data = process_large_audio(
            filepath=upload_path,
            result_dir=result_dir,
            spec_type=params.get('spec_type', 'mel'),
            segment_duration=float(params.get('segment_duration', 2.0)),
            overlap_ratio=float(params.get('overlap', 50)) / 100.0,
            target_sr=int(params['sample_rate']) if params.get('sample_rate', 'None').isdigit() else None,
            is_mono=(params.get('channels', 'mono') == 'mono'),
            progress_callback=progress_callback
        )

        # 計算時間參數，用於生成 CetaceanInfo
        segment_duration = float(params.get('segment_duration', 2.0))
        overlap_ratio = float(params.get('overlap', 50)) / 100.0
        
        try:
            target_sr = int(params.get('sample_rate'))
        except (ValueError, TypeError):
             target_sr = audio_info.fs if audio_info.fs else 44100

        frame_length_samples = int(segment_duration * target_sr)
        hop_length_samples = int(frame_length_samples * (1 - overlap_ratio))

        # 雙表寫入迴圈
        for i, res_item in enumerate(results_data):
            # 1. 寫入 Result (檔案路徑)
            new_result = Result(
                upload_id=audio_id,
                audio_filename=res_item['audio'],
                spectrogram_filename=res_item['display_spectrogram'],
                spectrogram_training_filename=res_item['training_spectrogram']
            )
            db.session.add(new_result)

            # 2. 寫入 CetaceanInfo (生物資訊)
            start_sample = i * hop_length_samples
            end_sample = start_sample + frame_length_samples
            
            new_cetacean = CetaceanInfo(
                audio_id=audio_id,
                start_sample=start_sample,
                end_sample=end_sample,
                event_duration=segment_duration,
                event_type=0,   # 0: 未知
                detect_type=2   # 2: 系統自動切割
            )
            db.session.add(new_cetacean)
        
        audio_info.status = 'COMPLETED'
        audio_info.progress = 100
        db.session.commit()

    except Exception as e:
        print(f"音訊處理任務 {audio_id} 失敗: {e}")
        audio_info.status = 'FAILED'
        db.session.commit()
        raise

# --- 任務 2: 模型訓練 ---

@celery.task(name='app.tasks.train_yolo_model')
def train_yolo_model(upload_ids, training_run_id, model_name='yolov8n-cls.pt'):
    """
    背景任務：使用已標記的資料來訓練 YOLOv8 分類模型。
    """
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
            epochs=50, 
            imgsz=224,
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
        
        if os.path.exists(best_model_path):
            try:
                # 重新載入模型進行驗證，以取得完整的混淆矩陣
                val_model = YOLO(best_model_path)
                metrics = val_model.val(data=dataset_dir, verbose=False)
                
                # A. Top-1 準確率
                accuracy_top1 = metrics.top1
                
                # B. 各類別指標 (Precision, Recall, F1)
                if hasattr(metrics, 'confusion_matrix'):
                    conf_matrix = metrics.confusion_matrix.matrix
                    names = metrics.names
                    class_ids = sorted(names.keys())
                    
                    for i in class_ids:
                        name = names[i]
                        tp = conf_matrix[i, i]
                        fp = conf_matrix[:, i].sum() - tp
                        fn = conf_matrix[i, :].sum() - tp
                        
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                        
                        per_class_list.append({
                            'name': name,
                            'precision': round(float(precision), 3),
                            'recall': round(float(recall), 3),
                            'f1-score': round(float(f1), 3)
                        })
            except Exception as e:
                print(f"計算詳細指標時發生錯誤: {e}")

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
                            # YOLOv8 CSV 欄位名稱
                            if 'metrics/accuracy_top1' in data_dict:
                                accuracy_top1 = data_dict['metrics/accuracy_top1']
                except: pass

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

# --- 任務 3: AI 自動標記 (完整版) ---

@celery.task(name='app.tasks.auto_label_task')
def auto_label_task(upload_id, model_path):
    """
    背景任務：對 CetaceanInfo 進行自動標記。
    會更新 status 與 progress，支援前端動態監控。
    """
    if not os.path.exists(model_path): return
    
    audio_info = AudioInfo.query.get(upload_id)
    if not audio_info: return

    try:
        # 1. 更新狀態為 PROCESSING，讓前端可以顯示進度條
        audio_info.status = 'PROCESSING'
        audio_info.progress = 0
        db.session.commit()

        model = YOLO(model_path)
        
        # 準備 Label Mapping (Class Name -> ID)
        all_labels_map = {label.name: label.id for label in Label.query.all()}
        
        # 取得資料並對齊 (Result 存圖片路徑, CetaceanInfo 存標記)
        results_list = Result.query.filter_by(upload_id=upload_id).order_by(Result.id).all()
        cetaceans_list = CetaceanInfo.query.filter_by(audio_id=upload_id).order_by(CetaceanInfo.id).all()
        
        total_items = len(results_list)
        count = 0

        # 逐一預測
        for i, (res_item, cetacean_item) in enumerate(zip(results_list, cetaceans_list)):
            
            # 定期更新進度 (每 2% 更新一次)
            if total_items > 0 and i % max(1, int(total_items * 0.02)) == 0:
                current_prog = int((i / total_items) * 100)
                if current_prog != audio_info.progress:
                    audio_info.progress = current_prog
                    db.session.commit()

            image_path = os.path.join(current_app.root_path, 'static', res_item.audio_info.result_path, res_item.spectrogram_training_filename)
            
            if not os.path.exists(image_path): continue
            
            try:
                preds = model(image_path, verbose=False)
                if preds and preds[0].probs:
                    top1_index = preds[0].probs.top1
                    label_name = model.names[top1_index]
                    
                    predicted_id = 0
                    # 嘗試將預測結果轉為 ID
                    if label_name in all_labels_map:
                        predicted_id = all_labels_map[label_name]
                    elif str(label_name).isdigit():
                        predicted_id = int(label_name)
                    
                    if predicted_id != 0:
                        cetacean_item.event_type = predicted_id
                        cetacean_item.detect_type = 1 # 標記為 AI 辨識
                        count += 1
            except Exception as e:
                print(f"預測錯誤 (Index {i}): {e}")
                continue

        # 完成後更新狀態
        audio_info.progress = 100
        audio_info.status = 'COMPLETED' # 設定回 COMPLETED，表示這是一個已完成的分析檔 (只是多了標記)
        db.session.commit()
        
        print(f"自動標記完成，更新了 {count} 筆資料。")

    except Exception as e:
        print(f"自動標記任務失敗: {e}")
        audio_info.status = 'COMPLETED' # 發生錯誤也設回完成，避免卡在 Processing
        db.session.commit()

    finally:
        # 清理暫存模型檔
        if os.path.exists(model_path):
            try:
                os.remove(model_path)
                parent = os.path.dirname(model_path)
                if not os.listdir(parent):
                    os.rmdir(parent)
            except: pass