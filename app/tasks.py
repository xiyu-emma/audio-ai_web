import os
import shutil
import random
import json
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


# 從 __init__.py 引入 celery 和 db 實例
from . import celery, db
from .models import Upload, Result, TrainingRun, Label
from .audio_utils import process_large_audio
from flask import current_app

# --- 任務 1: 音訊處理 ---

@celery.task(name='app.tasks.process_audio_task', bind=True)
def process_audio_task(self, upload_id):
    """
    背景任務：處理上傳的音訊檔案，切割成片段並產生頻譜圖。
    """
    upload = Upload.query.get(upload_id)
    if not upload:
        print(f"找不到 Upload 紀錄，ID: {upload_id}")
        return

    try:
        upload.status = 'PROCESSING'
        upload.progress = 0
        db.session.commit()

        upload_path = os.path.join(current_app.root_path, current_app.config['UPLOAD_FOLDER'], f"{upload.id}_{upload.original_filename}")
        result_dir = os.path.join(current_app.root_path, 'static', upload.result_path)
        params = upload.get_params()

        def progress_callback(processed_count, total_count):
            if total_count > 0:
                progress = int((processed_count / total_count) * 100)
                if progress != upload.progress:
                    upload.progress = progress
                    db.session.commit()

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

        for res_item in results_data:
            new_result = Result(
                upload_id=upload_id,
                audio_filename=res_item['audio'],
                spectrogram_filename=res_item['display_spectrogram'],
                spectrogram_training_filename=res_item['training_spectrogram']
            )
            db.session.add(new_result)
        
        upload.status = 'COMPLETED'
        upload.progress = 100
        db.session.commit()
        
        if os.path.exists(upload_path):
            os.remove(upload_path)

    except Exception as e:
        print(f"音訊處理任務 {upload_id} 失敗: {e}")
        upload.status = 'FAILED'
        db.session.commit()
        raise

# --- 任務 2: 模型訓練 ---

@celery.task(name='app.tasks.train_yolo_model')
def train_yolo_model(upload_ids, training_run_id, model_name='yolov8n-cls.pt'):
    """
    背景任務：使用已標記的資料來訓練 YOLOv8 分類模型。
    """
    training_run = TrainingRun.query.get(training_run_id)
    if not training_run:
        print(f"找不到 TrainingRun 紀錄，ID: {training_run_id}")
        return

    try:
        training_run.status = 'RUNNING'
        training_run.progress = 5
        db.session.commit()

        print(f"\n--- [訓練任務 #{training_run_id}] 開始準備資料集 ---")
        base_dir = os.path.join(current_app.root_path, 'static', 'training_runs', str(training_run_id))
        dataset_dir = os.path.join(base_dir, 'dataset')
        
        results_to_train = Result.query.filter(Result.upload_id.in_(upload_ids), Result.label_id.isnot(None)).all()
        if not results_to_train:
            raise ValueError("找不到任何已標記的圖片來進行訓練。")

        data_by_label = defaultdict(list)
        for result in results_to_train:
            data_by_label[result.label.name].append(result)
        
        total_val_images = 0
        for label_name, items in data_by_label.items():
            random.shuffle(items)
            split_index = int(len(items) * 0.8)
            train_items, val_items = items[:split_index], items[split_index:]
            total_val_images += len(val_items)

            for item_list, target_dir in [(train_items, os.path.join(dataset_dir, 'train')), (val_items, os.path.join(dataset_dir, 'val'))]:
                label_folder = os.path.join(target_dir, label_name)
                os.makedirs(label_folder, exist_ok=True)
                for item in item_list:
                    source_path = os.path.join(current_app.root_path, 'static', item.upload.result_path, item.spectrogram_training_filename)
                    if os.path.exists(source_path):
                        shutil.copy(source_path, label_folder)
        
        if total_val_images == 0:
            raise ValueError("資料集劃分後，驗證集為空。請確保每個類別至少有 2-3 張已標記的圖片。")
        
        training_run.progress = 15
        db.session.commit()
        
        print(f"\n--- [訓練任務 #{training_run_id}] 資料集準備完成，開始訓練模型 ---")
        model = YOLO(model_name)
        
        def on_epoch_end_callback(trainer):
            current_epoch = trainer.epoch + 1
            total_epochs = trainer.epochs
            progress = 15 + int((current_epoch / total_epochs) * 80)
            if progress != training_run.progress:
                training_run.progress = progress
                db.session.commit()
                print(f"Epoch {current_epoch}/{total_epochs} 完成, 進度更新為: {progress}%")

        model.add_callback("on_epoch_end", on_epoch_end_callback)
        
        results = model.train(
            data=dataset_dir, epochs=50, imgsz=224,
            project=base_dir, name='train_results',
            val=True,
        )
        print("--- [訓練任務 #{training_run_id}] 模型訓練完成 ---")
        
        metrics = results
        class_names_list = [name for i, name in sorted(model.names.items())]
        per_class_list = []
        if hasattr(metrics, 'confusion_matrix') and metrics.confusion_matrix is not None:
            conf_matrix = metrics.confusion_matrix.matrix
            for i, name in enumerate(class_names_list):
                tp = conf_matrix[i, i]
                fp = conf_matrix[:, i].sum() - tp
                fn = conf_matrix[i, :].sum() - tp
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                per_class_list.append({
                    'name': name, 'precision': round(float(precision), 4),
                    'recall': round(float(recall), 4), 'f1-score': round(float(f1), 4)
                })

        metrics_dict = {
            'accuracy_top1': round(metrics.top1, 4) if hasattr(metrics, 'top1') else None,
            'per_class_list': per_class_list
        }

        training_run.metrics = json.dumps(metrics_dict)
        training_run.status = 'SUCCESS'
        training_run.progress = 100
        training_run.results_path = os.path.join('training_runs', str(training_run_id), 'train_results')
        db.session.commit()
        print(f"--- [訓練任務 #{training_run_id}] 成功完成並儲存結果 ---")

    except Exception as e:
        print(f"!!! [訓練任務 #{training_run_id}] 失敗: {e}")
        if 'training_run' in locals() and training_run:
            training_run.status = 'FAILURE'
            training_run.progress = 100
            db.session.commit()
        raise

# --- 任務 3: AI 自動標記 ---

def _get_h5_model_predictions(model, image_path, class_names):
    """輔助函式：處理 H5 模型的預測邏輯"""
    img_height, img_width = model.input_shape[1], model.input_shape[2]
    img = Image.open(image_path).convert('RGB').resize((img_width, img_height))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array, verbose=0)
    score = tf.nn.softmax(predictions[0])
    predicted_label_name = class_names[np.argmax(score)]
    return predicted_label_name

@celery.task(name='app.tasks.auto_label_task')
def auto_label_task(upload_id, model_path):
    """
    背景任務：使用上傳的模型(.pt 或 .h5)對指定的分析任務進行自動標記。
    """
    print(f"\n--- [自動標記任務] 開始，目標 Upload ID: {upload_id}, 模型: {model_path} ---")
    if not os.path.exists(model_path):
        print(f"!!! 失敗：找不到模型檔案 {model_path}")
        return

    model_type = os.path.splitext(model_path)[1]
    
    if model_type == '.h5' and not TENSORFLOW_AVAILABLE:
        print("!!! 失敗：偵測到 .h5 模型，但 TensorFlow 未安裝。")
        return

    try:
        model = None
        model_label_names = []
        
        # 1. 根據檔案類型載入模型並獲取標籤名稱
        if model_type == '.pt':
            model = YOLO(model_path)
            model_label_names = list(model.names.values())
            print(f"--- PT 模型載入成功。類別: {model_label_names}")
        elif model_type == '.h5':
            model = tf.keras.models.load_model(model_path)
            # 假設 class_names.json 存在於模型相同目錄下
            class_names_path = os.path.join(os.path.dirname(model_path), 'class_names.json')
            if not os.path.exists(class_names_path):
                raise FileNotFoundError("找不到 class_names.json，無法解析 .h5 模型的類別。")
            with open(class_names_path, 'r', encoding='utf-8') as f:
                model_label_names = json.load(f)
            print(f"--- H5 模型載入成功。類別: {model_label_names}")
        else:
            raise ValueError(f"不支援的模型檔案類型: {model_type}")

        # 2. 準備標籤資料
        existing_labels = {label.name: label for label in Label.query.all()}
        for name in model_label_names:
            if name not in existing_labels:
                new_label = Label(name=name)
                db.session.add(new_label)
                print(f"--- 在資料庫中新增標籤: '{name}' ---")
        db.session.commit()
        all_labels_map = {label.name: label.id for label in Label.query.all()}
        
        # 3. 查詢並執行預測
        results_to_label = Result.query.filter_by(upload_id=upload_id).all()
        print(f"--- 找到 {len(results_to_label)} 張圖片，開始預測... ---")
        
        count = 0
        for result in results_to_label:
            image_path = os.path.join(current_app.root_path, 'static', result.spectrogram_training_url)
            if not os.path.exists(image_path): continue

            predicted_label_name = None
            if model_type == '.pt':
                predictions = model(image_path, verbose=False)
                if predictions and predictions[0].probs is not None:
                    top1_index = predictions[0].probs.top1
                    predicted_label_name = model.names[top1_index]
            elif model_type == '.h5':
                predicted_label_name = _get_h5_model_predictions(model, image_path, model_label_names)

            if predicted_label_name:
                predicted_label_id = all_labels_map.get(predicted_label_name)
                result.label_id = predicted_label_id
                count += 1
        
        db.session.commit()
        print(f"--- 自動標記完成！共更新了 {count} 筆紀錄。 ---")

    except Exception as e:
        print(f"!!! [自動標記任務] 執行期間發生錯誤: {e}")
        db.session.rollback()
        raise
        
    finally:
        # 4. 清理暫存模型
        if os.path.exists(model_path):
            os.remove(model_path)
            print(f"--- 已刪除暫存模型檔案: {model_path} ---")
        # 清理可能存在的 h5 模型的 class_names.json
        class_names_path = os.path.join(os.path.dirname(model_path), 'class_names.json')
        if os.path.exists(class_names_path):
             os.remove(class_names_path)
             print(f"--- 已刪除暫存的 class_names.json ---")

