import os
import shutil
import random
import json
import numpy as np
from collections import defaultdict
from flask import current_app

from .. import db
from ..models import Result, CetaceanInfo, Label, TrainingRun

class CnnTrainer:
    @staticmethod
    def train(upload_ids, training_run_id, model_name='resnet18', train_params=None):
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
