import os
import json
import shutil
from flask import render_template, request, redirect, url_for, current_app
from ..main_router import main_bp

from ..models import TrainingRun
from .. import db, celery

@main_bp.route('/training/start', methods=['POST'])
def start_training():
    """啟動訓練任務 - 支援多模型選擇和自訂參數"""
    upload_ids = request.form.getlist('upload_ids')
    model_type = request.form.get('model_type', 'yolov8n-cls')
    
    # 收集訓練參數
    train_params = {
        'epochs': int(request.form.get('epochs', 50)),
        'batch_size': int(request.form.get('batch_size', 16)),
        'learning_rate': float(request.form.get('learning_rate', 0.001)),
        'image_size': int(request.form.get('image_size', 224))
    }
    
    if upload_ids:
        # 儲存參數到資料庫
        params = {
            'model_type': model_type,
            'upload_ids': upload_ids,
            **train_params
        }
        run = TrainingRun(status='PENDING', params=json.dumps(params))
        db.session.add(run)
        db.session.commit()
        
        # 根據模型類型選擇對應的任務
        if model_type.startswith('yolov8'):
            task_name = 'app.tasks.train_yolo_model'
            model_name = f"{model_type}.pt"
        else:
            # CNN 模型 (resnet18, efficientnet_b0)
            task_name = 'app.tasks.train_cnn_model'
            model_name = model_type
        
        celery.send_task(task_name, args=[upload_ids, run.id, model_name, train_params])
        return redirect(url_for('main.training_status', new_run_id=run.id))
    
    return redirect(url_for('main.history'))

@main_bp.route('/training/delete_selected', methods=['POST'])
def delete_selected_runs():
    """批次刪除選定的訓練紀錄"""
    run_ids = request.form.getlist('run_ids')
    if not run_ids:
        return redirect(url_for('main.training_status'))
    runs_to_delete = TrainingRun.query.filter(TrainingRun.id.in_(run_ids)).all()
    for run in runs_to_delete:
        if run.results_path:
            # 嘗試刪除實體資料夾
            base_path = os.path.dirname(os.path.dirname(run.results_path))
            physical_path = os.path.join(current_app.root_path, 'static', base_path, str(run.id))
            try:
                if os.path.isdir(physical_path):
                    shutil.rmtree(physical_path)
                    print(f"已刪除資料夾: {physical_path}")
            except OSError as e:
                print(f"刪除訓練資料夾時發生錯誤 {physical_path}: {e.strerror}")
        db.session.delete(run)
    db.session.commit()
    return redirect(url_for('main.training_status'))

@main_bp.route('/training/status')
def training_status():
    runs = TrainingRun.query.order_by(TrainingRun.timestamp.desc()).all()
    return render_template('training_status.html', runs=runs)

@main_bp.route('/training/report/<int:run_id>')
def training_report(run_id):
    """渲染單次模型訓練的詳細報告 (支援多模型類型)"""
    run = TrainingRun.query.get_or_404(run_id)
    if run.status != 'SUCCESS' or not run.results_path:
        return "此訓練任務尚未成功完成，無法查看報告。", 404
    
    # 解析訓練參數
    params = run.get_params() if hasattr(run, 'get_params') else {}
    if isinstance(params, str):
        params = json.loads(params) if params else {}
    
    model_type = params.get('model_type', 'yolov8n-cls')
    is_yolo = model_type.startswith('yolov8')
    
    # 模型顯示名稱對照
    model_display_names = {
        'yolov8n-cls': 'YOLOv8n Classification',
        'yolov8s-cls': 'YOLOv8s Classification',
        'resnet18': 'ResNet18 (PyTorch)',
        'efficientnet_b0': 'EfficientNet-B0 (PyTorch)'
    }
    
    results_base_path = run.results_path.replace('\\', '/')
    if results_base_path.startswith('/'):
        results_base_path = results_base_path[1:]

    # 根據模型類型準備可用的圖片
    report_images = {
        'results': f"{results_base_path}/results.png",
    }
    
    # 混淆矩陣：優先使用手動生成的版本
    import os
    static_path = os.path.join(current_app.root_path, 'static')
    manual_cm = os.path.join(static_path, results_base_path, 'confusion_matrix_manual.png')
    original_cm = os.path.join(static_path, results_base_path, 'confusion_matrix.png')
    
    if os.path.exists(manual_cm):
        report_images['confusion_matrix'] = f"{results_base_path}/confusion_matrix_manual.png"
    elif os.path.exists(original_cm):
        report_images['confusion_matrix'] = f"{results_base_path}/confusion_matrix.png"
    else:
        report_images['confusion_matrix'] = f"{results_base_path}/confusion_matrix.png"
    
    # YOLO 特有圖片
    if is_yolo:
        report_images.update({
            'val_batch0_labels': f"{results_base_path}/val_batch0_labels.jpg",
            'val_batch0_pred': f"{results_base_path}/val_batch0_pred.jpg",
        })
    
    metrics = run.get_metrics()
    
    # 查詢此次訓練使用的音檔
    from ..models import AudioInfo
    upload_ids = params.get('upload_ids', [])
    used_audios = []
    if upload_ids:
        used_audios = AudioInfo.query.filter(AudioInfo.id.in_(upload_ids)).all()
    
    return render_template(
        'training_report.html', 
        run=run, 
        images=report_images, 
        metrics=metrics,
        params=params,
        model_type=model_type,
        model_display_name=model_display_names.get(model_type, model_type),
        is_yolo=is_yolo,
        used_audios=used_audios
    )
