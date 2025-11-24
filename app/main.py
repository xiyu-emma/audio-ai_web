import os
import json
import shutil
from datetime import datetime
from flask import (
    Blueprint, current_app, render_template, request, redirect, url_for, jsonify, Response
)
from werkzeug.utils import secure_filename
import csv
import io

# 從 __init__.py 引入 db 和 celery 實例
from . import db, celery
from .models import Upload, Result, Label, TrainingRun

main_bp = Blueprint('main', __name__)

# --- 主要頁面路由 ---

@main_bp.route('/')
def index():
    """渲染主上傳頁面。"""
    return render_template('index.html')

@main_bp.route('/history')
def history():
    """渲染分析歷史紀錄頁面。"""
    sort_order = request.args.get('sort', 'desc')
    query = Upload.query
    if sort_order == 'asc':
        all_uploads = query.order_by(Upload.upload_timestamp.asc()).all()
    else:
        all_uploads = query.order_by(Upload.upload_timestamp.desc()).all()
    return render_template('history.html', uploads=all_uploads, current_sort=sort_order)

@main_bp.route('/results/<int:upload_id>')
def results(upload_id):
    """渲染單次分析結果的詳細頁面。"""
    page = request.args.get('page', 1, type=int)
    upload_record = Upload.query.get_or_404(upload_id)
    params = upload_record.get_params()
    try:
        segment_duration = float(params.get('segment_duration', 2.0))
        overlap_percent = float(params.get('overlap', 50))
        overlap_ratio = overlap_percent / 100.0
        hop_length_seconds = segment_duration * (1 - overlap_ratio)
    except (ValueError, TypeError):
        hop_length_seconds = 1.0
    pagination = Result.query.filter_by(upload_id=upload_id).order_by(Result.id.asc()).paginate(
        page=page, per_page=10, error_out=False
    )
    return render_template(
        'result.html',
        upload=upload_record,
        pagination=pagination,
        hop_length_seconds=hop_length_seconds
    )

@main_bp.route('/labeling/<int:upload_id>')
def labeling_page(upload_id):
    """渲染頻譜圖標記頁面。"""
    page = request.args.get('page', 1, type=int)
    upload_record = Upload.query.get_or_404(upload_id)
    pagination = Result.query.filter_by(upload_id=upload_id).order_by(Result.id.asc()).paginate(
        page=page, per_page=50, error_out=False
    )
    return render_template('label.html', upload=upload_record, pagination=pagination)

# --- 上傳與背景任務路由 ---

@main_bp.route('/upload', methods=['POST'])
def upload():
    """處理檔案上傳並啟動背景分析任務。"""
    if 'file' not in request.files: return redirect(request.url)
    file = request.files['file']
    if file.filename == '': return redirect(request.url)
    try:
        params_dict = {
            'spec_type': request.form['spec_type'],
            'segment_duration': float(request.form['segment_duration']),
            'overlap': float(request.form['overlap']),
            'sample_rate': request.form.get('sample_rate', 'None'),
            'channels': request.form.get('channels', 'mono')
        }
    except (KeyError, ValueError):
        return "提供了無效的參數。", 400
    if file:
        filename = secure_filename(file.filename)
        params_json = json.dumps(params_dict)
        new_upload = Upload(
            original_filename=filename,
            result_path="pending",
            params=params_json,
            status='PENDING'
        )
        db.session.add(new_upload)
        db.session.commit()
        upload_id = new_upload.id
        result_dir_name = os.path.join(current_app.root_path, 'static', 'results', str(upload_id))
        os.makedirs(result_dir_name, exist_ok=True)
        new_upload.result_path = os.path.join('results', str(upload_id))
        db.session.commit()
        upload_path = os.path.join(current_app.root_path, current_app.config['UPLOAD_FOLDER'], f"{upload_id}_{filename}")
        file.save(upload_path)
        celery.send_task('app.tasks.process_audio_task', args=[upload_id])
        return redirect(url_for('main.history', new_upload_id=upload_id))
    return redirect(url_for('main.index'))

@main_bp.route('/history/delete_selected', methods=['POST'])
def delete_selected_uploads():
    """批次刪除選定的分析紀錄。"""
    upload_ids = request.form.getlist('upload_ids')
    if not upload_ids:
        return redirect(url_for('main.history'))
    uploads_to_delete = Upload.query.filter(Upload.id.in_(upload_ids)).all()
    for upload in uploads_to_delete:
        physical_path = os.path.join(current_app.root_path, 'static', upload.result_path)
        try:
            if os.path.isdir(physical_path): shutil.rmtree(physical_path)
        except OSError as e:
            print(f"刪除資料夾時發生錯誤 {physical_path}: {e}")
        temp_upload_file = os.path.join(current_app.root_path, current_app.config['UPLOAD_FOLDER'], f"{upload.id}_{upload.original_filename}")
        if os.path.exists(temp_upload_file):
            try: os.remove(temp_upload_file)
            except OSError as e: print(f"刪除暫存檔案時發生錯誤 {temp_upload_file}: {e}")
        db.session.delete(upload)
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"刪除資料庫紀錄時發生錯誤: {e}")
    return redirect(url_for('main.history'))

# --- 訓練相關路由 ---

@main_bp.route('/training/status')
def training_status():
    """渲染模型訓練狀態頁面。"""
    runs = TrainingRun.query.order_by(TrainingRun.timestamp.desc()).all()
    return render_template('training_status.html', runs=runs)

@main_bp.route('/training/report/<int:run_id>')
def training_report(run_id):
    """渲染單次模型訓練的詳細報告。"""
    run = TrainingRun.query.get_or_404(run_id)
    if run.status != 'SUCCESS' or not run.results_path:
        return "此訓練任務尚未成功完成，無法查看報告。", 404
    results_base_path = run.results_path
    report_images = {
        'results': os.path.join(results_base_path, 'results.png'),
        'confusion_matrix': os.path.join(results_base_path, 'confusion_matrix.png'),
        'val_batch0_labels': os.path.join(results_base_path, 'val_batch0_labels.jpg'),
        'val_batch0_pred': os.path.join(results_base_path, 'val_batch0_pred.jpg'),
    }
    metrics = run.get_metrics()
    return render_template('training_report.html', run=run, images=report_images, metrics=metrics)

@main_bp.route('/training/start', methods=['POST'])
def start_training():
    """啟動模型訓練的背景任務。"""
    upload_ids = request.form.getlist('upload_ids')
    if not upload_ids:
        return redirect(url_for('main.history'))
    model_name = request.form.get('model_name', 'yolov8n-cls.pt')
    params = {'model_name': model_name, 'upload_ids': upload_ids}
    new_run = TrainingRun(status='PENDING', params=json.dumps(params))
    db.session.add(new_run)
    db.session.commit()
    celery.send_task('app.tasks.train_yolo_model', args=[upload_ids, new_run.id, model_name])
    return redirect(url_for('main.training_status', new_run_id=new_run.id))

@main_bp.route('/labeling/auto_label', methods=['POST'])
def auto_label():
    """處理模型上傳並啟動背景自動標記任務"""
    if 'model_file' not in request.files:
        return "找不到模型檔案。", 400
    
    file = request.files['model_file']
    upload_id = request.form.get('upload_id')

    if file.filename == '' or not upload_id:
        return "缺少模型檔案或任務ID。", 400

    if file and (file.filename.endswith('.pt') or file.filename.endswith('.h5')):
        filename = secure_filename(file.filename)
        
        temp_model_dir = os.path.join(current_app.root_path, 'static', 'temp_models')
        os.makedirs(temp_model_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        saved_model_path = os.path.join(temp_model_dir, f"{timestamp}_{upload_id}_{filename}")
        file.save(saved_model_path)

        celery.send_task('app.tasks.auto_label_task', args=[int(upload_id), saved_model_path])

        return redirect(url_for('main.labeling_page', upload_id=upload_id))

    return "檔案格式不正確，請上傳 .pt 或 .h5 檔案。", 400

@main_bp.route('/training/delete_selected', methods=['POST'])
def delete_selected_runs():
    """批次刪除選定的訓練紀錄。"""
    run_ids = request.form.getlist('run_ids')
    if not run_ids:
        return redirect(url_for('main.training_status'))
    runs_to_delete = TrainingRun.query.filter(TrainingRun.id.in_(run_ids)).all()
    for run in runs_to_delete:
        if run.results_path:
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

# --- API Routes ---

@main_bp.route('/api/upload/<int:upload_id>/status')
def get_upload_status(upload_id):
    """API: 獲取單個分析任務的狀態和進度。"""
    upload = Upload.query.get_or_404(upload_id)
    return jsonify({
        'id': upload.id,
        'status': upload.status,
        'progress': upload.progress
    })

@main_bp.route('/api/training/<int:run_id>/status')
def get_training_run_status(run_id):
    """API: 獲取單個訓練任務的狀態和進度。"""
    run = TrainingRun.query.get_or_404(run_id)
    return jsonify({
        'id': run.id,
        'status': run.status,
        'progress': run.progress
    })

@main_bp.route('/api/labels', methods=['GET', 'POST'])
def handle_labels():
    """API: 獲取所有標籤或新增一個標籤。"""
    if request.method == 'GET':
        labels = Label.query.order_by(Label.name).all()
        return jsonify([{'id': label.id, 'name': label.name} for label in labels])
    if request.method == 'POST':
        data = request.get_json()
        if not data or not data.get('name'):
            return jsonify({'error': 'Label name is required'}), 400
        if Label.query.filter_by(name=data['name']).first():
            return jsonify({'error': 'Label name already exists'}), 409
        new_label = Label(name=data['name'])
        db.session.add(new_label)
        db.session.commit()
        return jsonify({'id': new_label.id, 'name': new_label.name}), 201

@main_bp.route('/api/labels/<int:label_id>', methods=['DELETE'])
def delete_label(label_id):
    """API: 刪除指定的標籤。"""
    label = Label.query.get_or_404(label_id)
    db.session.delete(label)
    db.session.commit()
    return jsonify({'success': True})

@main_bp.route('/api/results/<int:result_id>/label', methods=['POST'])
def update_result_label(result_id):
    """API: 更新單個頻譜圖的標籤。"""
    result = Result.query.get_or_404(result_id)
    data = request.get_json()
    label_id = data.get('label_id')
    if label_id is not None and not isinstance(label_id, int):
        return jsonify({'error': 'Invalid label_id format'}), 400
    result.label_id = label_id
    db.session.commit()
    return jsonify({'success': True})

@main_bp.route('/labeling/<int:upload_id>/download_csv')
def download_labels_csv(upload_id):
    """下載帶有標籤的頻譜圖數據CSV檔案。"""
    upload_record = Upload.query.get_or_404(upload_id)
    
    # 獲取所有結果和標籤
    results = Result.query.filter_by(upload_id=upload_id).order_by(Result.id.asc()).all()
    
    # 創建CSV內容
    output = io.StringIO()
    writer = csv.writer(output)
    
    # 寫入標題行
    writer.writerow([
        'Result_ID', 'Spectrogram_Filename', 'Audio_Filename', 
        'Label_ID', 'Label_Name', 'Spectrogram_URL', 'Spectrogram_Training_URL'
    ])
    
    # 寫入數據行
    for result in results:
        label_name = result.label.name if result.label else ''
        label_id = result.label_id if result.label_id else ''
        audio_filename = result.audio_filename if result.audio_filename else ''
        
        writer.writerow([
            result.id,
            result.spectrogram_filename,
            audio_filename,
            label_id,
            label_name,
            result.spectrogram_url,
            result.spectrogram_training_url
        ])
    
    # 創建回應
    output.seek(0)
    csv_content = output.getvalue()
    output.close()
    
    # 生成檔案名稱
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 確保檔案名稱是安全的
    safe_filename = secure_filename(upload_record.original_filename)
    filename = f"spectrogram_labels_{safe_filename}_{upload_id}_{timestamp}.csv"
    
    # 創建回應物件
    response = Response(
        csv_content,
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename="{filename}"'}
    )
    
    return response

