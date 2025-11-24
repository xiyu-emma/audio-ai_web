import os
import json
import shutil
import csv
import io
import zipfile
from datetime import datetime
from flask import (
    Blueprint, current_app, render_template, request, redirect, url_for, jsonify, send_file
)
from werkzeug.utils import secure_filename

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

# --- 下載功能路由 ---
# 修改 app/main.py

@main_bp.route('/download_dataset_zip/<int:upload_id>')
def download_dataset_zip(upload_id):
    """
    將指定 upload_id 的結果資料夾打包成 ZIP。
    
    修改重點：
    1. 【過濾檔案】：只打包「訓練用頻譜圖 (無座標)」和「音訊檔」，排除「顯示用頻譜圖 (有座標)」。
    2. 【分流存放】：圖片存入 images/，音訊存入 audio/。
    3. 【CSV 內容】：包含 filename (路徑), label_name, time_segment (時間段)。
    """
    upload = Upload.query.get_or_404(upload_id)
    
    # 確保按順序取出，以便計算時間
    results = Result.query.filter_by(upload_id=upload_id).order_by(Result.id).all()
    
    folder_path = os.path.join(current_app.root_path, 'static', upload.result_path)
    if not os.path.exists(folder_path):
        return "找不到結果資料夾，無法下載。", 404

    # --- 1. 計算時間參數 ---
    params = upload.get_params()
    try:
        segment_duration = float(params.get('segment_duration', 2.0))
        overlap_percent = float(params.get('overlap', 50))
        hop_length = segment_duration * (1 - overlap_percent / 100.0)
    except (ValueError, TypeError):
        segment_duration = 2.0
        hop_length = 1.0

    def format_time(seconds):
        m = int(seconds // 60)
        s = seconds % 60
        return f"{m:02d}:{s:06.3f}"

    memory_file = io.BytesIO()

    try:
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            # --- A. 加入實體檔案 (加入過濾邏輯) ---
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    filename_lower = file.lower()
                    
                    # 邏輯判斷：決定是否要打包此檔案
                    should_include = False
                    arcname = file
                    
                    # 情況 1: 是音訊檔 -> 保留
                    if filename_lower.endswith(('.wav', '.mp3')):
                        arcname = f"audio/{file}"
                        should_include = True
                    
                    # 情況 2: 是圖片檔，且是「訓練用」圖檔 (包含 _spec_training_) -> 保留
                    # audio_utils.py 產生的檔名格式為: {basename}_spec_training_{i}.png 
                    elif filename_lower.endswith(('.png', '.jpg')) and '_spec_training_' in filename_lower:
                        arcname = f"images/{file}"
                        should_include = True
                    
                    # 情況 3: 是「顯示用」圖檔 (包含 _spec_display_) -> 跳過 (不打包)
                    elif '_spec_display_' in filename_lower:
                        should_include = False
                    
                    # 執行寫入
                    if should_include:
                        zf.write(file_path, arcname)
            
            # --- B. 生成 labels.csv ---
            csv_buffer = io.StringIO()
            csv_writer = csv.writer(csv_buffer)
            
            # 表頭
            csv_writer.writerow(['filename', 'label_name', 'time_segment'])
            
            for i, res in enumerate(results):
                # 這裡使用的是 spectrogram_training_filename，對應上面過濾保留的圖片 [cite: 2]
                fname = res.spectrogram_training_filename
                csv_filename = f"images/{fname}"
                
                label_name = res.label.name if res.label else ""
                
                # 計算時間
                start_seconds = i * hop_length
                end_seconds = start_seconds + segment_duration
                time_str = f"{format_time(start_seconds)} - {format_time(end_seconds)}"
                
                csv_writer.writerow([csv_filename, label_name, time_str])
            
            zf.writestr('labels.csv', csv_buffer.getvalue())

        memory_file.seek(0)
        download_filename = f"dataset_{upload.id}_{secure_filename(upload.original_filename)}.zip"

        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=download_filename
        )

    except Exception as e:
        print(f"打包 ZIP 時發生錯誤: {e}")
        return f"打包失敗: {e}", 500

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