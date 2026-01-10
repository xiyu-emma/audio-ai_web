import os
import json
import shutil
import csv
import io
import zipfile
from functools import wraps
from datetime import datetime
from flask import (
    Blueprint, current_app, render_template, request, redirect, url_for, jsonify, send_file
)
from werkzeug.utils import secure_filename

# 從 __init__.py 引入 db 和 celery 實例
from . import db, celery
# 引入所有相關模型
from .models import AudioInfo, Result, Label, TrainingRun, ProjectInfo, PointInfo, CetaceanInfo

main_bp = Blueprint('main', __name__)

# --- Helper Functions ---

# API 權限驗證 (給洋聲使用)
API_TOKENS = {
    "yang_sheng_partner": "sk_test_1234567890abcdef"
}

def require_api_token(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({'success': False, 'error': '缺少 Authorization Header'}), 401
        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != 'bearer':
            return jsonify({'success': False, 'error': 'Authorization 格式錯誤'}), 401
        token = parts[1]
        if token not in API_TOKENS.values():
            return jsonify({'success': False, 'error': '無效的 API Token'}), 403
        return f(*args, **kwargs)
    return decorated_function

# --- 主要頁面路由 ---

@main_bp.route('/')
def index():
    """渲染主上傳頁面。"""
    return render_template('index.html')

@main_bp.route('/history')
def history():
    """渲染分析歷史紀錄頁面。"""
    sort_order = request.args.get('sort', 'desc')
    query = AudioInfo.query
    
    if sort_order == 'asc':
        all_uploads = query.order_by(AudioInfo.record_time.asc()).all()
    else:
        all_uploads = query.order_by(AudioInfo.record_time.desc()).all()
        
    return render_template('history.html', uploads=all_uploads, current_sort=sort_order)

@main_bp.route('/results/<int:upload_id>')
def results(upload_id):
    """
    渲染單次分析結果頁面。
    在這裡進行動態配對，將 Result 的圖片 URL 塞給 CetaceanInfo 物件。
    """
    page = request.args.get('page', 1, type=int)
    upload_record = AudioInfo.query.get_or_404(upload_id)
    params = upload_record.get_params()
    
    # 計算顯示用的秒數
    try:
        segment_duration = float(params.get('segment_duration', 2.0))
        overlap_percent = float(params.get('overlap', 50))
        hop_length_seconds = segment_duration * (1 - overlap_percent / 100.0)
    except (ValueError, TypeError):
        hop_length_seconds = 1.0

    # 1. 查詢資料分頁 (CetaceanInfo)
    pagination = CetaceanInfo.query.filter_by(audio_id=upload_id).order_by(CetaceanInfo.id.asc()).paginate(
        page=page, per_page=10, error_out=False
    )
    
    # 2. 查詢對應的圖片 (Result)
    offset = (page - 1) * 10
    limit = 10
    results_slice = Result.query.filter_by(upload_id=upload_id).order_by(Result.id.asc()).offset(offset).limit(limit).all()
    
    # 3. 動態合併
    for cetacean, result in zip(pagination.items, results_slice):
        setattr(cetacean, 'spectrogram_url', result.spectrogram_url)
        setattr(cetacean, 'audio_url', result.audio_url)
        label_obj = Label.query.get(cetacean.event_type) if cetacean.event_type else None
        setattr(cetacean, 'label_name', label_obj.name if label_obj else 'Unknown')

    return render_template(
        'result.html',
        upload=upload_record,
        pagination=pagination,
        hop_length_seconds=hop_length_seconds
    )

@main_bp.route('/labeling/<int:upload_id>')
def labeling_page(upload_id):
    """渲染標記頁面。"""
    page = request.args.get('page', 1, type=int)
    upload_record = AudioInfo.query.get_or_404(upload_id)
    
    pagination = CetaceanInfo.query.filter_by(audio_id=upload_id).order_by(CetaceanInfo.id.asc()).paginate(
        page=page, per_page=50, error_out=False
    )
    
    offset = (page - 1) * 50
    results_slice = Result.query.filter_by(upload_id=upload_id).order_by(Result.id.asc()).offset(offset).limit(50).all()
    
    for cetacean, result in zip(pagination.items, results_slice):
        setattr(cetacean, 'spectrogram_url', result.spectrogram_url)
        setattr(cetacean, 'spectrogram_training_url', result.spectrogram_training_url)
        setattr(cetacean, 'label_id', cetacean.event_type) 

    return render_template('label.html', upload=upload_record, pagination=pagination)

# --- 下載功能路由 ---

@main_bp.route('/download_dataset_zip/<int:upload_id>')
def download_dataset_zip(upload_id):
    """打包下載：結合 Result (檔案) 與 CetaceanInfo (標籤)。"""
    upload = AudioInfo.query.get_or_404(upload_id)
    
    results_all = Result.query.filter_by(upload_id=upload_id).order_by(Result.id).all()
    cetaceans_all = CetaceanInfo.query.filter_by(audio_id=upload_id).order_by(CetaceanInfo.id).all()
    
    folder_path = os.path.join(current_app.root_path, 'static', upload.result_path)
    if not os.path.exists(folder_path):
        return "找不到結果資料夾，無法下載。", 404

    params = upload.get_params()
    try:
        segment_duration = float(params.get('segment_duration', 2.0))
        overlap_percent = float(params.get('overlap', 50))
        hop_length = segment_duration * (1 - overlap_percent / 100.0)
    except:
        segment_duration = 2.0
        hop_length = 1.0

    def format_time(seconds):
        m = int(seconds // 60)
        s = seconds % 60
        return f"{m:02d}:{s:06.3f}"

    memory_file = io.BytesIO()
    try:
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            # A. 加入實體檔案 
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    filename_lower = file.lower()
                    arcname = file
                    should_include = False
                    
                    if filename_lower.endswith(('.wav', '.mp3')):
                        arcname = f"audio/{file}"
                        should_include = True
                    elif filename_lower.endswith(('.png', '.jpg')) and '_spec_training_' in filename_lower:
                        arcname = f"images/{file}"
                        should_include = True
                    
                    if should_include:
                        zf.write(file_path, arcname)
            
            # B. 生成 labels.csv 
            csv_buffer = io.StringIO()
            csv_writer = csv.writer(csv_buffer)
            csv_writer.writerow(['filename', 'event_type', 'label_name', 'time_segment', 'detect_type'])
            
            label_map = {l.id: l.name for l in Label.query.all()}
            
            for i, (res, cet) in enumerate(zip(results_all, cetaceans_all)):
                fname = res.spectrogram_training_filename
                csv_filename = f"images/{fname}"
                etype = cet.event_type
                label_name = label_map.get(etype, 'Unknown') if etype else 'Unknown'
                
                start_seconds = i * hop_length
                end_seconds = start_seconds + segment_duration
                time_str = f"{format_time(start_seconds)} - {format_time(end_seconds)}"
                
                csv_writer.writerow([
                    csv_filename, etype, label_name, time_str, 
                    'AI' if cet.detect_type == 1 else 'Manual'
                ])
            
            zf.writestr('labels.csv', csv_buffer.getvalue())

        memory_file.seek(0)
        download_filename = f"dataset_{upload.id}_{secure_filename(upload.file_name)}.zip"
        return send_file(memory_file, mimetype='application/zip', as_attachment=True, download_name=download_filename)

    except Exception as e:
        print(f"打包 ZIP 時發生錯誤: {e}")
        return f"打包失敗: {e}", 500

# --- 上傳與背景任務路由 ---

@main_bp.route('/upload', methods=['POST'])
def upload():
    """Web 上傳介面"""
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
    except: return "參數錯誤", 400

    if file:
        filename = secure_filename(file.filename)
        file_ext = os.path.splitext(filename)[1].lower().replace('.', '')
        params_json = json.dumps(params_dict)
        
        default_point = PointInfo.query.first()
        point_id = default_point.id if default_point else None

        new_audio = AudioInfo(
            file_name=filename, file_path="pending", file_type=file_ext,
            result_path="pending", params=params_json, status='PENDING', point_id=point_id
        )
        db.session.add(new_audio)
        db.session.commit()
        
        upload_id = new_audio.id
        result_dir_relative = os.path.join('results', str(upload_id))
        result_dir_absolute = os.path.join(current_app.root_path, 'static', result_dir_relative)
        os.makedirs(result_dir_absolute, exist_ok=True)
        
        upload_filename = f"{upload_id}_{filename}"
        upload_path_absolute = os.path.join(current_app.root_path, current_app.config['UPLOAD_FOLDER'], upload_filename)
        file.save(upload_path_absolute)
        
        new_audio.file_path = upload_path_absolute
        new_audio.result_path = result_dir_relative
        db.session.commit()
        
        celery.send_task('app.tasks.process_audio_task', args=[upload_id])
        return redirect(url_for('main.history', new_upload_id=upload_id)) # 導向帶有新 ID 的網址以觸發輪詢
    return redirect(url_for('main.index'))

@main_bp.route('/history/delete_selected', methods=['POST'])
def delete_selected_uploads():
    """批次刪除分析紀錄"""
    upload_ids = request.form.getlist('upload_ids')
    if not upload_ids: return redirect(url_for('main.history'))
    
    uploads = AudioInfo.query.filter(AudioInfo.id.in_(upload_ids)).all()
    for u in uploads:
        path = os.path.join(current_app.root_path, 'static', u.result_path)
        if os.path.exists(path): shutil.rmtree(path, ignore_errors=True)
        if u.file_path and os.path.exists(u.file_path): os.remove(u.file_path)
        db.session.delete(u)
    db.session.commit()
    return redirect(url_for('main.history'))

# --- 狀態查詢 API (支援前端動態進度條) ---

@main_bp.route('/api/upload/<int:upload_id>/status')
def upload_status(upload_id):
    """查詢分析任務狀態"""
    upload = AudioInfo.query.get_or_404(upload_id)
    return jsonify({
        'id': upload.id,
        'status': upload.status,
        'progress': upload.progress
    })

@main_bp.route('/api/training/<int:run_id>/status')
def run_status(run_id):
    """查詢訓練任務狀態"""
    run = TrainingRun.query.get_or_404(run_id)
    return jsonify({
        'id': run.id,
        'status': run.status,
        'progress': run.progress
    })

# --- 訓練與標記路由 ---

@main_bp.route('/api/cetacean/<int:cetacean_id>/label', methods=['POST'])
def update_cetacean_label(cetacean_id):
    """API: 更新單筆鯨豚資訊的標籤"""
    cetacean = CetaceanInfo.query.get_or_404(cetacean_id)
    data = request.get_json()
    label_id = data.get('label_id') 
    
    if label_id is not None:
        cetacean.event_type = int(label_id)
        cetacean.detect_type = 0 # 0: 人工標記
        db.session.commit()
        return jsonify({'success': True})
    return jsonify({'error': 'Missing label_id'}), 400

@main_bp.route('/labeling/auto_label', methods=['POST'])
def auto_label():
    """啟動自動標記任務"""
    if 'model_file' not in request.files: return "無檔案", 400
    file = request.files['model_file']
    upload_id = request.form.get('upload_id')
    
    if file and upload_id:
        filename = secure_filename(file.filename)
        temp_dir = os.path.join(current_app.root_path, 'static', 'temp_models')
        os.makedirs(temp_dir, exist_ok=True)
        saved_path = os.path.join(temp_dir, f"{datetime.now().strftime('%Y%m%d')}_{upload_id}_{filename}")
        file.save(saved_path)
        
        celery.send_task('app.tasks.auto_label_task', args=[int(upload_id), saved_path])
        return redirect(url_for('main.labeling_page', upload_id=upload_id))
    return "錯誤", 400

@main_bp.route('/training/start', methods=['POST'])
def start_training():
    """啟動訓練任務"""
    upload_ids = request.form.getlist('upload_ids')
    model_name = request.form.get('model_name', 'yolov8n-cls.pt')
    
    if upload_ids:
        params = {'model_name': model_name, 'upload_ids': upload_ids}
        run = TrainingRun(status='PENDING', params=json.dumps(params))
        db.session.add(run)
        db.session.commit()
        celery.send_task('app.tasks.train_yolo_model', args=[upload_ids, run.id, model_name])
        return redirect(url_for('main.training_status', new_run_id=run.id)) # 導向並觸發輪詢
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
    """渲染單次模型訓練的詳細報告 (包含路徑修復)"""
    run = TrainingRun.query.get_or_404(run_id)
    if run.status != 'SUCCESS' or not run.results_path:
        return "此訓練任務尚未成功完成，無法查看報告。", 404
    
    results_base_path = run.results_path.replace('\\', '/')
    if results_base_path.startswith('/'):
        results_base_path = results_base_path[1:]

    report_images = {
        'results': f"{results_base_path}/results.png",
        'confusion_matrix': f"{results_base_path}/confusion_matrix.png",
        'val_batch0_labels': f"{results_base_path}/val_batch0_labels.jpg",
        'val_batch0_pred': f"{results_base_path}/val_batch0_pred.jpg",
    }
    
    metrics = run.get_metrics()
    return render_template('training_report.html', run=run, images=report_images, metrics=metrics)

# --- 洋聲專用 API ---

@main_bp.route('/api/v1/upload', methods=['POST'])
@require_api_token
def api_upload_audio():
    """洋聲 API: 上傳並觸發分析"""
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    try:
        point_id = int(request.form.get('point_id'))
        if not PointInfo.query.get(point_id): return jsonify({'error': 'Invalid point_id'}), 404
    except: return jsonify({'error': 'Missing point_id'}), 400

    filename = secure_filename(file.filename)
    new_audio = AudioInfo(
        file_name=filename, file_path="pending", file_type="wav",
        point_id=point_id, result_path="pending", status='PENDING'
    )
    db.session.add(new_audio)
    db.session.commit()
    
    return jsonify({'success': True, 'upload_id': new_audio.id}), 201

# --- 一般 API ---
@main_bp.route('/api/labels', methods=['GET', 'POST'])
def handle_labels():
    if request.method == 'GET':
        return jsonify([{'id': l.id, 'name': l.name} for l in Label.query.all()])
    if request.method == 'POST':
        data = request.get_json()
        if Label.query.filter_by(name=data['name']).first(): return jsonify({'error': 'Exists'}), 409
        new_label = Label(name=data['name'])
        db.session.add(new_label)
        db.session.commit()
        return jsonify({'id': new_label.id, 'name': new_label.name}), 201