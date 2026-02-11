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
from .models import AudioInfo, Result, Label, TrainingRun, ProjectInfo, PointInfo, CetaceanInfo, BBoxAnnotation

main_bp = Blueprint('main', __name__)

# --- Helper Functions ---

# API 權限驗證 (給洋聲使用) - 從環境變數取得 Token
API_TOKENS = {
    "yang_sheng_partner": os.environ.get('YANG_SHENG_API_TOKEN', 'sk_test_1234567890abcdef')
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
        hop_length_seconds=hop_length_seconds,
        params=params
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
    
    # 查詢已完成的訓練任務供自動標記使用
    training_runs = TrainingRun.query.filter_by(status='SUCCESS').order_by(TrainingRun.timestamp.desc()).all()

    return render_template('label.html', upload=upload_record, pagination=pagination, training_runs=training_runs)

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

            # C. 生成 bbox_annotations.csv (框選標記資料)
            bbox_csv_buffer = io.StringIO()
            bbox_csv_writer = csv.writer(bbox_csv_buffer)
            bbox_csv_writer.writerow([
                'filename', 'segment_index', 'label', 
                'time_start_sec', 'time_end_sec', 
                'freq_min_hz', 'freq_max_hz'
            ])
            
            # 取得頻譜圖參數以計算頻率軸
            try:
                sample_rate = params.get('sample_rate', 'None')
                if sample_rate == 'None' or sample_rate is None:
                    # 如果沒有指定取樣率，嘗試從第一個音訊檔讀取
                    first_audio = results_all[0] if results_all else None
                    if first_audio and first_audio.audio_filename:
                        first_audio_path = os.path.join(folder_path, first_audio.audio_filename)
                        if os.path.exists(first_audio_path):
                            import soundfile as sf
                            info = sf.info(first_audio_path)
                            sample_rate = info.samplerate
                        else:
                            sample_rate = 44100  # 預設值
                    else:
                        sample_rate = 44100
                else:
                    sample_rate = float(sample_rate)
                
                f_min = float(params.get('f_min', 0))
                f_max = float(params.get('f_max', 0))
                
                # 如果 f_max 為 0，使用 Nyquist 頻率
                if f_max <= 0:
                    f_max = sample_rate / 2
            except:
                sample_rate = 44100
                f_min = 0
                f_max = sample_rate / 2
            
            # 查詢所有框選標記
            bbox_count = 0
            for i, res in enumerate(results_all):
                annotations = BBoxAnnotation.query.filter_by(result_id=res.id).all()
                
                if annotations:
                    fname = res.spectrogram_training_filename
                    csv_filename = f"images/{fname}"
                    segment_start_time = i * hop_length
                    
                    for bbox in annotations:
                        # 計算時間軸（秒）
                        time_start = segment_start_time + (bbox.x * segment_duration)
                        time_end = segment_start_time + ((bbox.x + bbox.width) * segment_duration)
                        
                        # 計算頻率軸（Hz）
                        # 注意：Y 軸需要反轉（Y=0 是圖片頂部=高頻，Y=1 是圖片底部=低頻）
                        freq_max_bbox = f_min + (1 - bbox.y) * (f_max - f_min)
                        freq_min_bbox = f_min + (1 - (bbox.y + bbox.height)) * (f_max - f_min)
                        
                        bbox_csv_writer.writerow([
                            csv_filename,
                            i,
                            bbox.label,
                            f"{time_start:.3f}",
                            f"{time_end:.3f}",
                            f"{freq_min_bbox:.2f}",
                            f"{freq_max_bbox:.2f}"
                        ])
                        bbox_count += 1
            
            # 只有在有框選標記時才加入檔案
            if bbox_count > 0:
                zf.writestr('bbox_annotations.csv', bbox_csv_buffer.getvalue())

        memory_file.seek(0)
        download_filename = f"dataset_{upload.id}_{secure_filename(upload.file_name)}.zip"
        return send_file(memory_file, mimetype='application/zip', as_attachment=True, download_name=download_filename)

    except Exception as e:
        print(f"打包 ZIP 時發生錯誤: {e}")
        return f"打包失敗: {e}", 500

# --- 上傳與背景任務路由 ---

@main_bp.route('/upload', methods=['POST'])
def upload():
    """Web 上傳介面 - 支援多檔案上傳"""
    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return redirect(url_for('main.index'))
    
    try:
        params_dict = {
            'spec_type': request.form['spec_type'],
            'segment_duration': float(request.form['segment_duration']),
            'overlap': float(request.form['overlap']),
            'sample_rate': request.form.get('sample_rate', 'None'),
            'channels': request.form.get('channels', 'mono'),
            # 頻譜圖進階參數
            'n_fft': int(request.form.get('n_fft', 1024)),
            'window_overlap': float(request.form.get('window_overlap', 50)),  # 改為百分比
            'window_type': request.form.get('window_type', 'hann'),
            'n_mels': int(request.form.get('n_mels', 128)),
            'f_min': float(request.form.get('f_min', 0)),
            'f_max': float(request.form.get('f_max', 0)),  # 0 表示使用 Nyquist 頻率
            'power': float(request.form.get('power', 2.0))
        }
    except: 
        return "參數錯誤", 400

    params_json = json.dumps(params_dict)
    default_point = PointInfo.query.first()
    point_id = default_point.id if default_point else None
    
    uploaded_ids = []
    
    for file in files:
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            file_ext = os.path.splitext(filename)[1].lower().replace('.', '')
            
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
            
            # 派送 Celery 任務 - 自動排隊處理
            celery.send_task('app.tasks.process_audio_task', args=[upload_id])
            uploaded_ids.append(upload_id)
    
    if uploaded_ids:
        # 導向歷史頁面，顯示第一筆新上傳的檔案
        return redirect(url_for('main.history', new_upload_id=uploaded_ids[0]))
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
    """啟動自動標記任務 - 使用已訓練的模型"""
    upload_id = request.form.get('upload_id')
    run_id = request.form.get('run_id')
    
    if not upload_id or not run_id:
        return "缺少必要參數", 400
    
    # 從 TrainingRun 取得模型資訊
    training_run = TrainingRun.query.get(run_id)
    if not training_run or training_run.status != 'SUCCESS':
        return "找不到有效的訓練模型", 404
    
    # 取得模型路徑
    model_path = os.path.join(
        current_app.root_path, 'static', 
        training_run.results_path, 'weights', 'best.pt'
    )
    
    if not os.path.exists(model_path):
        return f"模型檔案不存在: {model_path}", 404
    
    # 從訓練參數取得模型類型
    params = training_run.get_params() if hasattr(training_run, 'get_params') else {}
    if isinstance(params, str):
        params = json.loads(params) if params else {}
    model_type = params.get('model_type', 'yolov8n-cls')
    
    # 從 metrics 取得類別資訊
    metrics = training_run.get_metrics() if hasattr(training_run, 'get_metrics') else {}
    if isinstance(metrics, str):
        metrics = json.loads(metrics) if metrics else {}
    
    classes_list = []
    if 'per_class_list' in metrics:
        classes_list = [item['name'] for item in metrics['per_class_list']]
    
    celery.send_task('app.tasks.auto_label_task_v2', args=[
        int(upload_id), 
        model_path, 
        model_type, 
        classes_list
    ])
    return redirect(url_for('main.labeling_page', upload_id=upload_id))

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
    
    return render_template(
        'training_report.html', 
        run=run, 
        images=report_images, 
        metrics=metrics,
        params=params,
        model_type=model_type,
        model_display_name=model_display_names.get(model_type, model_type),
        is_yolo=is_yolo
    )

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

# --- 進階框選標記 ---

@main_bp.route('/label-advanced/<int:upload_id>')
def label_advanced_page(upload_id):
    """渲染進階框選標記頁面 - 一張圖一頁"""
    upload_record = AudioInfo.query.get_or_404(upload_id)
    index = request.args.get('index', 0, type=int)

    results_all = Result.query.filter_by(upload_id=upload_id).order_by(Result.id.asc()).all()
    total = len(results_all)

    if total == 0:
        return "此分析尚無頻譜圖結果。", 404

    # 確保 index 在合法範圍
    index = max(0, min(index, total - 1))
    current_result = results_all[index]

    return render_template(
        'label_advanced.html',
        upload=upload_record,
        result=current_result,
        current_index=index,
        total=total
    )

@main_bp.route('/api/bbox/<int:result_id>', methods=['GET'])
def get_bbox_annotations(result_id):
    """取得一張頻譜圖的所有框選標記"""
    Result.query.get_or_404(result_id)
    annotations = BBoxAnnotation.query.filter_by(result_id=result_id).all()
    return jsonify([
        {
            'id': a.id,
            'label': a.label,
            'x': a.x,
            'y': a.y,
            'width': a.width,
            'height': a.height
        }
        for a in annotations
    ])

@main_bp.route('/api/bbox/<int:result_id>', methods=['POST'])
def save_bbox_annotations(result_id):
    """儲存一張頻譜圖的所有框選標記 (先刪後寫)"""
    Result.query.get_or_404(result_id)
    data = request.get_json()
    boxes = data.get('boxes', [])

    # 刪除舊的標記
    BBoxAnnotation.query.filter_by(result_id=result_id).delete()

    # 寫入新的標記
    for box in boxes:
        annotation = BBoxAnnotation(
            result_id=result_id,
            label=box['label'],
            x=float(box['x']),
            y=float(box['y']),
            width=float(box['width']),
            height=float(box['height'])
        )
        db.session.add(annotation)

    db.session.commit()
    return jsonify({'success': True, 'count': len(boxes)})