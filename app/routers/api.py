import os
import json
from functools import wraps
from flask import request, jsonify, current_app
from werkzeug.utils import secure_filename
from ..main_router import main_bp

from ..models import AudioInfo, PointInfo
from .. import db, celery

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

@main_bp.route('/api/v1/upload', methods=['POST'])
@require_api_token
def api_upload_audio():
    """洋聲 API: 上傳並觸發分析"""
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'No filename'}), 400
    
    try:
        point_id = int(request.form.get('point_id'))
        if not PointInfo.query.get(point_id): return jsonify({'error': 'Invalid point_id'}), 404
    except Exception as e:
        print(f"point_id 解析錯誤: {e}")
        return jsonify({'error': 'Missing point_id'}), 400

    # 擷取頻譜圖等進階參數
    try:
        params_dict = {
            'spec_type': request.form.get('spec_type', 'yamnet_log_mel'),
            'segment_duration': float(request.form.get('segment_duration', 2.0)),
            'overlap': float(request.form.get('overlap', 50.0)),
            'sample_rate': request.form.get('sample_rate', 'None'),
            'channels': request.form.get('channels', 'mono'),
            'n_fft': int(request.form.get('n_fft', 1024)),
            'window_overlap': float(request.form.get('window_overlap', 50.0)),
            'window_type': request.form.get('window_type', 'hann'),
            'n_mels': int(request.form.get('n_mels', 128)),
            'f_min': float(request.form.get('f_min', 0.0)),
            'f_max': float(request.form.get('f_max', 0.0)),
            'power': float(request.form.get('power', 2.0))
        }
    except Exception as e:
        return jsonify({'error': f'參數格式錯誤: {str(e)}'}), 400

    filename = secure_filename(file.filename)
    file_ext = os.path.splitext(filename)[1].lower().replace('.', '')
    params_json = json.dumps(params_dict)

    new_audio = AudioInfo(
        file_name=filename, file_path="pending", file_type=file_ext,
        point_id=point_id, result_path="pending", params=params_json, status='PENDING'
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
    
    return jsonify({'success': True, 'upload_id': new_audio.id}), 201
