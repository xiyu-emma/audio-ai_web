import os
import json
import shutil
from flask import request, redirect, url_for, current_app, jsonify
from werkzeug.utils import secure_filename
import pandas as pd
import librosa
from ..main_router import main_bp

# 引入相關模型和實例
from ..models import AudioInfo, PointInfo, BBoxAnnotation, Result, CetaceanInfo
from .. import db, celery

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
    except Exception as e:
        print(f"上傳參數解析錯誤: {e}")
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

@main_bp.route('/api/import_excel', methods=['POST'])
def import_excel():
    """匯入 Excel 標記資料"""
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': '沒有選擇檔案'}), 400
    
    default_label = request.form.get('default_label', 'whale')
    
    LABEL_TO_EVENT_TYPE = {
        'whale': 1, 'unknown': 0, 'whale_unknown': 10, 'whale_upsweep': 11, 'whale_downsweep': 12,
        'whale_concave': 13, 'whale_convex': 14, 'whale_sine': 15, 'whale_click': 16,
        'whale_burst': 17, 'whale_constant': 18, 'noise': 90, 'ship': 91, 'piling': 92
    }
    default_event_type = LABEL_TO_EVENT_TYPE.get(default_label, 0)
    
    success_count = 0
    total_labels_inserted = 0
    errors = []
    
    import re
    for file in files:
        if file.filename.endswith('.csv'):
            try:
                df = pd.read_csv(file)
            except Exception as e:
                errors.append(f"處理 CSV 發生錯誤: {e}")
                continue
        elif file.filename.endswith(('.xlsx', '.xls')):
            try:
                df = pd.read_excel(file)
            except Exception as e:
                errors.append(f"處理 Excel 發生錯誤: {e}")
                continue
        else:
            errors.append(f"檔案 {file.filename} 不是支援的格式")
            continue
            
        if 'filename' in df.columns:
                NAME_TO_EVENT_TYPE = {
                    "Unlabeled": 0, "Whale": 1, "Unknown Vocalization": 10, "Upsweep": 11,
                    "Downsweep": 12, "Concave": 13, "Convex": 14, "Sine": 15, "Click": 16,
                    "Burst": 17, "Constant": 18, "Noise": 90, "Ship": 91, "Piling": 92
                }
                EVENT_TYPE_TO_STR = {
                    0: "unknown", 1: "whale", 10: "whale_unknown", 11: "whale_upsweep",
                    12: "whale_downsweep", 13: "whale_concave", 14: "whale_convex",
                    15: "whale_sine", 16: "whale_click", 17: "whale_burst", 18: "whale_constant",
                    90: "noise", 91: "ship", 92: "piling"
                }
                
                label_col = 'label_name' if 'label_name' in df.columns else 'label' if 'label' in df.columns else None
                if not label_col:
                    errors.append(f"檔案 {file.filename} 缺少 label 欄位")
                    continue
                    
                added = 0
                for idx, row in df.iterrows():
                    csv_filename = str(row['filename']).strip()
                    new_label = str(row[label_col]).strip()
                    original_audio = str(row.get('original_audio', '')).strip()
                    
                    new_event_type = NAME_TO_EVENT_TYPE.get(new_label, 0)
                    new_bbox_label = EVENT_TYPE_TO_STR.get(new_event_type, "unknown")
                    
                    targets = []
                    if original_audio and original_audio.lower() != 'nan':
                        audios = AudioInfo.query.filter_by(file_name=original_audio).all()
                        for audio in audios:
                            r = Result.query.filter_by(upload_id=audio.id, spectrogram_training_filename=csv_filename).first()
                            if r:
                                targets.append((r, audio.id, csv_filename))
                    else:
                        match = re.search(r'upload_(\d+)_(.+)', csv_filename)
                        if match:
                            u_id = match.group(1)
                            fname = match.group(2)
                            r = Result.query.filter_by(upload_id=u_id, spectrogram_training_filename=fname).first()
                            if r:
                                targets.append((r, u_id, fname))
                    
                    for r, u_id, fname in targets:
                        r_idx_match = re.search(r'_spec_training_(\d+)\.', fname)
                        if not r_idx_match:
                            r_idx_match = re.search(r'^(\d+)_', fname)
                        
                        if r_idx_match:
                            r_idx = int(r_idx_match.group(1))
                            cet = CetaceanInfo.query.filter_by(audio_id=u_id).order_by(CetaceanInfo.id.asc()).offset(r_idx).first()
                            if cet:
                                cet.event_type = new_event_type
                                cet.detect_type = 0
                                
                                existing_bbox = BBoxAnnotation.query.filter_by(result_id=r.id).first()
                                if existing_bbox:
                                    existing_bbox.label = new_bbox_label
                                else:
                                    new_box = BBoxAnnotation(
                                        result_id=r.id,
                                        label=new_bbox_label,
                                        x=0.0, y=0.0, width=1.0, height=1.0
                                    )
                                    db.session.add(new_box)
                                added += 1
                db.session.commit()
                total_labels_inserted += added
                success_count += 1
        else:
            errors.append(f"檔案 {file.filename} 尚不符合支援的格式")
            continue
            
    return jsonify({
        'success': True,
        'success_count': success_count,
        'total_labels': total_labels_inserted,
        'errors': errors
    })
