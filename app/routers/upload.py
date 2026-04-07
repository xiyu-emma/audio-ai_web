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
    
    for file in files:
        if not file.filename.endswith('.xlsx'):
            continue
            
        filename = secure_filename(file.filename)
        base_filename = os.path.splitext(filename)[0]
        
        # 尋找對應的音檔 - 支援多個同名且未標記的音檔
        audios = AudioInfo.query.filter(AudioInfo.file_name.like(f"{base_filename}%")).all()
        if not audios:
            errors.append(f"找不到檔名為 {base_filename} 相關的音檔記錄")
            continue
            
        unlabeled_audios = []
        for a in audios:
            has_label = BBoxAnnotation.query.join(Result).filter(Result.upload_id == a.id).first() is not None
            if not has_label:
                unlabeled_audios.append(a)
                
        if not unlabeled_audios:
            errors.append(f"檔名 {base_filename} 的相關音檔皆已標記過，不進行覆蓋")
            continue
            
        try:
            df = pd.read_excel(file, header=None)
            if len(df) <= 2:
                errors.append(f"檔案 {filename} 沒有資料")
                continue
                
            for audio in unlabeled_audios:
                try:
                    params = audio.get_params()
                    segment_duration = float(params.get('segment_duration', 2.0))
                    overlap_ratio = float(params.get('overlap', 50)) / 100.0
                    step_sec = segment_duration * (1 - overlap_ratio)
            
                    results = Result.query.filter_by(upload_id=audio.id).order_by(Result.id.asc()).all()
                    cetaceans = CetaceanInfo.query.filter_by(audio_id=audio.id).order_by(CetaceanInfo.id.asc()).all()
                    if not results:
                        errors.append(f"音檔 {base_filename} 尚未產生頻譜圖")
                        continue
            
                    result_mapping = {}
                    for res in results:
                        try:
                            r_idx = int(res.spectrogram_filename.split('_')[-1].split('.')[0])
                            result_mapping[r_idx] = res.id
                        except:
                            pass
            
                    result_id_to_cetacean = {}
                    if len(results) == len(cetaceans):
                        for r, c in zip(results, cetaceans):
                            result_id_to_cetacean[r.id] = c
            
                    sr = audio.fs if audio.fs else 24000
                    spec_f_max = float(params.get('f_max', 0))
                    if spec_f_max <= 0:
                        spec_f_max = sr / 2.0
                    spec_f_min = float(params.get('f_min', 0))
            
                    spec_type = params.get('spec_type', 'mel')
                    if spec_type in ['mel', 'yamnet_log_mel']:
                        spec_f_min = 125.0
                        spec_f_max = 7500.0
                
                    f_range = spec_f_max - spec_f_min
                    if f_range <= 0: f_range = sr / 2.0
            
                    labels_inserted = 0
                    labeled_result_ids = set()
            
                    for index, row in df.iloc[2:].iterrows():
                        try:
                            start_m, start_s = pd.to_numeric(row[0], errors='coerce'), pd.to_numeric(row[1], errors='coerce')
                            end_m, end_s = pd.to_numeric(row[2], errors='coerce'), pd.to_numeric(row[3], errors='coerce')
                            start_f, end_f = pd.to_numeric(row[4], errors='coerce'), pd.to_numeric(row[5], errors='coerce')
                            max_f, min_f = pd.to_numeric(row[6], errors='coerce'), pd.to_numeric(row[7], errors='coerce')
                    
                            if pd.isna(start_m) or pd.isna(start_s): continue
                    
                            start_time_sec = float(start_m) * 60 + float(start_s)
                            end_time_sec = float(end_m) * 60 + float(end_s)
                            if start_time_sec >= end_time_sec: continue
                    
                            if pd.isna(max_f) or pd.isna(min_f):
                                max_f_label = max(start_f, end_f) if not pd.isna(start_f) else (spec_f_max)
                                min_f_label = min(start_f, end_f) if not pd.isna(start_f) else (spec_f_min)
                            else:
                                max_f_label = float(max_f)
                                min_f_label = float(min_f)
                    
                            if max_f_label < min_f_label:
                                max_f_label, min_f_label = min_f_label, max_f_label
                        
                            if spec_type in ['mel', 'yamnet_log_mel']:
                                mel_max_f_label = librosa.hz_to_mel(max_f_label)
                                mel_min_f_label = librosa.hz_to_mel(min_f_label)
                                mel_spec_max = librosa.hz_to_mel(spec_f_max)
                                mel_spec_min = librosa.hz_to_mel(spec_f_min)
                        
                                m_range = mel_spec_max - mel_spec_min
                                if m_range <= 0: m_range = 1.0
                        
                                y_percent = (mel_spec_max - mel_max_f_label) / m_range
                                h_percent = (mel_max_f_label - mel_min_f_label) / m_range
                            else:
                                y_percent = (spec_f_max - max_f_label) / f_range
                                h_percent = (max_f_label - min_f_label) / f_range
                    
                            if y_percent < 0:
                                h_percent += y_percent
                                y_percent = 0
                            if y_percent > 1: y_percent = 1
                            if h_percent > 1: h_percent = 1
                    
                            start_idx = max(0, int(start_time_sec // step_sec) - 1)
                            end_idx = int(end_time_sec // step_sec) + 1
                    
                            for i in range(start_idx, end_idx + 1):
                                if i not in result_mapping: continue
                        
                                seg_start = i * step_sec
                                seg_end = seg_start + segment_duration
                        
                                overlap_start = max(start_time_sec, seg_start)
                                overlap_end = min(end_time_sec, seg_end)
                        
                                if overlap_start < overlap_end:
                                    labeled_result_ids.add(result_mapping[i])
                                    x_percent = (overlap_start - seg_start) / segment_duration
                                    w_percent = (overlap_end - overlap_start) / segment_duration
                            
                                    bbox = BBoxAnnotation(
                                        result_id=result_mapping[i],
                                        label=default_label,
                                        x=float(x_percent),
                                        y=float(y_percent),
                                        width=float(w_percent),
                                        height=float(h_percent)
                                    )
                                    db.session.add(bbox)
                                    labels_inserted += 1
                            
                                    c_info = result_id_to_cetacean.get(result_mapping[i])
                                    if c_info:
                                        c_info.event_type = default_event_type
                                        c_info.detect_type = 0
                        except Exception as e:
                            print(f"解析 Excel 第 {index} 列錯誤: {e}")
                            continue
            
                    # 將未在 Excel 中標記的片段設為環境噪音 (90)
                    for res in results:
                        if res.id not in labeled_result_ids:
                            c_info = result_id_to_cetacean.get(res.id)
                            if c_info:
                                c_info.event_type = 90
                                c_info.detect_type = 0
                    
                            bbox = BBoxAnnotation(
                                result_id=res.id,
                                label='noise',
                                x=0.0,
                                y=0.0,
                                width=1.0,
                                height=1.0
                            )
                            db.session.add(bbox)
                            labels_inserted += 1

                    db.session.commit()
                    total_labels_inserted += labels_inserted
                
                except Exception as inner_e:
                    errors.append(f"處理音檔 {base_filename} (ID: {audio.id}) 時發生錯誤: {str(inner_e)}")
                    continue
            
            success_count += 1
            
        except Exception as e:
            errors.append(f"處理 {filename} 時發生錯誤: {str(e)}")
            
    return jsonify({
        'success': True,
        'success_count': success_count,
        'total_labels': total_labels_inserted,
        'errors': errors
    })
