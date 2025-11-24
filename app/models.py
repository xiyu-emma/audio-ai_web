from . import db
from datetime import datetime
from zoneinfo import ZoneInfo # 導入 Python 3.9+ 的標準時區模組
import json

# 定義一次性的時區物件，確保程式碼一致性
TAIPEI_TZ = ZoneInfo("Asia/Taipei")

class Upload(db.Model):
    __tablename__ = 'uploads'
    id = db.Column(db.Integer, primary_key=True)
    original_filename = db.Column(db.String(255), nullable=False)
    
    # --- 最終修正版 ---
    # 使用 lambda 函式，確保每次建立新紀錄時，
    # 都會產生一個明確帶有 "Asia/Taipei" 時區的「時區感知」時間物件。
    upload_timestamp = db.Column(db.DateTime, default=lambda: datetime.now(TAIPEI_TZ))
    
    result_path = db.Column(db.String(255), nullable=False)
    params = db.Column(db.Text, nullable=False)
    
    status = db.Column(db.String(50), default='PENDING', nullable=False)
    progress = db.Column(db.Integer, default=0, nullable=False)
    
    results = db.relationship('Result', backref='upload', lazy=True, cascade="all, delete-orphan")

    def get_params(self):
        try:
            return json.loads(self.params)
        except (json.JSONDecodeError, TypeError):
            return {}

class Result(db.Model):
    __tablename__ = 'results'
    id = db.Column(db.Integer, primary_key=True)
    upload_id = db.Column(db.Integer, db.ForeignKey('uploads.id'), nullable=False)
    audio_filename = db.Column(db.String(255), nullable=True)
    spectrogram_filename = db.Column(db.String(255), nullable=False)
    spectrogram_training_filename = db.Column(db.String(255), nullable=False)
    
    label_id = db.Column(db.Integer, db.ForeignKey('labels.id'), nullable=True)
    label = db.relationship('Label', backref='results')
    
    @property
    def audio_url(self):
        if self.audio_filename:
            return f"{self.upload.result_path}/{self.audio_filename}"
        return None

    @property
    def spectrogram_url(self):
        return f"{self.upload.result_path}/{self.spectrogram_filename}"

    @property
    def spectrogram_training_url(self):
        return f"{self.upload.result_path}/{self.spectrogram_training_filename}"

class Label(db.Model):
    __tablename__ = 'labels'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    description = db.Column(db.String(255), nullable=True)

class TrainingRun(db.Model):
    __tablename__ = 'training_runs'
    id = db.Column(db.Integer, primary_key=True)

    # --- 最終修正版 ---
    # 同上，確保訓練任務的時間戳也是「時區感知」的。
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(TAIPEI_TZ))

    status = db.Column(db.String(50), default='PENDING')
    results_path = db.Column(db.String(255), nullable=True)
    params = db.Column(db.Text, nullable=True) 
    metrics = db.Column(db.Text, nullable=True)
    
    progress = db.Column(db.Integer, default=0, nullable=False)

    def get_params(self):
        if self.params:
            try:
                return json.loads(self.params)
            except (json.JSONDecodeError, TypeError):
                return {}
        return {}

    def get_metrics(self):
        if self.metrics:
            try:
                return json.loads(self.metrics)
            except (json.JSONDecodeError, TypeError):
                return {}
        return {}

