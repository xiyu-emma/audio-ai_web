import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from celery import Celery, Task

# 1. 初始化核心元件
db = SQLAlchemy()
celery = Celery(__name__,
                broker=os.environ.get('CELERY_BROKER_URL', 'redis://redis:6379/0'),
                backend=os.environ.get('CELERY_RESULT_BACKEND', 'redis://redis:6379/0'),
                include=['app.tasks'])

# vvvvvvvvvvvvvv 這裡是本次修改的地方 vvvvvvvvvvvvvv
def init_db_command():
    """
    一個獨立的函式，用於建立資料庫表格。
    """
    print("正在初始化資料庫...")
    db.create_all()
    print("資料庫初始化完成。")

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def create_app():
    """
    建立並設定 Flask app 的工廠函式。
    """
    app = Flask(__name__)

    # --- 2. 載入設定 ---
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'mysql+pymysql://user:password@db/audio_db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # 更新 Celery 的設定
    celery.conf.update(
        broker_url=os.environ.get('CELERY_BROKER_URL'),
        result_backend=os.environ.get('CELERY_RESULT_BACKEND')
    )

    # --- 3. 將元件與 app 綁定 ---
    db.init_app(app)

    # 檔案路徑設定
    app.config['UPLOAD_FOLDER'] = 'static/uploads'
    app.config['RESULT_FOLDER'] = 'static/results'

    # 確保所有資料夾都存在
    with app.app_context():
        os.makedirs(os.path.join(app.root_path, app.config['UPLOAD_FOLDER']), exist_ok=True)
        os.makedirs(os.path.join(app.root_path, app.config['RESULT_FOLDER']), exist_ok=True)
        os.makedirs(os.path.join(app.root_path, 'static', 'training_runs'), exist_ok=True)

    # --- 4. 註冊藍圖 (Blueprints) 與建立資料庫 ---
    with app.app_context():
        from . import main
        app.register_blueprint(main.main_bp)
        
        # vvvvvvvvvvvvvv 這裡是本次修改的地方 vvvvvvvvvvvvvv
        # 我們將 db.create_all() 從這裡移除
        # db.create_all() 
        
        # 註冊我們新的資料庫初始化指令
        @app.cli.command("init-db")
        def init_db_wrapper():
            init_db_command()
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    # --- 5. 讓 Celery 任務能感知到 Flask 的應用程式上下文 ---
    class FlaskTask(Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = FlaskTask

    return app

app = create_app()
