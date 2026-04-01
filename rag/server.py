import os
import sys
from pathlib import Path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from utils.config_handler import rag_conf

from api_integration import register_vector_routes

# 添加当前目录到 python path 以便导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)  # 启用跨域支持，允许前端访问
    
    # 注册向量数据库路由
    register_vector_routes(app)
    
    @app.route('/')
    def index():
        return "RAG Backend Service is Running!"

    return app

if __name__ == '__main__':
    app = create_app()
    host = rag_conf["FLASK_HOST"]
    port = int(rag_conf["FLASK_PORT"])
    debug = bool(rag_conf["FLASK_DEBUG"])
    
    print("启动 RAG 后端服务...")
    print(f"API 地址: http://{host if host != '0.0.0.0' else 'localhost'}:{port}/api/vector/")
    app.run(host=host, port=port, debug=debug)