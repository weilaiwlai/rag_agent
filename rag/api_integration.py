"""
API集成模块
将向量数据库功能集成到Flask应用中
"""

from flask import Blueprint, request, jsonify
import os
import logging
from typing import Dict, Any, List
from werkzeug.utils import secure_filename
from pathlib import Path

from vector_db_manager import VectorDatabaseManager
from vector_retriever import VectorRetriever
from document_loader import DocumentLoader

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from utils.config_handler import rag_conf

# 创建蓝图
vector_bp = Blueprint('vector', __name__, url_prefix='/api/vector')

# 全局变量存储向量系统实例
vector_manager: VectorDatabaseManager = None
vector_retriever: VectorRetriever = None

# 临时上传目录
UPLOAD_FOLDER = '/tmp/vector_uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_vector_system(
    milvus_host: str = rag_conf["MILVUS_HOST"],
    milvus_port: str = rag_conf["MILVUS_PORT"],
    embedding_model: str = rag_conf["embedding_model_name"],
    dashscope_api_key: str = rag_conf["dashscope_api_key"]
):
    """初始化向量系统"""
    global vector_manager, vector_retriever
    
    try:
        vector_manager = VectorDatabaseManager(
            milvus_host=milvus_host,
            milvus_port=milvus_port,
            embedding_model=embedding_model,
            dashscope_api_key=dashscope_api_key
        )
        vector_retriever = VectorRetriever(vector_manager)
        logger.info(f"向量系统初始化成功，连接到 Milvus at {milvus_host}:{milvus_port}")
        return True
    except Exception as e:
        logger.error(f"向量系统初始化失败: {str(e)}")
        return False


@vector_bp.route('/upload_document', methods=['POST'])
def upload_document():
    """上传并处理文档"""
    global vector_manager
    
    if not vector_manager:
        return jsonify({
            'success': False,
            'message': '向量系统未初始化'
        }), 400
    
    try:
        data = request.get_json()
        if not data or 'file_path' not in data or 'collection_name' not in data:
            return jsonify({
                'success': False,
                'message': '请提供 file_path 和 collection_name 参数'
            }), 400
        
        file_path = data['file_path']
        collection_name = data['collection_name']
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return jsonify({
                'success': False,
                'message': f'文件不存在: {file_path}'
            }), 400
        
        # 处理文档
        try:
            # 调用更新后的process_file方法，它返回详细的状态信息
            result = vector_manager.process_file(file_path, collection_name)
            
            # 根据返回的状态进行处理
            if result['status'] == 'already_exists':
                # 文件已存在，返回成功状态
                db_info = vector_manager.get_database_info(collection_name)
                return jsonify({
                    'success': True,
                    'message': result['message'],
                    'file_status': 'already_exists',
                    'database_info': db_info
                })
            elif result['status'] == 'processed_new':
                # 新文件已处理，返回成功状态
                db_info = vector_manager.get_database_info(collection_name)
                return jsonify({
                    'success': True,
                    'message': result['message'],
                    'file_status': 'processed_new',
                    'database_info': db_info
                })
            elif result['status'] == 'load_failed':
                # 文档加载失败
                return jsonify({
                    'success': False,
                    'message': result['message'],
                    'file_status': 'load_failed'
                }), 500
            else:  # exception 或其他错误情况
                return jsonify({
                    'success': False,
                    'message': result['message'],
                    'file_status': result['status']
                }), 500
        except Exception as e:
            logger.error(f"文档处理异常: {str(e)}")
            return jsonify({
                'success': False,
                'message': f'文档处理异常: {str(e)}',
                'file_status': 'exception_occurred'
            }), 500
            
    except Exception as e:
        logger.error(f"文档上传API错误: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'文档处理失败: {str(e)}'
        }), 500


@vector_bp.route('/upload_file', methods=['POST'])
def upload_file():
    """上传文件流处理"""
    global vector_manager
    
    if not vector_manager:
        return jsonify({'success': False, 'message': '向量系统未初始化'}), 400

    if 'file' not in request.files:
        return jsonify({'success': False, 'message': '未找到文件部分'}), 400
        
    file = request.files['file']
    collection_name = request.form.get('collection_name', 'agent_rag')
    
    if file.filename == '':
        return jsonify({'success': False, 'message': '未选择文件'}), 400
        
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        try:
            # 调用更新后的process_file方法，它返回详细的状态信息
            result = vector_manager.process_file(file_path, collection_name)
            
            # 根据返回的状态进行处理
            if result['status'] == 'already_exists':
                # 文件已存在，返回成功状态
                db_info = vector_manager.get_database_info(collection_name)
                return jsonify({
                    'success': True,
                    'message': result['message'],
                    'file_status': 'already_exists',
                    'database_info': db_info
                })
            elif result['status'] == 'processed_new':
                # 新文件已处理，返回成功状态
                db_info = vector_manager.get_database_info(collection_name)
                return jsonify({
                    'success': True,
                    'message': f'文件上传并处理成功: {filename}',
                    'file_status': 'processed_new',
                    'database_info': db_info
                })
            elif result['status'] == 'load_failed':
                # 文档加载失败
                return jsonify({
                    'success': False,
                    'message': result['message'],
                    'file_status': 'load_failed'
                }), 500
            else:  # exception 或其他错误情况
                return jsonify({
                    'success': False,
                    'message': result['message'],
                    'file_status': result['status']
                }), 500
        except Exception as e:
            logger.error(f"文件处理异常: {str(e)}")
            return jsonify({
                'success': False,
                'message': str(e),
                'file_status': 'exception_occurred'
            }), 500
        finally:
            # 可选：处理完后删除临时文件，或者保留
            os.remove(file_path)
            pass

    return jsonify({'success': False, 'message': '上传失败'}), 500


@vector_bp.route('/query', methods=['POST'])
def query_documents():
    """查询文档"""
    global vector_retriever
    
    if not vector_retriever:
        return jsonify({
            'success': False,
            'message': '向量系统未初始化'
        }), 400
    
    try:
        data = request.get_json()
        if not data or 'question' not in data or 'collection_name' not in data:
            return jsonify({
                'success': False,
                'message': '请提供 question 和 collection_name 参数'
            }), 400
        
        question = data['question']
        collection_name = data['collection_name']
        k = data.get('k', 5)  # 返回结果数量
        
        # 执行查询
        result = vector_retriever.answer_question(question, k=k, collection_name=collection_name)
        
        return jsonify({
            'success': True,
            'question': question,
            'answer': result.answer,
            'confidence': result.confidence,
            'question_type': result.question_type,
            'sources': [
                {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': score
                }
                for doc, score in zip(result.source_documents, result.scores)
            ]
        })
        
    except Exception as e:
        logger.error(f"查询API错误: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'查询失败: {str(e)}'
        }), 500


@vector_bp.route('/search', methods=['POST'])
def search_similar():
    """相似性搜索"""
    global vector_retriever
    
    if not vector_retriever:
        return jsonify({
            'success': False,
            'message': '向量系统未初始化'
        }), 400
    
    try:
        data = request.get_json()
        if not data or 'query' not in data or 'collection_name' not in data:
            return jsonify({
                'success': False,
                'message': '请提供 query 和 collection_name 参数'
            }), 400
        
        query = data['query']
        collection_name = data['collection_name']
        k = data.get('k', 5)
        
        # 执行相似性搜索
        results = vector_retriever.search_similar_content(query, k=k, collection_name=collection_name)
        
        return jsonify({
            'success': True,
            'query': query,
            'results': [
                {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': score
                }
                for doc, score in results
            ]
        })
        
    except Exception as e:
        logger.error(f"搜索API错误: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'搜索失败: {str(e)}'
        }), 500


@vector_bp.route('/collection_info', methods=['GET'])
def get_collection_info():
    """获取集合信息"""
    global vector_manager
    
    if not vector_manager:
        return jsonify({
            'success': False,
            'message': '向量系统未初始化'
        }), 400
    
    collection_name = request.args.get('collection_name')
    if not collection_name:
        return jsonify({
            'success': False,
            'message': '请提供 collection_name 参数'
        }), 400

    try:
        db_info = vector_manager.get_database_info(collection_name)
        return jsonify({
            'success': True,
            'database_info': db_info
        })
        
    except Exception as e:
        logger.error(f"数据库信息API错误: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'获取数据库信息失败: {str(e)}'
        }), 500


@vector_bp.route('/clear_collection', methods=['POST'])
def clear_collection():
    """清空集合"""
    global vector_manager
    
    if not vector_manager:
        return jsonify({
            'success': False,
            'message': '向量系统未初始化'
        }), 400
    
    data = request.get_json()
    if not data or 'collection_name' not in data:
        return jsonify({
            'success': False,
            'message': '请提供 collection_name 参数'
        }), 400

    collection_name = data['collection_name']

    try:
        vector_manager.clear_database(collection_name)
        return jsonify({
            'success': True,
            'message': f"集合 '{collection_name}' 已清空"
        })
        
    except Exception as e:
        logger.error(f"清空数据库API错误: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'清空数据库失败: {str(e)}'
        }), 500


# 错误处理
@vector_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'message': '接口不存在'
    }), 404


@vector_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'message': '服务器内部错误'
    }), 500


def register_vector_routes(app):
    """注册向量数据库路由到Flask应用"""
    app.register_blueprint(vector_bp)
    
    # 自动初始化向量系统
    with app.app_context():
        init_vector_system()
    
    logger.info("向量数据库API路由已注册")