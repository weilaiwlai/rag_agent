import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
import logging
 
# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
 
# 导入你的RAG系统组件
from rag.vector_db_manager import VectorDatabaseManager
from utils.config_handler import rag_conf
 
# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
 
class DataIngestor:
    """
    数据摄取器：将数据集向量化并存入向量数据库
    """
    def __init__(self, collection_name: str = None):
        """
        初始化数据摄取器
        
        Args:
            collection_name: 要存入的集合名称
        """
        self.db_manager = VectorDatabaseManager(
            milvus_host=rag_conf["MILVUS_HOST"],
            milvus_port=rag_conf["MILVUS_PORT"],
            collection_name=collection_name or rag_conf["COLLECTION_NAME"],
        )
        # self.processor = VectorDatabaseManager.add_documents_to_db
        self.collection_name = collection_name or rag_conf["COLLECTION_NAME"]
        
    def ingest_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        将文档列表向量化并存入向量数据库
        
        Args:
            documents: 文档列表，每个文档应包含 'text' 和可选的 'metadata'
            
        Returns:
            是否成功存入
        """
        try:
            # 准备文档对象
            doc_objects = []
            for idx, doc_data in enumerate(documents):
                # 构建文档对象
                content = doc_data.get('text', '')
                metadata = doc_data.get('metadata', {})
                
                # 确保文档ID唯一
                doc_id = metadata.get('id', f'doc_{idx}_{hash(content[:50])}')
                
                # 创建文档对象
                from langchain_core.documents import Document
                doc_obj = Document(
                    page_content=content,
                    metadata={
                        'id': str(doc_id),
                        'source': metadata.get('source', 'unknown'),
                        'title': metadata.get('title', ''),
                        'created_at': metadata.get('created_at', ''),
                        **metadata
                    }
                )
                doc_objects.append(doc_obj)
            
            # 使用DocumentProcessor处理文档
            processed_docs = self.db_manager.add_documents_to_db(doc_objects)
            
            logger.info(f"成功存入 {len(doc_objects)} 个文档到集合 '{self.collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"文档存入失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def ingest_beir_dataset(self, dataset_name: str) -> bool:
        """
        将BEIR数据集的语料库向量化并存入向量数据库
        
        Args:
            dataset_name: BEIR数据集名称
            
        Returns:
            是否成功存入
        """
        try:
            from beir import util
            from beir.datasets.data_loader import GenericDataLoader
            
            # 下载BEIR数据集
            logger.info(f"正在下载BEIR数据集: {dataset_name}")
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
            data_path = util.download_and_unzip(url, "./datasets")
            
            # 加载数据
            logger.info("正在加载数据集语料库...")
            corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
            
            logger.info(f"数据集加载完成: {len(corpus)} 个文档")
            
            # 转换语料库为文档列表
            documents = []
            for doc_id, doc_data in corpus.items():
                # 组合标题和文本
                title = doc_data.get('title', '')
                text = doc_data.get('text', '')
                combined_text = f"{title}. {text}".strip()
                
                document = {
                    'text': combined_text,
                    'metadata': {
                        'id': doc_id,
                        'source': f'beir_{dataset_name}',
                        'title': title,
                        'dataset': dataset_name
                    }
                }
                documents.append(document)
            
            logger.info(f"准备向量化 {len(documents)} 个文档")
            
            # 存入数据库
            success = self.ingest_documents(documents)
            
            if success:
                logger.info(f"BEIR数据集 {dataset_name} 成功存入集合 '{self.collection_name}'")
            
            return success
            
        except Exception as e:
            logger.error(f"BEIR数据集存入失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def ingest_from_json(self, json_file_path: str) -> bool:
        """
        从JSON文件读取数据并存入向量数据库
        
        Args:
            json_file_path: JSON文件路径
            
        Returns:
            是否成功存入
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 支持两种格式：
            # 1. 列表格式：[{"text": "...", "metadata": {...}}, ...]
            # 2. 字典格式：{"documents": [{"text": "...", "metadata": {...}}, ...]}
            if isinstance(data, list):
                documents = data
            elif isinstance(data, dict) and 'documents' in data:
                documents = data['documents']
            else:
                raise ValueError("JSON文件格式不正确，应为文档列表或包含'documents'键的字典")
            
            logger.info(f"从JSON文件读取到 {len(documents)} 个文档")
            
            # 存入数据库
            success = self.ingest_documents(documents)
            
            if success:
                logger.info(f"从 {json_file_path} 读取的数据成功存入集合 '{self.collection_name}'")
            
            return success
            
        except Exception as e:
            logger.error(f"JSON文件存入失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def check_collection_status(self) -> Dict[str, Any]:
        """
        检查集合状态
        
        Returns:
            集合状态信息
        """
        try:
            # 检查集合是否存在
            exists = self.db_manager.collection_exists(self.collection_name)
            
            if exists:
                # 获取集合中的文档数量
                # 注意：这里需要根据你的VectorDatabaseManager实现来获取文档数量
                # 假设有一个方法可以获取文档数量
                try:
                    doc_count = self.db_manager.get_collection_doc_count(self.collection_name)
                except AttributeError:
                    # 如果没有这个方法，我们暂时返回未知
                    doc_count = "unknown"
            else:
                doc_count = 0
            
            status = {
                "collection_name": self.collection_name,
                "exists": exists,
                "document_count": doc_count,
                "milvus_connected": self.db_manager.health_check()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"检查集合状态失败: {e}")
            return {
                "collection_name": self.collection_name,
                "error": str(e)
            }
 
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='将数据集向量化并存入向量数据库')
    parser.add_argument('--source-type', choices=['beir', 'json'], default='beir',
                       help='数据源类型')
    parser.add_argument('--source', type=str, default='scifact',
                       help='数据源（BEIR数据集名称或JSON文件路径）')
    parser.add_argument('--collection', type=str, default=None,
                       help='目标集合名称（默认：配置文件中的默认值）')
    parser.add_argument('--check-status', action='store_true',
                       help='仅检查集合状态而不进行数据存入')
    
    args = parser.parse_args()
    
    # 创建数据摄取器
    ingestor = DataIngestor(collection_name=args.collection)
    
    if args.check_status:
        # 仅检查状态
        status = ingestor.check_collection_status()
        print("集合状态:")
        for key, value in status.items():
            print(f"  {key}: {value}")
        return
    
    print(f"开始数据摄取...")
    print(f"数据源类型: {args.source_type}")
    print(f"数据源: {args.source}")
    print(f"目标集合: {args.collection or rag_conf['COLLECTION_NAME']}")
    
    success = False
    
    if args.source_type == 'beir':
        success = ingestor.ingest_beir_dataset(args.source)
    elif args.source_type == 'json':
        if not os.path.exists(args.source):
            logger.error(f"JSON文件不存在: {args.source}")
            return
        success = ingestor.ingest_from_json(args.source)
    
    if success:
        print("\n数据摄取完成！")
        
        # 检查最终状态
        final_status = ingestor.check_collection_status()
        print("\n最终集合状态:")
        for key, value in final_status.items():
            print(f"  {key}: {value}")
    else:
        print("\n数据摄取失败！")
 
if __name__ == "__main__":
    main()