import logging
from typing import List, Dict, Any, Optional
from pymilvus import utility, connections, Collection
from vector_db_manager import VectorDatabaseManager
from utils.config_handler import rag_conf

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KnowledgeBaseViewer:
    """知识库查看器"""
    
    def __init__(self, 
                 milvus_host: str = None,
                 milvus_port: str = None):
        """
        初始化知识库查看器
        
        Args:
            milvus_host: Milvus 服务主机
            milvus_port: Milvus 服务端口
        """
        self.milvus_host = milvus_host or rag_conf["MILVUS_HOST"]
        self.milvus_port = milvus_port or rag_conf["MILVUS_PORT"]
        
        # 连接到Milvus
        self._connect_to_milvus()
    
    def _connect_to_milvus(self):
        """连接到Milvus服务"""
        try:
            logger.info(f"Connecting to Milvus: host={self.milvus_host}, port={self.milvus_port}")
            connections.connect("default", host=self.milvus_host, port=self.milvus_port, timeout=30)
            logger.info(f"成功连接到Milvus: {self.milvus_host}:{self.milvus_port}")
        except Exception as e:
            logger.error(f"连接Milvus失败: {e}")
            raise

    def list_all_collections(self) -> List[str]:
        """
        列出所有集合
        
        Returns:
            集合名称列表
        """
        try:
            collections = utility.list_collections()
            logger.info(f"找到 {len(collections)} 个集合: {collections}")
            return collections
        except Exception as e:
            logger.error(f"获取集合列表失败: {e}")
            return []

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        获取指定集合的统计信息
        
        Args:
            collection_name: 集合名称
            
        Returns:
            集合统计信息字典
        """
        try:
            # 检查集合是否存在
            if not utility.has_collection(collection_name):
                logger.warning(f"集合 '{collection_name}' 不存在")
                return {}

            # 获取集合引用并加载
            collection = Collection(collection_name)
            collection.load()  # 加载集合到内存以进行查询

            # 获取集合统计信息
            stats = {
                "collection_name": collection_name,
                "document_count": collection.num_entities,
                "primary_field": collection.primary_field.name,
                "description": collection.description,
                "is_loaded": not collection.is_empty
            }

            # 获取集合schema信息
            schema_info = {}
            for field in collection.schema.fields:
                schema_info[field.name] = {
                    "dtype": field.dtype.name,
                    "is_primary": field.is_primary,
                    "is_partition_key": field.is_partition_key,
                    "description": field.description
                }
            stats["schema"] = schema_info

            logger.info(f"成功获取集合 '{collection_name}' 的统计信息")
            return stats

        except Exception as e:
            logger.error(f"获取集合统计信息失败: {e}")


    def list_documents_in_collection(self, collection_name: str, limit: int = 100) -> Dict[str, Any]:
        """
        列出集合中的文档内容，按source分组整合
        
        Args:
            collection_name: 集合名称
            limit: 限制返回的文档数量
            
        Returns:
            包含文档信息列表和总片段数的字典
        """
        try:
            # 检查集合是否存在
            if not utility.has_collection(collection_name):
                logger.warning(f"集合 '{collection_name}' 不存在")
                return {"documents": [], "total_count": 0}

            # 获取集合引用并加载
            collection = Collection(collection_name)
            collection.load()  # 加载集合到内存以进行查询

            # 获取总片段数 (不受limit影响)
            total_count = collection.num_entities

            # 获取集合schema以确定字段名称
            schema = collection.schema
            text_field = None
            primary_field = collection.primary_field.name

            # 寻找文本类型的字段作为内容字段
            for field in schema.fields:
                if field.dtype.name in ["VARCHAR", "STRING"] and field.name != primary_field:
                    text_field = field.name
                    break

            # 确定输出字段：所有非向量字段（确保包含source字段）
            output_fields = [field.name for field in schema.fields 
                            if field.dtype.name != "FLOAT_VECTOR"]

            # 查询文档 - 不限制数量以获取所有文档进行正确分组
            res = collection.query(
                expr="",  # 空表达式表示查询所有数据
                output_fields=output_fields,
                limit=10000  
            )

            # 按source分组
            grouped_docs = {}
            for item in res:
                source = item.get('source', None)
                if source not in grouped_docs:
                    grouped_docs[source] = []
                grouped_docs[source].append(item)

            # 构建结果列表
            documents_info = []
            for source, items in grouped_docs.items():
                # 合并内容
                if text_field:
                    combined_content = "\n".join([item.get(text_field, "") for item in items])
                else:
                    combined_content = "\n".join([str(item) for item in items])
                
                # 获取第一个item的信息
                if items:
                    first_item = items[0]
                    # 构建元数据
                    metadata = {k: v for k, v in first_item.items() if k != text_field}
                    metadata['source'] = source  # 确保source在元数据中
                    # 计算总长度
                    full_length = len(combined_content)
                    # 使用第一个item的id
                    doc_id = first_item.get(primary_field, "unknown")
                    doc_info = {
                        "id": doc_id,
                        "content": combined_content,
                        "full_length": full_length,
                        "metadata": metadata,
                        "fragment_count": len(items)  # 添加片段数量
                    }
                    documents_info.append(doc_info)
            
            # 限制返回的文档数量 (按source分组的数量限制)
            if limit > 0:
                documents_info = documents_info[:limit]

            logger.info(f"成功获取集合 '{collection_name}' 中的 {len(documents_info)} 个文档（按source分组），总片段数：{total_count}")
            return {"documents": documents_info, "total_count": total_count}

        except Exception as e:
            logger.error(f"获取集合文档列表失败: {e}")
            import traceback
            traceback.print_exc()
            return {"documents": [], "total_count": 0}
    
    def delete_document_by_id(self, collection_name: str, doc_id: Any) -> bool:
        """
        根据ID删除文档
        
        Args:
            collection_name: 集合名称
            doc_id: 文档ID
            
        Returns:
            删除是否成功
        """
        try:
            # 检查集合是否存在
            if not utility.has_collection(collection_name):
                logger.warning(f"集合 '{collection_name}' 不存在")
                return False
            # 获取集合引用并加载
            collection = Collection(collection_name)
            collection.load()  # 加载集合到内存以进行操作
            # 构建删除表达式
            expr = f"{collection.primary_field.name} == {doc_id}"
            # 执行删除操作
            delete_result = collection.delete(expr)
            
            logger.info(f"成功删除文档 ID {doc_id} 从集合 '{collection_name}'")
            logger.info(f"删除的实体数量: {delete_result.delete_count}")
            
            # 刷新集合以确保删除生效
            collection.flush()
            
            return True
        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def delete_documents_by_expr(self, collection_name: str, expression: str) -> bool:
        """
        根据表达式批量删除文档
        
        Args:
            collection_name: 集合名称
            expression: 删除条件表达式
            
        Returns:
            删除是否成功
        """
        try:
            # 检查集合是否存在
            if not utility.has_collection(collection_name):
                logger.warning(f"集合 '{collection_name}' 不存在")
                return False
            # 获取集合引用并加载
            collection = Collection(collection_name)
            collection.load()  # 加载集合到内存以进行操作
            # 执行删除操作
            delete_result = collection.delete(expression)
            
            logger.info(f"成功执行删除操作，集合 '{collection_name}'")
            logger.info(f"删除的实体数量: {delete_result.delete_count}")
            
            # 刷新集合以确保删除生效
            collection.flush()
            
            return True
        except Exception as e:
            logger.error(f"批量删除文档失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def search_in_collection(self, collection_name: str, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        在指定集合中搜索相似文档
        
        Args:
            collection_name: 集合名称
            query_text: 查询文本
            top_k: 返回的最相似文档数量
            
        Returns:
            搜索结果列表
        """
        try:
            # 使用VectorDatabaseManager进行搜索
            db_manager = VectorDatabaseManager(
                milvus_host=self.milvus_host,
                milvus_port=self.milvus_port,
                collection_name=collection_name
            )
            
            results = db_manager.search(query_text, k=top_k, collection_name=collection_name)
            
            search_results = []
            for doc, score in results:
                result_item = {
                    "content": doc.page_content,  # 显示完整内容，不截断
                    "metadata": dict(doc.metadata),
                    "similarity_score": score
                }
                search_results.append(result_item)
            
            logger.info(f"在集合 '{collection_name}' 中搜索到 {len(search_results)} 个结果")
            return search_results
            
        except Exception as e:
            logger.error(f"搜索集合文档失败: {e}")
            import traceback
            traceback.print_exc()
            return []
    def delete_documents_by_source(self, collection_name: str, source: str) -> bool:
        """
        根据source字段删除文档的所有片段
        
        Args:
            collection_name: 集合名称
            source: 源标识符
            
        Returns:
            删除是否成功
        """
        try:
            # 检查集合是否存在
            if not utility.has_collection(collection_name):
                logger.warning(f"集合 '{collection_name}' 不存在")
                return False
            
            # 获取集合引用并加载
            collection = Collection(collection_name)
            collection.load()  # 加载集合到内存以进行操作
            
            expr = f"source == '{source}'"
            
            # 添加调试日志
            logger.info(f"准备执行删除操作，表达式: {expr}")
            logger.info(f"原始source值: {source}")
            
            # 执行删除操作
            delete_result = collection.delete(expr)
            
            logger.info(f"成功删除集合 '{collection_name}' 中 source 为 '{source}' 的文档片段")
            logger.info(f"删除的实体数量: {delete_result.delete_count}")
            
            # 刷新集合以确保删除生效
            collection.flush()
            
            return True
        except Exception as e:
            logger.error(f"按source删除文档失败: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """测试函数"""
    from utils.config_handler import rag_conf
    
    # 创建知识库查看器
    viewer = KnowledgeBaseViewer(
        milvus_host=rag_conf["MILVUS_HOST"],
        milvus_port=rag_conf["MILVUS_PORT"]
    )
    
    print("="*60)
    print("知识库内容查看工具")
    print("="*60)
    
    # 1. 列出所有集合
    print("\n1. 所有集合:")
    collections = viewer.list_all_collections()
    for i, collection in enumerate(collections, 1):
        print(f"   {i}. {collection}")
    
    if not collections:
        print("   没有找到任何集合")
        return
    
    # 2. 显示每个集合的详细信息
    print("\n2. 集合详细信息:")
    for collection_name in collections:
        print(f"\n   集合: {collection_name}")
        stats = viewer.get_collection_stats(collection_name)
        if stats:
            print(f"      文档数量: {stats['document_count']}")
            print(f"      主键字段: {stats['primary_field']}")
            print(f"      描述: {stats['description']}")
            print(f"      字段信息: {list(stats['schema'].keys())}")
        else:
            print("      无法获取统计信息")
    
    # 3. 显示第一个集合的文档内容（最多显示10个）
    if collections:
        first_collection = collections[0]
        print(f"\n3. 集合 '{first_collection}' 中的部分文档内容:")
        docs = viewer.list_documents_in_collection(first_collection, limit=5)
        
        for i, doc in enumerate(docs['documents'], 1):
            print(f"\n   文档 {i} (ID: {doc['id']}):")
            print(f"      内容: {doc['content'][:200]}...")  # 只显示前200个字符
            print(f"      完整长度: {doc['full_length']}")
            print(f"      片段数量: {doc['fragment_count']}")
            print(f"      元数据: {doc['metadata']}")
    
    print(f"\n   总片段数: {docs['total_count']}")
    
    print("\n" + "="*60)
    print("知识库内容查看完成")
    print("="*60)


if __name__ == "__main__":
    main()