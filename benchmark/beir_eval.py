# beir_evaluation_runner.py
import os
import sys
import argparse
import json
from typing import Dict, List, Optional
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
import logging
from pathlib import Path
# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入你的RAG系统组件
from rag.vector_db_manager import VectorDatabaseManager
from rag.vector_retriever import VectorRetriever
from utils.config_handler import rag_conf
import hashlib

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import time

global global_query_time
global_query_time = 0
global global_rerank_time
global_rerank_time = 0

# 全局变量用于缓存多查询结果
MULTI_QUERY_CACHE = {}
CACHE_FILE_PATH = "multi_query_cache.json"
 
def load_multi_query_cache():
    """从文件加载多查询缓存"""
    global MULTI_QUERY_CACHE
    try:
        if os.path.exists(CACHE_FILE_PATH):
            with open(CACHE_FILE_PATH, 'r', encoding='utf-8') as f:
                MULTI_QUERY_CACHE = json.load(f)
            logger.info(f"已从 {CACHE_FILE_PATH} 加载 {len(MULTI_QUERY_CACHE)} 个多查询缓存条目")
        else:
            logger.info(f"缓存文件 {CACHE_FILE_PATH} 不存在，将创建新的缓存")
    except Exception as e:
        logger.error(f"加载多查询缓存失败: {e}")
        MULTI_QUERY_CACHE = {}
 
def save_multi_query_cache():
    """保存多查询缓存到文件"""
    try:
        with open(CACHE_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(MULTI_QUERY_CACHE, f, ensure_ascii=False, indent=2)
        logger.info(f"已将 {len(MULTI_QUERY_CACHE)} 个多查询缓存条目保存到 {CACHE_FILE_PATH}")
    except Exception as e:
        logger.error(f"保存多查询缓存失败: {e}")

def get_query_hash(query_text: str) -> str:
    """生成查询文本的哈希值作为唯一标识"""
    return hashlib.md5(query_text.encode('utf-8')).hexdigest()

class RAGSystemEvaluator:
    """
    评估你的RAG系统的类
    """
    def __init__(self, collection_name: str = None, top_k: int = 10, multi_query: bool = False, hyde: bool = False, rerank: str = 'RRF'):
        """
        初始化评估器
        
        Args:
            collection_name: 要评估的知识库集合名称
            top_k: 默认返回的顶部结果数
        """
        # 初始化你的RAG系统的组件
        self.db_manager = VectorDatabaseManager(
            milvus_host=rag_conf["MILVUS_HOST"],
            milvus_port=rag_conf["MILVUS_PORT"],
        )
        self.retriever = VectorRetriever(self.db_manager)
        self.collection_name = collection_name or rag_conf["COLLECTION_NAME"]
        self.top_k = top_k
        self.multi_query = multi_query
        self.hyde = hyde
        self.rerank = rerank
        
    def search(self, queries: Dict[str, str], top_k: int = None) -> Dict[str, Dict[str, float]]:
        """
        在你的RAG系统中批量搜索
        
        Args:
            queries: {query_id: query_text} 字典
            top_k: 每个查询返回的顶部结果数
            
        Returns:
            {query_id: {doc_id: score}} 结果字典
        """
        if top_k is None:
            top_k = self.top_k
            
        results = {}
        
        for qid, query_text in queries.items():
            logger.info(f"正在检索查询 {qid}: {query_text[:50]}...")
            try:
                start_time = time.time()
                query_time = 0
                if self.multi_query:
                    all_queries = [query_text]
                    # 检查缓存
                    query_hash = get_query_hash(query_text)
                    cache_key = f"multi_query_{query_hash}"
                    if cache_key in MULTI_QUERY_CACHE:
                        # 从缓存获取结果
                        cache_time = time.time()
                        cached_data = MULTI_QUERY_CACHE[cache_key]
                        generated_queries = cached_data["queries"]
                        generation_time = cached_data["time"]
                        cache_time = time.time()-cache_time
                        logger.info(f"多查询模式 - 从缓存获取结果 (耗时: {generation_time:.4f}s): {generated_queries}")
                        query_time = generation_time - cache_time    #读缓存时间远小于生成时间
                    else:
                        # 调用LLM生成查询并记录时间
                        llm_start_time = time.time()
                        generated_queries = self.retriever.generate_multi_queries(query_text, num_queries=rag_conf["multi_query_nums"])
                        llm_end_time = time.time()
                        generation_time = llm_end_time - llm_start_time
                        
                        # 缓存结果
                        MULTI_QUERY_CACHE[cache_key] = {
                            "queries": generated_queries,
                            "time": generation_time,
                            "timestamp": time.time()
                        }
                        save_multi_query_cache()  # 立即保存到文件
                    logger.info(f"多查询模式开启，生成了 {len(generated_queries)} 个新查询: {generated_queries}")
                    search_results_dict = {}
                    all_queries.extend(generated_queries)     
                    for query in all_queries:
                        query_results = self.retriever.search_similar_content(
                            query=query,
                            collection_name=self.collection_name,
                            k=top_k  
                        )
                        search_results_dict[query] = query_results
                elif self.hyde:
                    all_queries = [query_text]
                    hyde_docs = self.retriever.generate_hypothetical_document(query_text, num_docs=rag_conf["hyde_docs_nums"])
                    all_queries.extend(hyde_docs)
                    logger.info(f"HyDE模式开启，生成了 {len(hyde_docs)} 个假设文档作为查询")
                    search_results_dict = {}     
                    for query in all_queries:
                        query_results = self.retriever.search_similar_content(
                            query=query,
                            collection_name=self.collection_name,
                            k=top_k  
                        )
                        search_results_dict[query] = query_results
                else:
                    # 使用RAG系统的search_similar_content方法
                    search_results = self.retriever.search_similar_content(
                            query=query_text,
                            collection_name=self.collection_name,
                            k=top_k  
                        )
                end_time = time.time()
                query_time += end_time - start_time
                start_time = time.time()
                if self.multi_query or self.hyde:       # 多查询或HyDE模式下，需要对每个查询的结果进行重排
                    if self.rerank == 'model': #使用交叉编码器重排
                        content, scores = self.retriever.cross_encoder_rerank(query_text, search_results_dict, top_n=top_k)
                        search_results = list(zip(content, scores))
                    elif self.rerank == 'RRF': #如果不使用交叉编码器重排，使用RRF
                        search_results = self.retriever.reciprocal_rank_fusion(search_results_dict, top_n=top_k)
                    elif self.rerank == 'rrf+model':
                        RRF_result=self.retriever.reciprocal_rank_fusion(search_results_dict, top_n=top_k*2)
                        search_results_dict={}
                        search_results_dict[0]=RRF_result
                        content, scores = self.retriever.cross_encoder_rerank(query_text, search_results_dict, top_n=top_k)
                        search_results = list(zip(content, scores))
                end_time = time.time()
                rerank_time = end_time - start_time
                # 将结果转换为BEIR期望的格式 {doc_id: score}
                formatted_results = {}
                for idx, (doc, score) in enumerate(search_results):
                    # # 使用文档元数据中的source作为ID，否则使用文档内容的哈希
                    # doc_id = doc.metadata.get('source', f'doc_{idx}_{abs(hash(doc.page_content[:100]))}')
                    # # 确保ID是字符串且唯一
                    # doc_id = str(doc_id).replace(':', '_').replace('/', '_')
                    
                    # # 如果ID重复，添加索引以确保唯一性
                    # original_doc_id = doc_id
                    # counter = 0
                    # while doc_id in formatted_results:
                    #     counter += 1
                    #     doc_id = f"{original_doc_id}_{counter}"
                    if self.rerank == 'model' or self.rerank == 'rrf+model':
                        doc_id = doc['metadata']['id']
                    else:
                        doc_id = doc.metadata.get('id')
                    # BEIR期望的是相关性得分，越高越好
                    # 你的系统返回的score已经是相似度分数，直接使用
                    formatted_results[doc_id] = float(score)
                results[qid] = formatted_results
                global global_query_time
                global_query_time += query_time
                global global_rerank_time
                global_rerank_time += rerank_time
                logger.info(f"查询 {qid} 返回了 {len(formatted_results)} 个结果，查询耗时 {query_time:.4f} 秒, 重排耗时 {rerank_time:.4f} 秒")
                
            except Exception as e:
                logger.error(f"查询 {qid} 搜索失败: {e}")
                import traceback
                traceback.print_exc()
                results[qid] = {}
        
        return results

def evaluate_rag_on_beir_dataset(
    dataset_name: str,
    collection_name: str = None,
    top_k_values: List[int] = [1, 3, 5, 10],
    output_dir: str = "evaluation_results",
    multi_query: bool = False,
    hyde: bool = False,
    rerank: str = 'RRF',
):
    """
    在指定的BEIR数据集上评估你的RAG系统
    
    Args:
        dataset_name: BEIR数据集名称
        collection_name: 你的RAG系统中的集合名称
        top_k_values: 评估的k值列表
        output_dir: 结果输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 下载BEIR数据集
    logger.info(f"正在下载BEIR数据集: {dataset_name}")
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    data_path = util.download_and_unzip(url, "./datasets")
    
    # 加载数据
    logger.info("正在加载数据集...")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    
    logger.info(f"数据集加载完成: {len(corpus)} 个文档, {len(queries)} 个查询")
    
    # 初始化你的RAG评估器
    rag_evaluator = RAGSystemEvaluator(collection_name=collection_name, top_k=max(top_k_values), multi_query=multi_query, hyde=hyde, rerank=rerank)
    
    # 创建模拟检索器
    class MockRetriever:
        def __init__(self, evaluator):
            self.evaluator = evaluator
            self.k_values = top_k_values
            
        def search(self, corpus, queries, top_k, *args, **kwargs):
            # 使用你的RAG系统执行搜索
            return self.evaluator.search(queries, top_k)
    
    # 创建评估对象
    mock_retriever = MockRetriever(rag_evaluator)
    evaluator = EvaluateRetrieval(mock_retriever, k_values=top_k_values)
    
    # 执行检索
    logger.info("开始检索...")
    evaluator.results = mock_retriever.search(corpus, queries, max(top_k_values))

    logger.info(f"检索:{evaluator.results}")
    
    # 执行评估
    logger.info("开始评估...")
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, evaluator.results, evaluator.k_values)
    
    # 计算MRR作为额外指标
    try:
        from beir.retrieval.custom_metrics import mrr
        mrr_scores = mrr(qrels, evaluator.results, evaluator.k_values)
    except ImportError:
        logger.warning("无法导入MRR指标，跳过")
        mrr_scores = {}
    
    # 输出结果摘要
    global global_query_time
    global global_rerank_time
    print("\n" + "="*60)
    print(f"RAG系统评估结果 - 数据集: {dataset_name}")
    print("="*60)
    print(f"知识库集合: {collection_name or rag_conf['COLLECTION_NAME']}")
    print(f"总查询数: {len(queries)}")
    print(f"总文档数: {len(corpus)}")
    print(f"平均查询耗时: {global_query_time/len(queries):.4f} 秒")
    print(f"平均重排耗时: {global_rerank_time/len(queries):.4f} 秒")
    print(f"评估指标: {top_k_values}")
    print("\n性能指标:")
    
    metrics_summary = {}
    for metric_name, metric_values in {"NDCG": ndcg, "MAP": _map, "Recall": recall, "Precision": precision}.items():
        print(f"  {metric_name}:")
        for k, score in metric_values.items():
            print(f"    @{k}: {score:.4f}")
            if metric_name not in metrics_summary:
                metrics_summary[metric_name] = {}
            metrics_summary[metric_name][f"@{k}"] = round(score, 4)
    
    if mrr_scores:
        print(f"  MRR:")
        for k, score in mrr_scores.items():
            print(f"    @{k}: {score:.4f}")
            metrics_summary["MRR"] = {f"@{k}": round(score, 4) for k, score in mrr_scores.items()}
    
    # 保存详细结果
    evaluation_results = {
        "evaluation_config": {
            "dataset_name": dataset_name,
            "collection_name": collection_name or rag_conf["COLLECTION_NAME"],
            "top_k_values": top_k_values,
            "total_queries": len(queries),
            "total_corpus": len(corpus)
        },
        "metrics": metrics_summary,
        "raw_results": {
            "ndcg": {k: float(v) for k, v in ndcg.items()},
            "map": {k: float(v) for k, v in _map.items()},
            "recall": {k: float(v) for k, v in recall.items()},
            "precision": {k: float(v) for k, v in precision.items()},
            "mrr": {k: float(v) for k, v in mrr_scores.items()} if mrr_scores else {},
        },
        "sample_results": dict(list(evaluator.results.items())[:5]),  # 保存前5个查询的详细结果
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }
    
    # 生成结果文件名
    result_filename = f"{dataset_name}_collection_{collection_name or 'default'}{'_multi_query' if multi_query else ''}_eval.json"
    result_file = os.path.join(output_dir, result_filename)
    
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细结果已保存至: {result_file}")
    
    return evaluation_results

def list_available_datasets():
    """列出可用的BEIR数据集"""
    datasets = [
        "trec-covid", "bioasq", "nfcrobus", "hotpotqa", "fiqa", "signal1m", 
        "trec-news", "robust04", "arguana", "webis-touche2020", "cqadupstack",
        "quora", "dbpedia-entity", "scidocs", "fever", "climate-fever", "scifact"
    ]
    print("可用的BEIR数据集:")
    for ds in datasets:
        print(f"  - {ds}")
    return datasets

def main():
    parser = argparse.ArgumentParser(description='评估你的RAG系统在BEIR基准上的性能')
    parser.add_argument('--dataset', type=str, default='scifact', 
                       help='BEIR数据集名称 (默认: scifact)')
    parser.add_argument('--collection', type=str, default=None,
                       help='你的RAG系统中的集合名称 (默认: 配置文件中的默认值)')
    parser.add_argument('--top-k', nargs='+', type=int, default=[1,3,5,10],
                       help='评估的k值 (默认: [1,3,5,10])')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='结果输出目录 (默认: evaluation_results)')
    parser.add_argument('--list-datasets', action='store_true',
                       help='列出所有可用的BEIR数据集')
    parser.add_argument('--multi-query', type=bool, default=False,
                       help='是否启用多查询(默认: False)')
    parser.add_argument('--hyde', type=bool, default=False,
                       help='是否启用假设性文档(默认: False)')
    parser.add_argument('--rerank', type=str, choices=['RRF', 'model','rrf+model'], default='RRF',
                       help='重排序方法(默认:RRF)')

    
    args = parser.parse_args()
    
    if args.list_datasets:
        list_available_datasets()
        return
    
    print(f"开始评估你的RAG系统...")
    print(f"数据集: {args.dataset}")
    print(f"知识库集合: {args.collection or rag_conf['COLLECTION_NAME']}")
    print(f"评估指标: {args.top_k}")
    load_multi_query_cache()
    try:
        results = evaluate_rag_on_beir_dataset(
            dataset_name=args.dataset,
            collection_name=args.collection,
            top_k_values=args.top_k,
            output_dir=args.output_dir,
            multi_query=args.multi_query,
            hyde=args.hyde,
            rerank=args.rerank,
        )
        print("\n评估完成！")
    except Exception as e:
        logger.error(f"评估过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()