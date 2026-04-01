"""
向量检索器模块
基于向量数据库实现智能问答和内容检索
"""
import os
from openai import OpenAI
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage

import sys

from vector_db_manager import VectorDatabaseManager
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from utils.config_handler import rag_conf
from utils.prompt_loader import load_rag_prompts, load_hyde_prompts
from model.factory import chat_model, get_rerank_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """检索结果数据类"""
    content: str
    score: float
    metadata: Dict[str, Any]
    source: str
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
            "source": self.source
        }


@dataclass
class AnswerResult:
    """问答结果数据类"""
    answer: str
    question_type: str
    source_documents: List[Document]
    scores: List[float]

def print_prompt(prompt):
    print("="*20)
    #print(prompt.to_string())
    print("="*20)
    return prompt


class VectorRetriever:
    """向量检索器"""
    
    def __init__(self, 
                 db_manager: VectorDatabaseManager,
                 similarity_threshold: float = 0.5, # 适当调整阈值
                 max_results: int = 10):
        """
        初始化向量检索器
        
        Args:
            db_manager: 向量数据库管理器
            similarity_threshold: 相似度阈值
            max_results: 最大返回结果数
        """
        self.db_manager = db_manager
        self.similarity_threshold = similarity_threshold
        self.max_results = max_results
        self.chat_model_name = rag_conf["chat_model_name"]
        self.prompt_template = PromptTemplate.from_template(load_rag_prompts())
        self.chat_model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self):
        chain = self.prompt_template | print_prompt | self.chat_model | StrOutputParser()
        return chain
    
    def search_similar_content(self,
                             query: str,
                             collection_name: str,
                             k: int = None,
                             filter_expression: str = None,
                             include_scores: bool = True) -> List[Tuple[Document, float]]:
        """
        搜索相似内容
        
        Args:
            query: 查询文本
            collection_name: Milvus 集合名称
            k: 返回结果数量
            filter_expression: Milvus 过滤表达式
            include_scores: 是否包含相似度分数
            
        Returns:
            检索结果列表 (文档, 分数)
        """
        if k is None:
            k = self.max_results
        
        try:
            search_results = self.db_manager.search(query=query, k=k, collection_name=collection_name)

            results = []
            for doc, score in search_results:
                if score >= self.similarity_threshold:
                    results.append((doc, score))
            
            logger.info(f"在集合 '{collection_name}' 中检索查询, 返回 {len(results)} 个高质量结果")
            return results
            
        except Exception as e:
            logger.error(f"在集合 '{collection_name}' 中检索失败: {e}")
            return []
    
    def answer_question(self, 
                        question: str, 
                        collection_name: str, 
                        k: int = 5,
                        use_multi_query: bool = False,
                        use_hyde: bool = True,
                        use_cross_encoder_rerank: bool = True) -> AnswerResult:
        """
        回答问题 
        Args:
            question: 问题文本
            collection_name: Milvus 集合名称
            k: 上下文文档数量
        Returns:
            回答结果
        """
        try:
            question_type = QuestionClassifier.classify_question(question)

            all_queries = [question]
            if use_multi_query:
                generated_queries = self.generate_multi_queries(question, num_queries=rag_conf["multi_query_nums"])
                all_queries.extend(generated_queries)
                logger.info(f"多查询模式开启，生成了 {len(generated_queries)} 个新查询: {generated_queries}")
            elif use_hyde:
                hyde_docs = self.generate_hypothetical_document(question, num_docs=rag_conf["hyde_docs_nums"])
                all_queries.extend(hyde_docs)
                logger.info(f"HyDE模式开启，生成了 {len(hyde_docs)} 个假设文档作为查询")
                all_queries.extend(hyde_docs)
            
            search_results_dict = {}     
            for query in all_queries:
                query_results = self.search_similar_content(
                    query=query,
                    collection_name=collection_name,
                    k=k  
                )
                search_results_dict[query] = query_results

            if use_cross_encoder_rerank:
                contents, scores = self.cross_encoder_rerank(question, search_results_dict, top_n=k)
                source_documents = []
                context_parts = []
                for content_dict, score in zip(contents, scores):
                    page_content = content_dict['page_content']
                    metadata = content_dict['metadata']
                    doc = Document(page_content=page_content, metadata=metadata)
                    source_documents.append(doc)
                    context_parts.append(f"参考资料{len(context_parts)+1}: {doc.page_content}")
                context = "\n\n".join(context_parts)
                normalized_scores = scores
            else:
                rrf_results = self.reciprocal_rank_fusion(search_results_dict, top_n=k)
            
                context_parts = []
                source_documents = []
                scores = []
                
                for doc, original_score in rrf_results:
                    context_parts.append(f"参考资料{len(context_parts)+1}: {doc.page_content}")
                    source_documents.append(doc)
                    scores.append(original_score)
                
                context = "\n\n".join(context_parts)

                normalized_scores = self._calculate_score(scores)
            
            answer = self._generate_answer_with_llm(question, context)
            
            return AnswerResult(
                answer=answer,
                question_type=question_type,
                source_documents=source_documents,
                scores=normalized_scores
            )
            
        except Exception as e:
            logger.error(f"回答问题失败: {e}")
            return AnswerResult(
                answer=f"处理问题时出现错误: {str(e)}",
                question_type="错误",
                source_documents=[],
                scores=[]
            )
    
    def _generate_answer_with_llm(self, question: str, context: str) -> str:
        """使用 LLM 生成回答"""
        try:                                              
            rag_prompt = ""
            if context:
                rag_prompt += f"【参考资料】：\n{context}"
            else:
                rag_prompt += "【参考资料】：(无)"  

            response = self.chain.invoke(
                {
                    "input": question,
                    "context": rag_prompt,
                }
            )          
            
            return response
            
        except Exception as e:
            logger.error(f"LLM 生成失败: {e}")
            return "抱歉，生成回答时出现错误，请稍后再试。"
        
    def generate_multi_queries(self, original_query: str, num_queries: int = 3) -> List[str]:
        """使用大模型生成多个语义相似但表达不同的查询语句"""
        prompt_template = f"""
        你是一个查询扩展专家。请针对用户的问题，生成 {num_queries} 个语义相同但表达方式不同的查询。
        这些查询应该从不同的角度或使用不同的词汇来表达相同的意图，以帮助检索系统找到最相关的信息。
        请只输出查询语句，每行一个，不要包含任何序号或额外解释。

        用户问题: {original_query}

        请生成 {num_queries} 个变体查询:
        """
        try:
            response = self.chat_model.invoke(prompt_template, temperature=0.7) 
            raw_text = response.content.strip()
            queries = [q.strip() for q in raw_text.split('\n') if q.strip()]
            return queries[:num_queries] 
        except Exception as e:
            logger.warning(f"生成多查询失败，回退到原始查询: {e}")
            return [original_query]
        
    def generate_hypothetical_document(self, query: str, num_docs=3) -> List[str]:
        """生成多个假设性文档"""
        prompt_template = PromptTemplate.from_template(load_hyde_prompts())
        hypothetical_docs = []
        for _ in range(num_docs):
            response = self.chat_model.invoke(prompt_template.format(query=query), temperature=0.8)
            hypothetical_docs.append(response.content)
        return hypothetical_docs
    
    def reciprocal_rank_fusion(self, search_results_dict, top_n: int=5,  k=60):
        """使用倒数排序融合算法合并多个搜索结果"""
        fused_scores = {}
        doc_map = {}  # 映射哈希值到(doc, score)元组
        
        for query, doc_scores in search_results_dict.items():
            for rank, (doc, score) in enumerate(doc_scores, start=1):
                doc_hash = hash(doc.page_content)  # 使用文档内容的哈希值作为唯一标识符
                doc_map[doc_hash] = (doc, score)
                
                if doc_hash not in fused_scores:
                    fused_scores[doc_hash] = 0
                fused_scores[doc_hash] += 1 / (k + rank)
        
        reranked_results = sorted(
            fused_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
 
        return [(doc_map[doc_hash][0], doc_map[doc_hash][1]) for doc_hash, score in reranked_results[:top_n]]
    
    def cross_encoder_rerank(self, query: str, search_results_dict, top_n: int = 5):
        """使用交叉编码器重排序"""
        all_docs_list = []
        seen_content = set()  
        
        for query_text, doc_scores in search_results_dict.items():
            for doc, score in doc_scores:
                if doc.metadata is None:
                    doc.metadata = {}
                content_hash = hash(doc.page_content)
                if content_hash not in seen_content:
                    all_docs_list.append(doc)
                    seen_content.add(content_hash)
        
        rerankmodel = get_rerank_model(top_n)
        reranked_results = rerankmodel.compress_documents(documents=all_docs_list, query=query)
            
        contents = []
        scores = []
        for idx, doc in enumerate(reranked_results):
            relevance_score = doc.metadata.get('relevance_score', 'N/A')
            content_with_metadata = {
                'page_content': doc.page_content,
                'metadata': doc.metadata
            }
            contents.append(content_with_metadata)
            scores.append(relevance_score)
            
        return contents, scores
    
    def _calculate_score(self, scores: List[float]) -> List[float]:
        """计算回答置信度"""
        raw_similarities = [1.0 - d for d in scores]

        min_raw = min(raw_similarities)
        max_raw = max(raw_similarities)

        if max_raw == min_raw:
            return [1.0] * len(scores)
        
        normalized = [
            (s - min_raw) / (max_raw - min_raw)
            for s in raw_similarities
        ]
        return normalized
        
    def get_statistics(self, collection_name: str) -> Dict[str, Any]:
        """获取检索统计信息"""
        db_info = self.db_manager.get_database_info()
        
        stats = {
            "database_info": db_info,
            "similarity_threshold": self.similarity_threshold,
            "max_results": self.max_results,
            "retriever_status": "active" if db_info.get("is_initialized") else "inactive"
        }
        
        return stats


class QuestionClassifier:
    @classmethod
    def classify_question(cls, question: str) -> str:
        return "通用查询"


def main():
    """测试函数"""
    # 初始化 Milvus 连接
    try:
        db_manager = VectorDatabaseManager(
            milvus_host=rag_conf["MILVUS_HOST"],
            milvus_port=rag_conf["MILVUS_PORT"],
        )
        retriever = VectorRetriever(db_manager)
        collection_name = "agent_rag"
        print("向量系统初始化成功")
    except Exception as e:
        print(f"向量系统初始化失败: {e}")
        return

    # 准备测试数据
    info = db_manager.get_database_info()
    if not info.get("is_initialized"):
        print(f"集合 '{collection_name}' 不存在，正在创建并添加数据...")
        # 此处可以添加一个示例文件上传的逻辑
        # 例如: db_manager.process_file("path/to/your/data.csv", collection_name)
        print("请先手动上传数据以进行测试。")
        # return # 如果没有数据，可以选择退出

    # 测试问题回答
    test_questions = [
        "Milvus 是什么？",
        "如何将文本上传到向量数据库？",
        "RAG 工作流程的关键步骤是什么？"
    ]
    
    print("\n--- 测试问答功能 ---")
    for question in test_questions:
        print(f"\n问题: {question}")
        
        result = retriever.answer_question(question, collection_name=collection_name)
        print(f"回答: {result.answer}")
        print(f"置信度: {result.confidence:.2f}")
        print(f"问题类型: {result.question_type}")
        print(f"参考来源数: {len(result.source_documents)}")
    
    


if __name__ == "__main__":
    main()