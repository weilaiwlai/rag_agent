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
from model.factory import chat_model
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
                        k: int = 5) -> AnswerResult:
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
            hypo_docs = self.generate_hypothetical_document(question, num_docs=2)
            all_queries.extend(hypo_docs)
            
            search_results_dict = {}
            all_docs_map = {}  
            
            for query in all_queries:
                query_results = self.search_similar_content(
                    query=query,
                    collection_name=collection_name,
                    k=k  
                )

                doc_score_pairs = []
                for doc, score in query_results:
                    doc_id = hash(doc.page_content)
                    doc_score_pairs.append((doc_id, score))
                    all_docs_map[doc_id] = (doc, score)
                
                search_results_dict[query] = doc_score_pairs
            
            rrf_results = self.reciprocal_rank_fusion(search_results_dict, num_docs=k)
            
            relevant_docs_with_scores = []
            context_parts = []
            source_documents = []
            scores = []
            
            for doc_id, rrf_score in rrf_results:  
                if doc_id in all_docs_map:
                    doc, original_score = all_docs_map[doc_id]
                    relevant_docs_with_scores.append((doc, original_score))
                    context_parts.append(f"参考资料{len(context_parts)+1}: {doc.page_content}")
                    source_documents.append(doc)
                    scores.append(original_score)
            
            context = "\n\n".join(context_parts)
            
            answer = self._generate_answer_with_llm(question, context)
            
            normalized_scores = self._calculate_score(scores)
            
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
        
    def generate_hypothetical_document(self, query: str, num_docs=3) -> List[str]:
        """生成多个假设性文档"""
        prompt_template = PromptTemplate.from_template(load_hyde_prompts())
        hypothetical_docs = []
        for _ in range(num_docs):
            response = self.chat_model.invoke(prompt_template.format(query=query), temperature=0.8)
            hypothetical_docs.append(response.content)
        return hypothetical_docs
    
    def reciprocal_rank_fusion(self, search_results_dict, num_docs: int=5,  k=60):
        """使用倒数排序融合算法合并多个搜索结果"""
        fused_scores = {}
        for query, doc_scores in search_results_dict.items():
            for rank, (doc_id, score) in enumerate(doc_scores, start=1):
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0
                fused_scores[doc_id] += 1 / (k + rank)
        reranked_results = sorted(
            fused_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return reranked_results[:num_docs]
    
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
        """
        获取检索统计信息
        
        Args:
            collection_name: Milvus 集合名称

        Returns:
            统计信息字典
        """
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