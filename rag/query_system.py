"""
用途：基于 Milvus 中已存储的文本执行向量检索与 RAG 问答。
特点：使用 DashScope 的 OpenAI 兼容接口生成结构化回答，命令行示例可直接运行。
"""
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from langchain_community.embeddings import DashScopeEmbeddings
from openai import OpenAI
from pymilvus import connections, Collection, utility, DataType
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- 在这里定义您的配置参数 ---
QUESTION = "XXXXX"  # <-- 替换为您的问题
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "agent_rag")
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v1")
TOP_K = 5    # 检索的文档数量
# LLM配置（使用DashScope的OpenAI兼容接口）
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen-plus")
# ----------------------------------------


class SimpleQuerySystem:
    def __init__(self, host, port, collection_name, dashscope_api_key, embedding_model):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dashscope_api_key = dashscope_api_key
        self.embedding_model = embedding_model
        self.vector_field = None
        self.text_fields = []
        self._trace_logs: List[Dict[str, str]] = []
        
        # 连接到Milvus
        self.connect_milvus()
        
        # 初始化模型
        self.embeddings_model = DashScopeEmbeddings(
            model=self.embedding_model,
            dashscope_api_key=self.dashscope_api_key
        )
        
        # 初始化OpenAI兼容客户端（DashScope）
        self.chat_client = OpenAI(
            api_key=self.dashscope_api_key,
            base_url=LLM_BASE_URL,
        )

        # 解析并记住集合的字段信息
        self.resolve_fields()

    def _init_trace(self):
        self._trace_logs = []

    def _log(self, stage: str, message: str):
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        entry = {"timestamp": timestamp, "stage": stage, "message": message}
        self._trace_logs.append(entry)
        print(f"[VectorQS][{stage}] {message}")


    def _summarize_text(self, text: str, max_chars: int = 60) -> str:
        if not text:
            return ""
        clean = " ".join(text.strip().split())
        if len(clean) <= max_chars:
            return clean
        return clean[:max_chars].rstrip() + "…"

    def _format_context_payload(self, contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        formatted = []
        for idx, ctx in enumerate(contexts, 1):
            formatted.append({
                "rank": idx,
                "source": ctx.get('source') or "未知",
                "text": ctx.get('text', ''),
                "score": round(float(ctx.get('score', 0.0)), 4) if isinstance(ctx.get('score'), (int, float)) else None
            })
        return formatted
        
    def connect_milvus(self):
        """连接到Milvus数据库"""
        connections.connect("default", host=self.host, port=self.port)
        self._log("milvus.connect", f"已连接到 Milvus，地址为 {self.host}:{self.port}")
        
    def get_query_embedding(self, query_text: str) -> List[float]:
        """获取查询文本的嵌入向量"""
        self._log("embedding", f"生成查询嵌入向量: {query_text}")
        embedding = self.embeddings_model.embed_documents([query_text])[0]
        return embedding

    def resolve_fields(self):
        """解析集合schema，动态确定向量字段和可输出文本字段"""
        try:
            if not utility.has_collection(self.collection_name):
                self._log("schema", f"错误：集合 '{self.collection_name}' 不存在")
                return
            collection = Collection(name=self.collection_name)
            fields = collection.schema.fields
            # 找向量字段
            vector_fields = [f.name for f in fields if getattr(f, "dtype", None) == DataType.FLOAT_VECTOR]
            if vector_fields:
                self.vector_field = vector_fields[0]
            else:
                print("未找到向量字段（FLOAT_VECTOR）。请检查集合schema。")
            # 找文本字段
            self.text_fields = [f.name for f in fields if getattr(f, "dtype", None) == DataType.VARCHAR]
            if not self.text_fields:
                self._log("schema", "未找到文本字段（VARCHAR），结果将缺少文本内容")
            self._log("schema", f"解析字段完成：vector_field={self.vector_field}, text_fields={self.text_fields}")
        except Exception as e:
            self._log("schema", f"解析集合字段时出错：{e}")
        
    def search_similar_text(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """在Milvus中搜索相似文本"""
        # 检查集合是否存在
        if not utility.has_collection(self.collection_name):
            self._log("vector.search", f"错误：集合 '{self.collection_name}' 不存在")
            return []
            
        try:
            # 获取集合
            collection = Collection(name=self.collection_name)
            collection.load()
            self._log("vector.search", f"集合 '{self.collection_name}' 已加载")
            
            # 获取查询嵌入向量
            query_embedding = self.get_query_embedding(query_text)
            
            # 设置搜索参数
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10}
            }
            
            # 执行向量搜索
            self._log("vector.search", f"正在搜索相似文档，top_k={top_k}")
            # 使用解析到的字段名称
            anns_field = self.vector_field or "embedding"
            output_fields = self.text_fields[:2] if self.text_fields else []
            results = collection.search(
                data=[query_embedding],
                anns_field=anns_field,
                param=search_params,
                limit=top_k,
                output_fields=output_fields
            )
            
            # 提取搜索结果
            search_results = []
            for hits in results:
                for hit in hits:
                    # 动态映射文本字段
                    source = hit.entity.get(self.text_fields[0]) if len(self.text_fields) > 0 else None
                    text = hit.entity.get(self.text_fields[1]) if len(self.text_fields) > 1 else source
                    result = {"source": source, "text": text, "score": hit.score}
                    search_results.append(result)
                    self._log("vector.hit", f"找到匹配：{result['source']} (得分: {result['score']:.4f})")
            
            return search_results
            
        except Exception as e:
            self._log("vector.search", f"搜索时出错：{e}")
            return []
            
    def generate_response(self, contexts: List[Dict[str, Any]], question: str, model: Optional[str] = None) -> str:
        """基于检索到的上下文生成回答，如果无上下文则使用通用知识"""
        target_model = "qwen-plus"
        client = self.chat_client

        if not contexts:
            # 检索为空时，回退到模型通用知识回答
            self._log("llm", "检索结果为空，使用模型通用知识回答")
            messages = [
                {
                    "role": "system",
                    "content": "你是一个智能助手。用户的问题在知识库中未找到相关信息，请利用你的通用知识尝试回答，并友好地告知用户该信息可能不包含在上传的文档中。"
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
            try:
                completion = client.chat.completions.create(
                    model=target_model,
                    messages=messages,
                )
                return completion.choices[0].message.content
            except Exception as e:
                self._log("llm", f"通用回答生成失败: {e}")
                return "未找到与问题相关的信息。请尝试重新表述您的问题。"
            
        self._log("llm", "正在基于检索到的上下文生成回答")
        
        # 准备上下文
        context_texts = []
        for idx, ctx in enumerate(contexts, 1):
            context_texts.append(
                f"[{idx}] 来源: {ctx.get('source', '未知')}\n内容: {ctx.get('text', '')}"
            )
        
        combined_context = "\n\n---\n\n".join(context_texts)

        target_model = "qwen-plus"
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "你是一个中文 RAG 助手。\n"
                        "原则：\n"
                        "- 仅依据提供的上下文作答，不得编造；\n"
                        "- 当上下文不足或无关时，明确说明无法回答并指出缺失信息类型；\n"
                        "- 语言简洁准确，尽量提炼关键结论与要点；检索到的上下文和用户问的问题没关系的时候，忽略上下文！\n"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"问题：{question}\n\n"
                        f"上下文（按序编号）：\n{combined_context}\n\n"
                        "请：\n"
                        "1) 仅基于上述上下文回答；\n"
                        "2) 在相关句后用【编号】注明引用；\n"
                        "3) 若无法回答，说明缺少的上下文类型（例如：定义、步骤、数据）。"
                    ),
                },
            ]

            client = self.chat_client

            completion = client.chat.completions.create(
                model=target_model,
                messages=messages,
            )
            response_text = completion.choices[0].message.content
            self._log("llm", "回答生成完成")
            return response_text
            
        except Exception as e:
            self._log("llm", f"生成回答时出错：{e}")
            return f"抱歉，生成回答时出现错误：{e}"
            
    def get_answer(self, query_text: str, top_k: int = 5, model: Optional[str] = None) -> Dict[str, Any]:
        """获取问题的答案（包含上下文、摘要与日志）"""
        self._init_trace()
        self._log("question", f"处理查询：{query_text}")
        
        contexts = self.search_similar_text(query_text, top_k)
        
        # 无论是否有上下文，都尝试生成回答（generate_response 内部处理了回退逻辑）
        answer_text = self.generate_response(contexts, query_text, model=model)
        formatted_contexts = self._format_context_payload(contexts)
        
        payload = {
            "answer": answer_text,
            "summary": self._summarize_text(answer_text),
            "question": query_text,
            "contexts": formatted_contexts,
            "top_k": len(formatted_contexts),
            "logs": self._trace_logs.copy()
        }
        return payload

def main():
    """主函数"""
    print("初始化查询系统...")
    
    query_system = SimpleQuerySystem(
        host=MILVUS_HOST,
        port=MILVUS_PORT,
        collection_name=COLLECTION_NAME,
        dashscope_api_key=DASHSCOPE_API_KEY,
        embedding_model=EMBEDDING_MODEL
    )
    
    print(f"用户问题：{QUESTION}")
    print("=" * 50)
    
    # 获取答案
    answer = query_system.get_answer(QUESTION, TOP_K)
    
    print("\n回答：")
    print("=" * 50)
    print(answer.get("answer", ""))
    print("=" * 50)
    print(f"摘要：{answer.get('summary', '')}")
    if answer.get("contexts"):
        print("\n引用的上下文：")
        for ctx in answer["contexts"]:
            print(f"- [{ctx['rank']}] {ctx['source']} (score={ctx.get('score')})")

if __name__ == "__main__":
    main()
