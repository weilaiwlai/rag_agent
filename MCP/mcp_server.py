from fastmcp import FastMCP
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))  
from typing import List, Dict, Any, Optional, Tuple
from rag.vector_db_manager import VectorDatabaseManager
from rag.vector_retriever import VectorRetriever

#初始化
vector_db=VectorDatabaseManager()
retriever=VectorRetriever(vector_db)

# Initialize the server with a name  
mcp = FastMCP("RagServer")  

# 获取知识库检索工具
@mcp.tool
def rag_summarize(query:str,k: int = 5, filter_dict: Dict = None, collection_name: Optional[str] = ["agent_rag"]):
    '''
    检索知识库中的文档，collection_name默认为agent_rag，k为检索数量，默认5条，query为检索关键词
    '''
    results=vector_db.search(query,k=5,filter_dict=filter_dict,collection_name=collection_name)
    return results

#获取基于知识库的LLM回答
@mcp.tool
def get_llm_answer(question: str, 
                        collection_name: str="agent_rag", 
                        k: int = 5,
                        use_multi_query: bool = True,
                        use_hyde: bool = False,
                        use_cross_encoder_rerank: bool = True):
    '''
    获取基于知识库的LLM回答，collection_name默认为agent_rag，k为检索数量，默认5条，question为问题
    '''
    results=retriever.answer_question(question,collection_name=collection_name,k=k,use_multi_query=use_multi_query,use_hyde=use_hyde,use_cross_encoder_rerank=use_cross_encoder_rerank)
    return results
    

# Run the server  
if __name__ == "__main__":  
     mcp.run(transport="http", host="127.0.0.1", port=8001)