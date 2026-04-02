from fastmcp import FastMCP
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))  
# Initialize the server with a name  
mcp = FastMCP("my-first-server")  
from typing import List, Dict, Any, Optional, Tuple
from rag.vector_db_manager import VectorDatabaseManager
from rag.vector_retriever import VectorRetriever

#初始化
vector_db=VectorDatabaseManager()
retriever=VectorRetriever(vector_db)
# Define a tool using the @mcp.tool decorator  
@mcp.tool  
def get_weather(city: str) -> dict:  
    """Get the current weather for a city."""  
    # In production, you'd call a real weather API  
    # For now, we'll return mock data  
    weather_data = {  
        "new york": {"temp": 72, "condition": "sunny"},  
        "london": {"temp": 59, "condition": "cloudy"},  
        "tokyo": {"temp": 68, "condition": "rainy"},  
    }  
      
    city_lower = city.lower()  
    if city_lower in weather_data:  
        return {"city": city, **weather_data[city_lower]}  
    else:  
        return {"city": city, "temp": 70, "condition": "unknown"}  

# 获取知识库检索工具
@mcp.tool
def rag_summarize(query:str,k: int = 5, filter_dict: Dict = None, collection_name: Optional[str] = None):
    
    results=vector_db.search(query,k=5,filter_dict=filter_dict,collection_name=collection_name)
    return results
#获取基于知识库的LLM回答
@mcp.tool
def get_llm_answer(question: str, 
                        collection_name: str, 
                        k: int = 5,
                        use_multi_query: bool = True,
                        use_hyde: bool = False,
                        use_cross_encoder_rerank: bool = True):
    results=retriever.answer_question(question,collection_name=collection_name,k=k,use_multi_query=use_multi_query,use_hyde=use_hyde,use_cross_encoder_rerank=use_cross_encoder_rerank)
    return results
    

# Run the server  
if __name__ == "__main__":  
     mcp.run(transport="http", host="127.0.0.1", port=8000)