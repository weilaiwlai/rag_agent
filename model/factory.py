from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_openai import ChatOpenAI 
from utils.config_handler import rag_conf


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return ChatTongyi(
            model=rag_conf["chat_model_name"],
            api_key=rag_conf["dashscope_api_key"]
        )
        #return ChatOpenAI(
        #    api_key="EMPTY",  
        #    base_url="http://202.204.62.144:8000/v1", 
        #    model="/opt/models/Qwen3-0.6B"
        #)


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"], 
                                   dashscope_api_key=rag_conf["dashscope_api_key"])


chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()
