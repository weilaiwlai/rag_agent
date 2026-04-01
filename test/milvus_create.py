"""
用途：创建 Milvus 集合并从文件生成嵌入插入数据的示例/测试脚本。
说明：支持 txt/pdf/docx/csv 加载与切分；可单独运行进行快速入库验证。
"""
import os  # 用于文件操作和环境变量处理
from fastapi import FastAPI
# from langchain.document_loaders import TextLoader  # 加载文本文件
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter  # 用于将文档分割为小块
from langchain_community.document_loaders import PyPDFium2Loader, Docx2txtLoader, WebBaseLoader, \
    UnstructuredMarkdownLoader, CSVLoader  # 加载PDF和Docx文件
from langchain_community.embeddings import DashScopeEmbeddings  # 使用DashScope API生成文本嵌入向量
from pymilvus import (
    connections,  # 用于连接Milvus数据库
    Collection,  # 表示Milvus中的集合
    CollectionSchema,  # 用于定义集合的模式
    DataType,  # 定义字段的数据类型
    FieldSchema,  # 定义集合中的字段
    utility  # 用于执行集合相关的实用操作
)
from services.knowledge_service import KnowledgeService


# 定义 MilvusManager 类，用于处理与 Milvus 向量数据库的交互
class MilvusManager:
    """
    一个管理类，用于处理与 Milvus 向量数据库的交互，
    包括连接、创建集合、处理文件和插入数据。
    """

    def __init__(self, host: str, port: str, collection_name: str, embedding_model: str, dashscope_api_key: str):
        # 初始化Milvus连接参数和配置
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dashscope_api_key = dashscope_api_key  # DashScope API 密钥，用于生成嵌入
        self.embedding_model = embedding_model  # 嵌入模型名称
        self.connect_milvus()  # 建立Milvus连接
        self.create_collection_if_not_exists()  # 创建集合（如果集合不存在）

    def connect_milvus(self):
        """建立与 Milvus 服务器的连接。"""
        connections.connect("default", host=self.host, port=self.port)  # 连接到 Milvus 数据库
        print(f"已连接到 Milvus，地址为 {self.host}:{self.port}")  # 输出连接成功消息

    def get_embedding(self, texts):
        """使用 DashScopeEmbeddings 为一组文本生成嵌入向量。"""
        # 使用DashScope嵌入模型生成嵌入向量
        embeddings_model = DashScopeEmbeddings(
            model=self.embedding_model,
            dashscope_api_key=self.dashscope_api_key
        )
        return embeddings_model.embed_documents(texts)  # 返回嵌入向量

    def get_schema(self):
        """定义 Milvus 集合的模式，包括 'name' 字段。"""
        # 定义Milvus集合的字段模式
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),  # 主键ID，自动生成
            FieldSchema(name="user_id", dtype=DataType.INT64),  # 用户ID
            FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=255),  # 文件名
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=5000),  # 文本内容
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536)  # 嵌入向量，维度1536
        ]
        return CollectionSchema(fields=fields, description="包含文件名称的文本嵌入")  # 返回集合的模式

    def create_collection_if_not_exists(self):
        """如果集合不存在，则创建 Milvus 集合。"""
        # 检查集合是否存在，如果不存在则创建新集合
        if not utility.has_collection(self.collection_name):
            schema = self.get_schema()  # 获取集合模式
            collection = Collection(name=self.collection_name, schema=schema)  # 创建集合
            index_params = {
                "index_type": "AUTOINDEX",  # 自动创建索引
                "metric_type": "L2",  # 使用L2距离进行相似度搜索
                "params": {}
            }
            collection.create_index(field_name="embedding", index_params=index_params, index_name='vector_idx')  # 创建索引
            collection.load()  # 加载集合到内存
            print(f"集合 '{self.collection_name}' 已创建并加载。")  # 输出集合创建成功消息
        else:
            print(f"集合 '{self.collection_name}' 已存在。")  # 如果集合已经存在，输出提示
        self.collection = Collection(name=self.collection_name)  # 设置集合属性

    def insert_data(self, user_id, names, texts, embeddings):
        """将数据插入到 Milvus 集合中，并将 user_id 追加到 name 字段。"""
        # 追加 user_id 到每个 name
        names_with_user = [f"{name}_{user_id}" for name in names]

        data = [
            [user_id] * len(names_with_user),  # user_id
            names_with_user,  # name (已追加 user_id)
            texts,  # text
            embeddings  # embedding
        ]
        self.collection.insert(data)  # 插入数据到集合
        print(f"已向集合插入 {len(names_with_user)} 条记录。")  # 输出插入成功消息

    def process_file(self, file_path, file_name, user_id: int, knowledge_base_name: str, descrip: str):
        """
        处理文件：加载内容、分割文本、生成嵌入向量并插入 Milvus。
        """
        # 根据文件扩展名选择适当的加载器
        extension = file_name.split(".")[-1].lower()

        # 根据文件扩展名选择相应的加载器
        if extension == 'txt':
            loader = TextLoader(file_path, encoding='utf8')  # 加载txt文件
        elif extension == 'pdf':
            loader = PyPDFium2Loader(file_path)  # 加载pdf文件
        elif extension == 'docx':
            loader = Docx2txtLoader(file_path)  # 加载docx文件
        elif extension == 'md':
            loader = UnstructuredMarkdownLoader(file_path)
        elif extension == 'csv':
            loader = CSVLoader(file_path)
        else:
            raise ValueError(f"不支持的文件类型：{extension}")  # 抛出不支持文件类型的错误

        # 加载文档
        documents = loader.load()  # 加载文件内容为文档对象

        # 分割文本为块，每个块大小为800个字符，重叠部分为100个字符
        text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)  # 将文档分割为多个块
        texts = [doc.page_content for doc in docs]  # 提取每个文档块的文本内容


        # 确保文本块符合长度要求，强制进行硬性截断
        final_texts = []
        for text in texts:
            if len(text) > 800:
                # 对超长文本进行硬性截断
                for i in range(0, len(text), 800):
                    final_texts.append(text[i:i + 800])
            else:
                final_texts.append(text)

        # 为文本块生成嵌入向量
        embeddings = self.get_embedding(final_texts)

        # 为每个文本块分配文件名称
        names = [file_name] * len(final_texts)

        # 将数据插入 Milvus 集合
        self.insert_data(user_id, names, final_texts, embeddings)
        # 确保数据被持久化到Milvus
        self.collection.flush()  # 刷新数据到集合
        print("数据已刷新到集合。")  # 输出刷新成功消息
        # 记录知识库信息到数据库
        self.log_knowledge_base(
            user_id=user_id,
            name=knowledge_base_name,
            descrip=descrip,
            file_name=file_name,
            file_path=file_path,
            collection_name=self.collection_name,
        )



    def log_knowledge_base(
        self, user_id: int, name: str, descrip: str,
        file_name: str, file_path: str, collection_name: str,upload_type: str = "local",  # 默认为 local

    ):
        """
        记录知识库信息到数据库
        """
        KnowledgeService.record_knowledge_base(
            user_id=user_id,
            name=name,
            descrip=descrip,
            file_name=file_name,
            file_path=file_path,
            collection_name=collection_name,
            upload_type=upload_type,
        )

    def process_file_from_minio(
        self, file_path: str, file_name: str, user_id: int, knowledge_base_name: str, descrip: str
    ):
        """
        处理从 MinIO 下载的文件，并将其数据插入到 Milvus 和 MySQL。
        """
        # 根据文件扩展名选择适当的加载器
        extension = file_name.split(".")[-1].lower()

        if extension == "txt":
            loader = TextLoader(file_path, encoding="utf8")  # 加载 txt 文件
        elif extension == "pdf":
            loader = PyPDFium2Loader(file_path)  # 加载 pdf 文件
        elif extension == "docx":
            loader = Docx2txtLoader(file_path)  # 加载 docx 文件
        else:
            raise ValueError(f"不支持的文件类型：{extension}")

        # 加载文档内容
        documents = loader.load()  # 加载文件内容为文档对象

        # 分割文本为块
        text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)  # 将文档分割为多个块
        texts = [doc.page_content for doc in docs]  # 提取每个文档块的文本内容

        # 确保文本块符合长度要求，强制进行硬性截断
        final_texts = []
        for text in texts:
            if len(text) > 800:
                for i in range(0, len(text), 800):
                    final_texts.append(text[i:i + 800])
            else:
                final_texts.append(text)

        # 生成嵌入向量
        embeddings = self.get_embedding(final_texts)

        # 为每个文本块分配文件名称
        names = [file_name] * len(final_texts)

        # 插入到 Milvus
        self.insert_data(user_id, names, final_texts, embeddings)

        # 刷新 Milvus 集合
        self.collection.flush()

        print(f"文件 '{file_name}' 已成功插入 Milvus。")

        # 记录到 MySQL
        self.log_knowledge_base(
            user_id=user_id,
            name=knowledge_base_name,
            descrip=descrip,
            file_name=file_name,
            file_path=file_path,
            collection_name=self.collection_name,
            upload_type="minio",  # 标记为 MinIO 上传
        )

    def process_file_url(self, url, user_id,name,descrip):
        """
        处理文件：加载url文本内容、分割文本、生成嵌入向量并插入 Milvus。
        """
        loader=WebBaseLoader(url)
        documents = loader.load()  # 加载url为文档对象

        # 分割文本为块，每个块大小为800个字符，重叠部分为100个字符
        #已经分割文本，但可能由于某些原因导致仍然有块超出这个限制。
        # 通过进一步切割确保每个文本块的长度在模型的范围内来解决这个问题
        text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)  # 将文档分割为多个块
        texts = [doc.page_content for doc in docs]  # 提取每个文档块的文本内容

        # 确保文本块符合长度要求，强制进行硬性截断
        final_texts = []
        for text in texts:
            if len(text) > 800:
                # 对超长文本进行硬性截断
                for i in range(0, len(text), 800):
                    final_texts.append(text[i:i + 800])
            else:
                final_texts.append(text)

        # 为文本块生成嵌入向量
        embeddings = self.get_embedding(final_texts)

        # 为每个文本块分配文件名称
        names = [url] * len(final_texts)

        # 将数据插入 Milvus 集合
        self.insert_data(user_id,names, final_texts, embeddings)
        # 确保数据被持久化到Milvus
        self.collection.flush()  # 刷新数据到集合
        print("数据已刷新到集合。")  # 输出刷新成功消息

        # 记录到 MySQL
        self.log_knowledge_base(
            user_id=user_id,
            name=name,
            descrip=descrip,
            file_name=url,
            file_path=None,
            collection_name=self.collection_name,
            upload_type="url",  # 标记为 MinIO 上传
        )


# FastAPI 主程序块
if __name__ == '__main__':

    # 初始化 FastAPI 应用
    app = FastAPI()

    # 初始化 MilvusManager 管理对象
    milvus_manager = MilvusManager(
        host="8.138.133.120",  # Milvus数据库的主机地址
        port="19530",  # Milvus数据库的端口
        collection_name="text_collection_2",  # 集合名称
        embedding_model="text-embedding-v1",  # 使用的嵌入模型
        dashscope_api_key="sk-12ff5fafaba84039b11c37b3b2929186"  # DashScope API 密钥
    )
