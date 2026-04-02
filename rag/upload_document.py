"""
用途：将单个文档加载、切分、生成嵌入并上传到 Milvus 集合。
适用场景：命令行快速导入一个文件到指定集合，便于后续检索。
"""
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFium2Loader, Docx2txtLoader, CSVLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from pymilvus import connections, Collection, CollectionSchema, DataType, FieldSchema, utility
from langchain_milvus import BM25BuiltInFunction, Milvus
import sys
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from utils.config_handler import rag_conf
from model.factory import embed_model

load_dotenv()

FILE_PATH = "data"  #文件路径
COLLECTION_NAME = rag_conf["COLLECTION_NAME"]
MILVUS_HOST = rag_conf["MILVUS_HOST"]
MILVUS_PORT = rag_conf["MILVUS_PORT"]
DASHSCOPE_API_KEY = rag_conf["dashscope_api_key"]
EMBEDDING_MODEL = rag_conf["embedding_model_name"]

class SimpleDocumentUploader:
    def __init__(self, host, port, collection_name, dashscope_api_key, embedding_model):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dashscope_api_key = dashscope_api_key
        self.embedding_model = embedding_model
        
        # 初始化嵌入模型
        self.embeddings = embed_model
        
        # 连接到Milvus
        self.connect_milvus()
        
        # 删除现有集合（如果存在）
        self.drop_collection()
        
        # 初始化向量存储
        self.vectorstore = None
        
    def connect_milvus(self):
        """连接到Milvus数据库"""
        connections.connect("default", host=self.host, port=self.port)
        print(f"已连接到 Milvus，地址为 {self.host}:{self.port}")
        
    def get_embedding(self, texts):
        """生成文本嵌入向量"""
        return self.embeddings.embed_documents(texts)
        
    def process_file(self, file_path):
        """处理文件并上传到Milvus"""
        if not os.path.exists(file_path):
            print(f"错误：文件不存在 {file_path}")
            return False
            
        # 获取文件名和扩展名
        file_name = os.path.basename(file_path)
        extension = file_name.split(".")[-1].lower()
        
        # 根据文件类型选择加载器
        if extension == 'txt':
            loader = TextLoader(file_path, encoding='utf8')
        elif extension == 'pdf':
            loader = PyPDFium2Loader(file_path)
        elif extension == 'docx':
            loader = Docx2txtLoader(file_path)
        elif extension == 'csv':
            loader = CSVLoader(file_path)
        else:
            print(f"不支持的文件类型：{extension}")
            return False
            
        try:
            # 加载文档
            documents = loader.load()
            print(f"成功加载文档：{file_name}")
            
            # 添加源文件名到元数据
            for doc in documents:
                doc.metadata['source'] = file_name
            
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = text_splitter.split_documents(documents)
            
            # 限制每个文档块的长度
            for doc in docs:
                if len(doc.page_content) > 2000:
                    doc.page_content = doc.page_content[:2000]
            
            # 使用Milvus.from_documents创建向量存储并添加文档
            self.vectorstore = Milvus.from_documents(
                documents=docs,
                embedding=self.embeddings,
                builtin_function=BM25BuiltInFunction(),
                collection_name=self.collection_name,
                connection_args={"uri": f"http://{self.host}:{self.port}"},
                vector_field=["dense", "sparse"],
                drop_old=True  # 删除已存在的同名集合
            )
            
            # 设置BM25内置函数用于混合搜索
            self.vectorstore.builtin_function = BM25BuiltInFunction()
            
            print(f"文档 '{file_name}' 上传成功！共 {len(docs)} 个文档块")
            return True
            
        except Exception as e:
            print(f"处理文件时出错：{e}")
            import traceback
            traceback.print_exc()
            return False
        

    
    def drop_collection(self):
        """删除集合"""
        try:
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                print(f"集合 '{self.collection_name}' 已被删除")
            else:
                print(f"集合 '{self.collection_name}' 不存在")
        except Exception as e:
            print(f"删除集合时出错：{e}")

def get_supported_files(folder_path):
    """获取文件夹中所有支持的文件类型"""
    supported_extensions = ['.txt', '.pdf', '.docx', '.csv']
    files = []
    
    for root, dirs, filenames in os.walk(folder_path):
        for filename in filenames:
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in supported_extensions:
                files.append(os.path.join(root, filename))
    
    return files

def main():
    """主函数"""
    print("初始化文档上传器...")
    
    uploader = SimpleDocumentUploader(
        host=MILVUS_HOST,
        port=MILVUS_PORT,
        collection_name=COLLECTION_NAME,
        dashscope_api_key=DASHSCOPE_API_KEY,
        embedding_model=EMBEDDING_MODEL
    )
    
    # 检查FILE_PATH是文件还是文件夹
    if os.path.isfile(FILE_PATH):
        # 单个文件处理
        print(f"开始处理文档：{FILE_PATH}")
        success = uploader.process_file(FILE_PATH)
        
        if success:
            print("文档上传完成！")
        else:
            print("文档上传失败！")
    elif os.path.isdir(FILE_PATH):
        # 文件夹处理 - 遍历所有支持的文件
        print(f"开始处理文件夹：{FILE_PATH}")
        supported_files = get_supported_files(FILE_PATH)
        
        if not supported_files:
            print(f"在文件夹 {FILE_PATH} 中没有找到支持的文件类型 (.txt, .pdf, .docx, .csv)")
            return
        
        print(f"找到 {len(supported_files)} 个支持的文件")
        
        success_count = 0
        for file_path in supported_files:
            print(f"正在处理文件: {file_path}")
            if uploader.process_file(file_path):
                success_count += 1
        
        print(f"\n批量上传完成！成功上传 {success_count}/{len(supported_files)} 个文件")
    else:
        print(f"错误：路径 {FILE_PATH} 不存在")
        return
    
    # 显示集合信息
    try:
        collection = Collection(name=COLLECTION_NAME)
        count = collection.num_entities
        print(f"集合 '{COLLECTION_NAME}' 中共有 {count} 条记录")
    except Exception as e:
        print(f"获取集合信息时出错：{e}")

if __name__ == "__main__":
    main()