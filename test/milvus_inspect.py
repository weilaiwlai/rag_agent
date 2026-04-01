"""
Milvus 数据库检查工具
功能：
1. 列出所有存在的集合 (Collections)
2. 显示每个集合的详细统计信息 (行数、索引状态、字段结构)
3. 检查集合加载状态
"""

from pymilvus import connections, utility, Collection, MilvusException
import json
from datetime import datetime

# 配置 Milvus 连接
MILVUS_HOST = "202.204.62.144"
MILVUS_PORT = "19530"
ALIAS = "default"

def connect_to_milvus():
    """连接到 Milvus"""
    print(f"🔌 正在连接到 Milvus ({MILVUS_HOST}:{MILVUS_PORT})...")
    try:
        connections.connect(alias=ALIAS, host=MILVUS_HOST, port=MILVUS_PORT)
        print("✅ 连接成功!\n")
        return True
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return False

def get_collection_details(collection_name):
    """获取集合详细信息"""
    try:
        # 获取集合对象
        collection = Collection(collection_name)
        
        # 尝试加载集合以获取更多信息 (如果未加载)
        # 注意：频繁加载/释放可能影响性能，这里仅作查看用途
        # collection.load() 
        
        # 基础信息
        details = {
            "name": collection_name,
            "description": collection.description,
            "is_empty": collection.is_empty,
            "num_entities": collection.num_entities,  # 近似行数
            "primary_field": collection.primary_field.name if collection.primary_field else "Unknown",
            "schema": [],
            "indexes": []
        }

        # 字段 Schema
        for field in collection.schema.fields:
            field_info = {
                "name": field.name,
                "type": str(field.dtype),
                "is_primary": field.is_primary,
                "params": field.params
            }
            details["schema"].append(field_info)

        # 索引信息
        for index in collection.indexes:
            index_info = {
                "field_name": index.field_name,
                "params": index.params,
                "dropped": index.dropped
            }
            details["indexes"].append(index_info)
            
        # 统计信息 (如果集合已加载，这会更准确)
        # get_collection_stats returns a dict usually containing 'row_count'
        stats = utility.get_collection_stats(collection_name)
        details["stats"] = stats

        return details

    except Exception as e:
        return {"error": str(e)}

def main():
    if not connect_to_milvus():
        return

    try:
        # 1. 列出所有集合
        collections = utility.list_collections()
        print(f"📚 发现 {len(collections)} 个集合:")
        print("-" * 50)

        if not collections:
            print("   (无集合)")
        
        for idx, name in enumerate(collections):
            print(f"\n[{idx+1}] 集合名称: {name}")
            
            # 2. 获取并显示详细信息
            details = get_collection_details(name)
            
            if "error" in details:
                print(f"   ⚠️ 获取详情失败: {details['error']}")
                continue

            print(f"   📝 描述: {details['description'] or '无'}")
            
            # 行数与状态
            row_count = details.get('stats', {}).get('row_count', 'Unknown')
            print(f"   📊 数据行数: {row_count}")
            
            # 字段结构
            print("   🏗️ 字段结构:")
            for field in details['schema']:
                primary_mark = "🔑 " if field['is_primary'] else "   "
                print(f"      {primary_mark}{field['name']} ({field['type']})")

            # 索引状态
            if details['indexes']:
                print("   🔍 索引:")
                for idx_info in details['indexes']:
                    print(f"      - 字段: {idx_info['field_name']}, 参数: {idx_info['params']}")
            else:
                print("   ⚠️ 无索引 (搜索可能很慢)")

            print("-" * 50)

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
    finally:
        # 断开连接
        connections.disconnect(ALIAS)
        print("\n🔌 连接已断开")

if __name__ == "__main__":
    main()
