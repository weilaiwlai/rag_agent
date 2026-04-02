from pymilvus import connections, Collection

connections.connect(alias="default", host="202.204.62.144", port="19530")

coll = Collection("agent_rag")
print(coll.schema)        # 看字段结构
print(coll.indexes)       # 看索引
print(coll.num_entities)   # 看数据量