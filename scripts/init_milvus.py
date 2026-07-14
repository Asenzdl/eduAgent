# scripts/init_milvus.py
# 执行：python scripts/init_milvus.py
import asyncio

from pymilvus import MilvusClient, DataType
from backend.config import get_settings
from backend.core.milvus_repo import MilvusRepository

config = get_settings()

MILVUS_URI = f"http://{config.milvus_host}:{config.milvus_port}"
VECTOR_DIM = config.milvus_embedding_dim                  # 稠密向量维度
DB_NAME = config.milvus_db_name
COLLECTION_NAME = config.milvus_collection_name  # 单集合，靠 tenant_id 区分租户


def build_schema(client: MilvusClient):
    """构建集合 schema：稠密 + 稀疏双向量 + 标量字段（含 tenant_id）。"""
    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("id",               DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("embedding",        DataType.FLOAT_VECTOR, dim=VECTOR_DIM)     
    schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)              
    schema.add_field("content",          DataType.VARCHAR, max_length=4096)
    schema.add_field("source",           DataType.VARCHAR, max_length=512)
    schema.add_field("tenant_id",        DataType.VARCHAR, max_length=64)          
    schema.add_field("chunk_index",      DataType.INT64)
    schema.add_field("document_id",      DataType.VARCHAR, max_length=64)
    schema.add_field("course_id",        DataType.VARCHAR, max_length=64)
    schema.add_field("source_name",      DataType.VARCHAR, max_length=256)
    schema.add_field("chunk_type",       DataType.VARCHAR, max_length=32)
    schema.add_field("version",          DataType.VARCHAR, max_length=32)
    schema.add_field("updated_at",       DataType.INT64)
    return schema


def build_index_params(client: MilvusClient):
    """构建索引：稠密 HNSW、稀疏 SPARSE_INVERTED、标量 INVERTED。"""
    ip = client.prepare_index_params()
    # 稠密向量：HNSW + COSINE（语义相似度）
    ip.add_index(
        field_name="embedding", 
        index_type="HNSW", 
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 256})
    # 稀疏向量：SPARSE_INVERTED_INDEX + IP（内积）；drop_ratio_build 丢弃最低权重 token 省存储
    ip.add_index(
        field_name="sparse_embedding", 
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP", 
        params={"drop_ratio_build": 0.2})
    # 标量字段：INVERTED 索引，加速 filter 过滤
    ip.add_index(
        field_name="tenant_id", 
        index_type="INVERTED")
    ip.add_index(
        field_name="course_id", 
        index_type="INVERTED")
    return ip


async def main():
    print(f"连接 Milvus：{MILVUS_URI}")
    client = MilvusRepository.get_default_client()

    if DB_NAME not in await client.list_databases():
            await client.create_database(DB_NAME)
            print(f"已新建数据库 {DB_NAME}")
    
    if await client.has_collection(COLLECTION_NAME):
        print(f"🗑️  删除旧集合 '{COLLECTION_NAME}'...")
        await client.drop_collection(COLLECTION_NAME)

    # create_collection 传 index_params 会一并建索引并加载
    await client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=build_schema(client),
        index_params=build_index_params(client),
    )
    print(f"✅ 集合 '{COLLECTION_NAME}' 创建完成（含索引，已加载）")
    print("当前集合：", await client.list_collections())
    print("⚠️  集合已重建，原有数据已清空，请重新运行 build_knowledge_base.py 导入。")

if __name__ == "__main__":
    asyncio.run(main())
