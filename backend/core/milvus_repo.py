from pymilvus import AnnSearchRequest, AsyncMilvusClient, WeightedRanker
from pymilvus.exceptions import MilvusException

from backend.config import get_settings
from backend.core.logger import get_logger
from backend.core.schemas import ChunkMetadata, DocumentChunk

logger = get_logger(__name__)


class MilvusRepository:
    """Milvus 数据访问仓库

    注意：底层 gRPC 连接由库内部 AsyncConnectionManager 自动按 address|token 去重复用。
    """

    def __init__(self, client: AsyncMilvusClient, collection_name: str):
        self._client = client
        self._collection_name = collection_name
        self._loaded = False
        logger.info("init", collection_name=self._collection_name)

    @classmethod
    def from_settings(cls) -> "MilvusRepository":
        """从全局配置创建仓库实例，适用于**独立脚本**（无 FastAPI 注入可用时）。

        QA Agent 等 FastAPI 链路请使用 ``Depends(get_milvus_repo)`` 注入。
        """
        settings = get_settings()
        client = AsyncMilvusClient(
            uri=f"http://{settings.milvus_host}:{settings.milvus_port}",
            db_name=settings.milvus_db_name,
        )

        logger.info("from_settings", uri=settings.milvus_host, db_name=settings.milvus_db_name)
        return cls(client, collection_name=settings.milvus_collection_name)
    
    @classmethod
    def get_default_client(cls) -> AsyncMilvusClient:
        """获取默认 Milvus 客户端，用于独立脚本调用。"""
        return cls.from_settings()._client

    # ——————————————————————— 写入 ———————————————————————

    async def insert_chunks(self, chunks: list[DocumentChunk]) -> int:
        """批量插入文档块（仅用于新数据，PK 冲突则报错）。

        与 upsert_chunks 的区别：
        - insert：PK 不存在则插入，已存在则报错（快速失败）
        - upsert：PK 存在则覆盖，不存在则插入（额外 PK 检查开销）

        调用前确保已按 document_id 删除旧数据（delete_document_chunks），
        否则相同 PK 的重复插入会触发 PrimaryKeyError。

        Args:
            chunks: 待插入的 DocumentChunk 列表

        Returns:
            成功插入的 chunk 数量

        Raises:
            MilvusException: 插入失败（含 PK 冲突）
        """
        if not chunks:
            return 0
        try:
            data = [chunk.model_dump() for chunk in chunks]
            result = await self._client.insert(
                collection_name=self._collection_name,
                data=data,
            )
            logger.info("insert_chunks", requested=len(chunks),
                        inserted=result.get("insert_count", 0))
            return len(chunks)
        except MilvusException as e:
            logger.error("insert_chunks", count=len(chunks), error=str(e), exc_info=True)
            raise

    async def upsert_chunks(self, chunks: list[DocumentChunk]) -> int:
        """批量写入文档块（Upsert：primary key 存在则更新，不存在则插入）。
        """
        if not chunks:
            return 0
        try:
            data = [chunk.model_dump() for chunk in chunks]
            result = await self._client.upsert(
                collection_name=self._collection_name,
                data=data,
            )
            logger.info("upsert_chunks", requested=len(chunks),
                        upserted=result.get("upsert_count", 0))
            return len(chunks)
        except MilvusException as e:
            logger.error("upsert_chunks", count=len(chunks), error=str(e), exc_info=True)
            raise

    # ——————————————————————— 删除 ———————————————————————

    async def delete_document_chunks(self, document_ids: list[str]) -> None:
        """批量删除指定文档的所有 chunk。

        Args:
            document_ids: 要删除的文档 ID 列表。空列表时跳过。
        """
        if not document_ids:
            return
        try:
            expr = "document_id in {document_ids}"
            filter_params = {"document_ids": document_ids}
            result = await self._client.delete(
                collection_name=self._collection_name,
                filter=expr,
                filter_params=filter_params,
            )
            logger.info("delete_document_chunks", requested_doc=len(document_ids),
                        deleted_chunks=result.get("delete_count", 0))
        except MilvusException as e:
            logger.error("delete_document_chunks", ids=document_ids, error=str(e), exc_info=True)
            raise
        
    
    # ── 检索配置 ─────────────────────────────────────────────
    VECTOR_TOP_K = 10    # Hybrid 召回的候选数量，传给 Reranker 精排
    ANN_EF       = 64    # HNSW 搜索时候选集大小（精度/速度平衡点）

    async def _ensure_loaded(self) -> None:
        """首次搜索前加载集合到内存，后续调用直接跳过。"""
        if self._loaded:
            return
        try:
            await self._client.load_collection(self._collection_name)
            self._loaded = True
        except MilvusException as e:
            logger.error("collection.load_failed",
                         collection=self._collection_name, error=str(e))
            raise

    async def _hybrid_search(
        self,
        dense_vector: list[float],
        sparse_vector: dict,
        top_k: int = VECTOR_TOP_K,
        expr: str | None = None,
        expr_params: dict | None = None
    ) -> list[dict]:
        """
        对 knowledge_domain 做 Hybrid 检索（Dense + Sparse → WeightedRanker 融合）。

        两个 AnnSearchRequest 分别构造 Dense 和 Sparse 检索请求，
        由 Milvus 在服务端并行执行后，用 WeightedRanker 加权融合排序。

        Args:
            query_embedding: Dense Query 向量（1024 维，来自 encode_query）
            query_sparse:    Sparse Query 向量（{token_id: weight}，来自 encode_query）
            top_k:           每路召回数量（融合后同样取 top_k）
            filters:         Milvus bool 表达式，如 'tenant_id == "xxx"'

        Returns:
            候选文档列表，每项含 "content" / "score" / "metadata"。
            score 是 WeightedRanker 的加权排序信号，不是概率，
            直接交给 5.7 节的 Reranker 做精细打分。
        """
        try:
            await self._ensure_loaded()

            # ── Dense ANN 检索请求 ─────────────────────────────────────
            # COSINE 度量匹配 BGE-M3 dense 向量（L2 归一化后等价于余弦相似度）
            # ef=64：HNSW 搜索时的候选集大小，越大精度越高，64 是精度/速度平衡点
            dense_req = AnnSearchRequest(
                data=[dense_vector],
                anns_field="embedding",
                param={
                    "metric_type": "COSINE",
                    "params": {"ef": self.ANN_EF},
                },
                limit=top_k,
                expr=expr,
                expr_params=expr_params,
            )

            # ── Sparse 关键词检索请求 ──────────────────────────────────
            # IP（内积）是 BGE-M3 lexical_weights 的标准度量
            sparse_req = AnnSearchRequest(
                data=[sparse_vector],
                anns_field="sparse_embedding",
                param={"metric_type": "IP"},
                limit=top_k,
                expr=expr,
                expr_params=expr_params,
            )

            output_fields = [
                "content", "source", "source_name", "chunk_type",
                "course_id", "document_id", "chunk_index",
            ]

            # ── WeightedRanker(0.7, 0.3) ──────────────────────────────
            # 第一个权重对应第一个请求（Dense），第二个对应第二个请求（Sparse）
            # 两路结果在 Milvus 服务端并行检索，融合后返回
            results = await self._client.hybrid_search(
                collection_name=self._collection_name,
                reqs=[dense_req, sparse_req],
                ranker=WeightedRanker(0.7, 0.3),
                limit=top_k,
                output_fields=output_fields,
            )

            # ── 解析结果 ───────────────────────────────────────────────
            # MilvusClient 的 hybrid_search 结果用 distance 字段存融合后的分数
            # 这个分数是排序信号，不是概率，不做任何额外处理，直接传给 Reranker
            candidates = []
            for hit in results[0]:
                entity = hit["entity"]
                candidates.append({
                    "content":  entity.get("content") or "",
                    "score":    hit.get("distance") or 0.0,
                    "metadata": ChunkMetadata.model_validate(entity),
                })

            logger.info("hybrid_search.done", candidates=len(candidates))
            return candidates

        except MilvusException as e:
            logger.error("hybrid_search.failed", error=str(e), exc_info=True)
            return []


if __name__ == "__main__":
    from backend.core.embedding import BGEMEmbedder
    from rich.pretty import pprint
    import asyncio
    embedder = BGEMEmbedder.get_instance()
    query = "商品聚合多模态大模型主要讲的什么内容"
    dense, sparse = embedder.encode_query(query)
    
    async def main():
        repo = MilvusRepository.from_settings()
        expr = "course_id == {course_id} and tenant_id == {tenant_id}"
        expr_params = {"course_id": "01", "tenant_id": "tenant_default"}
        candidates = await repo._hybrid_search(
            dense_vector=dense,
            sparse_vector=sparse,
            top_k=3,
            expr=expr,
            expr_params=expr_params
        )
        pprint(candidates)
        """
        [
            [
                {
                    'id': 'a11906da68ef3debcefc3a8517145cce',
                    'distance': 0.7802555561065674,
                    'entity': {
                        'content': '该文本块介绍了商品聚合多模态大模型项目的核心目标与优势，聚焦于解决电商数据治理中的痛点。\n\n### 2. ...
                        'source_name': 'sample2 > 商品聚合多模态大模型微调原理与实战 > 二、项目介绍 > 2. 项目目标及优势',
                        'chunk_type': 'text',
                        'course_id': '01',
                        'document_id': 'ed456848e6e0',
                        'chunk_index': 13
                    }
                },
                ...,
            ]
        ]
        """
        
    asyncio.run(main())
    