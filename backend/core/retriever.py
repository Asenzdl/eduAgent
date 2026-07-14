# backend/core/retriever.py
"""检索 → 精排一体化编排。

将向量化、Hybrid 检索、精排组装成可复用的流水线函数，
调用方只需传 Query 文本和租户/课程范围，不感知内部组件。
"""

from backend.core.embedding import BGEMEmbedder
from backend.core.logger import get_logger
from backend.core.milvus_repo import MilvusRepository
from backend.core.reranker import BGEReranker, RankedDocument
from backend.core.schemas import ChunkMetadata

logger = get_logger(__name__)


async def _search(
    query: str,
    tenant_id: str = "tenant_default",
    course_id: str | None = None,
    top_k: int = 10,
) -> list[dict]:
    """文本 Query → candidates（检索 + 解析，不含精排）。

    Args:
        query:     用户提问文本（如 "什么是 Spring IOC？"）
        tenant_id: 租户 ID
        course_id: 课程 ID（可选）
        top_k:     召回数量（默认 10，送给精排用）

    Returns:
        [{"content": str, "metadata": dict}, ...]
        下游 Reranker 可直接消费。
    """
    embedder = BGEMEmbedder.get_instance()
    dense_vec, sparse_vec = embedder.encode_query(query)

    # ── 构建 Milvus 标量过滤表达式（参数化绑定，防注入）──────────────
    expr_params: dict[str, str] = {"tenant_id": tenant_id}
    expr = "tenant_id == {tenant_id}"
    if course_id:
        expr += " and course_id == {course_id}"
        expr_params["course_id"] = course_id

    repo = MilvusRepository.from_settings()
    entities = await repo.hybrid_search(dense_vec, sparse_vec, top_k=top_k, expr=expr, expr_params=expr_params)
    return [
        {"content": e["content"], "metadata": ChunkMetadata.project_from(e)}
        for e in entities
    ]


async def retrieve(
    query: str,
    tenant_id: str = "tenant_default",
    course_id: str | None = None,
    recall_top_k: int = 10,
    rerank_top_k: int = 3,
) -> tuple[list[RankedDocument], float]:
    """完整流水线：检索 → 精排。

    比 search() 多一步精排，直接返回排序后的结果和置信度，
    是 QA Agent 等上层模块的入口。

    Args:
        query:        用户 Query 文本
        tenant_id:    租户 ID
        course_id:    课程 ID（可选，进一步缩小检索范围）
        recall_top_k: Hybrid 召回数量（默认 10）
        rerank_top_k: 精排后返回数量（默认 3）

    Returns:
        (ranked_docs, confidence)
    """
    candidates = await _search(query, tenant_id, course_id, top_k=recall_top_k)

    if not candidates:
        logger.info("retrieve.empty", query_preview=query[:50])
        return [], 0.0

    reranker = BGEReranker.get_instance()
    return reranker.rerank_with_confidence(query, candidates, top_k=rerank_top_k)


if __name__ == "__main__":
    import asyncio
    from rich.pretty import pprint

    async def main():
        ranked_docs, confidence = await retrieve("商品聚合多模态大模型项目主要讲的是什么内容", course_id="01")
        pprint({"ranked_docs": ranked_docs, "confidence": confidence})
    
    asyncio.run(main())
