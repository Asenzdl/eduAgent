# backend/core/reranker.py

import os

os.environ["ACCELERATE_USE_META_DEVICE"] = "0"

from dataclasses import dataclass

import torch
from sentence_transformers import CrossEncoder

from backend.config import get_settings
from backend.core.logger import get_logger

logger = get_logger(__name__)
RERANK_MAX_INPUT_CHARS = 1200   # 截断过长文档，防止超出 CrossEncoder max_length=512


@dataclass
class RankedDocument:
    """精排后的单个文档结果"""
    content:        str    # 文档文本
    score:          float  # BGE-Reranker 输出的相关性概率 [0, 1]
    original_index: int    # 在原始召回列表中的位置（0 起）
    metadata:       dict   # 来源元数据（source_name / chunk_type / course_id 等）


class BGEReranker:
    """
    BGE-Reranker-v2-m3 精排服务（单例）。

    对 Hybrid 召回的候选文档做 CrossEncoder 精排，
    直接返回 [0, 1] 置信度，无需额外归一化。

    用法：
        reranker = BGEReranker.get_instance()
        docs, confidence = reranker.rerank_with_confidence(
            query="什么是 Spring IOC？",
            documents=candidates,
            top_k=3,
        )
    """

    _instance: "BGEReranker" | None = None

    def __init__(self):
        settings = get_settings()
        model_id = settings.reranker_model_path
        device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("reranker.loading", model_id=model_id, source=settings.reranker_model_source, device=device)
        self._model = CrossEncoder(model_id, device=device, max_length=512)
        logger.info("reranker.loaded", model_id=model_id)

    @classmethod
    def get_instance(cls) -> "BGEReranker":
        """获取单例，首次调用时加载模型"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def rerank_with_confidence(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 3,
    ) -> tuple[list[RankedDocument], float]:
        """
        精排并返回置信度。

        Args:
            query:     用户 Query
            documents: 候选文档列表，每项含 "content" 字段
            top_k:     返回文档数量

        Returns:
            (ranked_docs, confidence)
            ranked_docs:  按相关性降序排列的 RankedDocument，长度 <= top_k
            confidence:   Top-1 文档的 BGE 相关性概率 [0, 1]，
                          ≥ 0.75 → 高置信度，直接走 LLM 生成；
                          < 0.75 → 低置信度，触发 Web 兜底
        """
        if not documents:
            return [], 0.0

        # CrossEncoder 输入：(query, document) 对，截断过长文档
        pairs = [
            (query, (doc.get("content") or "")[:RERANK_MAX_INPUT_CHARS])
            for doc in documents
        ]

        # CrossEncoder 默认 sigmoid 激活，predict() 直接输出 [0, 1] 概率
        scores: list[float] = self._model.predict(pairs).tolist()

        ranked = sorted(
            [
                RankedDocument(
                    content=documents[i].get("content", ""),
                    score=scores[i],
                    original_index=i,
                    metadata=documents[i].get("metadata", {}),
                )
                for i in range(len(documents))
            ],
            key=lambda x: x.score,
            reverse=True,
        )

        top_results = ranked[:top_k]
        confidence = top_results[0].score if top_results else 0.0

        logger.info(
            "reranker.done",
            candidates=len(documents),
            top_k=top_k,
            confidence=round(confidence, 4),
        )

        return top_results, confidence

def retrieve(
    query: str,
    tenant_id: str,
    course_id: str | None,
    recall_top_k: int = 10,
    rerank_top_k: int = 3,
) -> tuple[list[RankedDocument], float]:
    """
    Hybrid 召回 → BGE 精排一体化 Pipeline。

    调用方只需传 Query 文本和租户/课程参数，向量化在内部完成。

    Args:
        query:        用户 Query 文本
        tenant_id:    租户 ID（Milvus 过滤条件）
        course_id:    课程 ID（可选，进一步缩小检索范围）
        recall_top_k: Hybrid 召回数量（默认 10，送给精排）
        rerank_top_k: 精排后返回数量（默认 3，送给 LLM）

    Returns:
        (ranked_docs, confidence)
        ranked_docs:  精排后 Top-rerank_top_k 文档
        confidence:   Top-1 文档的 BGE 置信度 [0, 1]
    """
    from backend.core.embedding import BGEMEmbedder
    from backend.core.milvus_repo import MilvusRepository  # 延迟导入，避免循环依赖

    # ── 第一步：向量化（在 Pipeline 内部完成，调用方无需感知）──────
    embedder = BGEMEmbedder.get_instance()
    dense_vec, sparse_vec = embedder.encode_query(query)

    # ── 第二步：Hybrid 召回 ─────────────────────────────────────────
    milvus_repo = MilvusRepository.from_settings()
    filters = milvus_repo._build_filter(tenant_id, course_id)
    candidates = milvus_repo._hybrid_search(
        query_embedding=dense_vec,
        query_sparse=sparse_vec,
        top_k=recall_top_k,
        filters=filters,
    )

    if not candidates:
        logger.info("retrieve.empty", query_preview=query[:50])
        return [], 0.0

    # ── 第三步：精排 ────────────────────────────────────────────────
    reranker = BGEReranker.get_instance()
    return reranker.rerank_with_confidence(query, candidates, top_k=rerank_top_k)


    
if __name__ == "__main__":
    from rich.pretty import pprint
    from backend.core.embedding import BGEMEmbedder
    from backend.core.milvus_repo import MilvusRepository
    model = BGEMEmbedder.get_instance()
    text = "商品聚合多模态大模型项目主要讲的是什么内容"
    dense, sparse = model.encode_query(text)
    kb = MilvusRepository.from_settings()
    documents = kb._hybrid_search(dense, sparse, top_k=5)
    model = BGEReranker.get_instance()
    top_results, confidence = model.rerank_with_confidence(query=text, documents=documents)
    pprint(top_results)
    pprint(confidence)
    
