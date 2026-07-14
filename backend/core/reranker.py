# backend/core/reranker.py
from __future__ import annotations

import asyncio
import os

os.environ["ACCELERATE_USE_META_DEVICE"] = "0"

from dataclasses import dataclass

import torch
from sentence_transformers import CrossEncoder

from backend.config import get_settings
from backend.core.logger import get_logger

logger = get_logger(__name__)
RERANK_MAX_INPUT_CHARS = 512   # 截断过长文档，防止超出 CrossEncoder max_length=512


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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = get_settings().reranker_model_path
        self._model = CrossEncoder(model_path, device=device, max_length=512)
        logger.info("reranker.loaded", model_path=model_path, device=device)
        
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
            (query, doc["content"][:RERANK_MAX_INPUT_CHARS])
            for doc in documents
        ]

        # CrossEncoder 默认 sigmoid 激活，predict() 直接输出 [0, 1] 概率
        scores: list[float] = self._model.predict(pairs).tolist()

        ranked = sorted(
            [
                RankedDocument(
                    content=documents[i]["content"],
                    score=scores[i],
                    original_index=i,
                    metadata=documents[i]["metadata"],
                )
                for i in range(len(documents))
            ],
            key=lambda x: x.score,
            reverse=True,
        )[:top_k]

        confidence = ranked[0].score if ranked else 0.0

        logger.info(
            "rerank_with_confidence.done",
            candidates=len(documents),
            top_k=top_k,
            confidence=round(confidence, 4),
        )

        return ranked, confidence

if __name__ == "__main__":
    from rich.pretty import pprint
    from backend.core.embedding import BGEMEmbedder
    from backend.core.milvus_repo import MilvusRepository
    
    async def main():
        embedder = BGEMEmbedder.get_instance()
        query = "商品聚合多模态大模型项目主要讲的是什么内容"
        dense, sparse = embedder.encode_query(query)
        milvus_repo = MilvusRepository.from_settings()
        documents = await milvus_repo.hybrid_search(dense, sparse, top_k=5)
        reranker = BGEReranker.get_instance()
        print(reranker)
        ranked, confidence = reranker.rerank_with_confidence(query=query, documents=documents)
        pprint(ranked)
        pprint(confidence)
        
    asyncio.run(main())
    
    
