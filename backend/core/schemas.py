import hashlib
import time

from pydantic import BaseModel, Field, model_validator

from enum import StrEnum

class ChunkType(StrEnum):
    TEXT  = "text"
    CODE  = "code"
    TABLE = "table"

class ChunkMetadata(BaseModel):
    source: str
    course_id: str
    document_id: str | None = None     # 可由 source 自动派生
    source_name: str = ""
    chunk_type: ChunkType = ChunkType.TEXT
    chunk_index: int = 0

    @classmethod
    def project_from(cls, data: dict) -> dict:
        """从任意 dict 投影出本模型声明的字段，不做校验。

        用于 Milvus 读路径：entity → metadata dict，
        只保留 ChunkMetadata 定义的字段，其他字段（content、embedding 等）自动丢弃。

        Args:
            data: 可能包含额外字段的源 dict。

        Returns:
            只含 ChunkMetadata 字段的 dict，值直接引用原 dict 对应值。
        """
        return {k: data[k] for k in cls.model_fields if k in data}

    @model_validator(mode='after')
    def _auto_document_id(self):
        """从 source 自动派生 document_id（MD5 前 12 位，幂等）。
        同一文件所有 chunk 的 source 相同 → document_id 相同。
        手动传入 document_id 时跳过，不覆盖。
        """
        if self.document_id is None:
            self.document_id = hashlib.md5(self.source.encode()).hexdigest()[:12]
        return self


class DocumentChunk(ChunkMetadata):
    """
    准备写入 Milvus 的单个文档块，字段与 Milvus Schema 一一对应。

    id:               全局唯一 ID（MD5 of content + document_id + chunk_index）
    content:          chunk 文本（Contextual RAG 模式下含 LLM 生成的上下文描述前缀）
    embedding:        Dense 向量（BGE-M3，1024 维，split 阶段为 None，embed 阶段回填）
    sparse_embedding: Sparse 向量（{token_id: weight}，同上）
    """
    id:                 str | None = None
    content:            str
    embedding:          list[float] | None = None
    sparse_embedding:   dict | None = None
    version:            str
    tenant_id:          str = "tenant_default"
    updated_at:         int = Field(default_factory=lambda: int(time.time()))

    @model_validator(mode='after')
    def _auto_id(self):
        """生成 chunk 全局唯一 ID（MD5）。内容+位置不变则 ID 不变，支持幂等 upsert。
        用 document_id + chunk_index + content 前缀组合，确保：
            - 同一文档不同位置的 chunk 不冲突
            - 内容不变时 ID 稳定（幂等重建时不会重复插入）
        """
        if self.id is None:
            raw = f"{self.document_id}_{self.chunk_index}_{self.content}"
            self.id = hashlib.md5(raw.encode()).hexdigest()
        return self
    
    
if __name__ == "__main__":
    # ── 手动指定 document_id ──
    test = DocumentChunk(
        content="IOC是什么？",
        embedding=[0.0, 0.0, 0.0],
        sparse_embedding={123: 0.5},
        source="backend/agents/qa/data/java-讲义.md",
        course_id="123",
        document_id="456",
        source_name="Java讲义 > 第3章 > 3.1 IOC",
        chunk_type='text',
        chunk_index=1,
        version="1.0",
    )
    print("手动 ID:", test.id)
    print("document_id:", test.document_id)

    # ── 自动派生 document_id ──
    auto = DocumentChunk(
        content="依赖注入的实现方式",
        embedding=[0.1, 0.2, 0.3],
        sparse_embedding={},
        source="backend/agents/qa/data/java-讲义.md",
        course_id="123",
        source_name="Java讲义 > 第3章 > 3.2 DI",
        chunk_type='text',
        chunk_index=2,
        version="1.0",
    )
    print("自动 document_id:", auto.document_id)
    print("自动 ID:", auto.id)

    # ── 同文件自动派生 → document_id 幂等 ──
    auto2 = DocumentChunk(
        content="另一个 chunk",
        embedding=[0.4, 0.5, 0.6],
        sparse_embedding={},
        source="backend/agents/qa/data/java-讲义.md",
        course_id="123",
        source_name="Java讲义 > 第3章 > 3.3 AOP",
        chunk_type='text',
        chunk_index=3,
        version="1.0",
    )
    print("同源 document_id:", auto.document_id, auto2.document_id, "相同:", auto.document_id == auto2.document_id)