
from dataclasses import dataclass
from pathlib import Path
from typing import Type

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import DirectoryLoader

from backend.core import get_logger
import logging

from backend.core.llm_factory import get_llm
from backend.core.schemas import DocumentChunk
logging.getLogger("pypdf").setLevel(logging.ERROR)
from langchain_core.documents import Document
import asyncio


from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from backend.core.embedding import BGEMEmbedder
from langchain_core.document_loaders import BaseLoader
from backend.core.milvus_repo import MilvusRepository


logger = get_logger(__name__)

DATA_DIR = "backend/agents/qa/data"

MARKDOWN_HEADERS = [
    ("#",   "H1"),
    ("##",  "H2"),
    ("###", "H3"),
    ("####", "H4"),
]

SEPARATORS = ["\n\n", "\n", "。", "，", " ", ""]


# ── 常量 ─────────────────────────────────────────────────────

MAX_CONTEXT_CONCURRENCY = 5    # Contextual 上下文生成的最大并发 LLM 请求数
BATCH_SIZE = 12                # BGE-M3 批量推理大小（12 = 速度与显存的经验平衡点）
TENANT_ID = "tenant_default"
VERSION = "1.0"

CONTEXTUAL_CHUNK_PROMPT = """\
<document>
{{document_text}}
</document>

以下是需要在整个文档中定位的 chunk：
<chunk>
{{chunk_content}}
</chunk>

请用一句简洁且精炼的中文上下文，用于将此文本块置于整篇文档的语境之中，以便改善该 chunk 的检索效果。
请仅输出这段精炼的上下文，不要附带任何其他内容。"""

"""
请用一句简洁且精炼的中文，描述这段内容在整个文档中的位置和作用，以便改善检索效果。
只输出这一句描述，不要附带任何前缀或标签。"""


def load_document(file_path: str, **kwargs) -> Type[BaseLoader]:
    """统一文档加载入口，根据扩展名选择 Loader"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{file_path}")
    ext = path.suffix.lower()
    if ext == ".pdf":
        params = kwargs.get(".pdf", {})
        return PyPDFLoader(file_path, **params)
    elif ext in (".md", ".markdown", ".txt"):
        params = kwargs.get(".md", {})
        return TextLoader(file_path, encoding="utf-8", **params)
    else:
        raise ValueError(
            f"不支持的文件类型：{ext}\n"
            f"当前支持：.pdf / .md / .markdown\n"
            f"提示：可用 markitdown 将 Word/PPT 转换为 .md 后再导入"
        )

# ── DocumentGroup 数据类 ──────────────────────────────────

@dataclass
class DocumentGroup:
    """一个文件的分块结果——极简容器。"""
    raw_doc: Document                  # 全文（Contextual RAG 拼接用）
    chunks: list[DocumentChunk]        # 不含向量的 draft chunk

def split_markdown(
    docs: list[Document],
    course_id: str,
    version: str = VERSION,
    tenant_id: str = TENANT_ID,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[DocumentGroup]:
    """Markdown 文档分块，直接产出 DocumentChunk（embedding=None），装入 DocumentGroup。"""
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=MARKDOWN_HEADERS,
        strip_headers=False,
    )
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        length_function=len,
    )

    def _wrap_chunk(raw: Document, idx: int) -> DocumentChunk:
        """将 raw Document 包装为 DocumentChunk（embedding=None）。"""
        title_content = [raw.metadata[h] for _, h in MARKDOWN_HEADERS if raw.metadata.get(h)]
        source_name = f"{filename} > {' > '.join(title_content)}" if title_content else filename
        return DocumentChunk(
            source=source, course_id=course_id, content=raw.page_content,
            source_name=source_name, chunk_type=raw.metadata.get("chunk_type", "text"),
            chunk_index=idx, version=version, tenant_id=tenant_id,
        )

    groups: list[DocumentGroup] = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        filename = Path(source).stem if source else "未知文件"

        # 按标题切分
        sections = markdown_splitter.split_text(doc.page_content)
        for section in sections:
            section.metadata.update(**doc.metadata)

        # 超长块再按文本切分 → 包装为 DocumentChunk
        chunks = [
            _wrap_chunk(raw, idx)
            for idx, raw in enumerate(recursive_splitter.split_documents(sections))
        ]

        groups.append(DocumentGroup(chunks=chunks, raw_doc=doc))

    total = sum(len(g.chunks) for g in groups)
    logger.info(f"  分块完成：{len(docs)} 个文件 → {total} 个 chunk")
    return groups


def embed_chunks(
    groups: list[DocumentGroup],
    batch_size: int = BATCH_SIZE,
) -> list[DocumentChunk]:
    """
    集中嵌入：摊平所有 DocumentChunk → BGE-M3 批量推理 → 回填 embedding / sparse_embedding。

    Args:
        groups:     split_markdown 返回的 DocumentGroup 列表
        batch_size: BGE-M3 内部批大小（控制显存/内存峰值）

    Returns:
        list[DocumentChunk]，embedding 和 sparse_embedding 已就位，可直接写入 Milvus
    """
    all_chunks: list[DocumentChunk] = []
    texts: list[str] = []
    for g in groups:
        for c in g.chunks:
            all_chunks.append(c)
            texts.append(c.content)
    if not all_chunks:
        return []

    embedder = BGEMEmbedder.get_instance()

    logger.info(f"嵌入 {len(texts)} 个 chunk ...")
    dense_vecs, sparse_vecs = embedder.encode(texts, batch_size=batch_size)

    for chunk, dense, sparse in zip(all_chunks, dense_vecs, sparse_vecs):
        chunk.embedding = dense
        chunk.sparse_embedding = sparse

    logger.info(f"嵌入完成：{len(all_chunks)} 个 DocumentChunk")
    return all_chunks



        
async def add_context(
    groups: list[DocumentGroup],
    concurrency: int = MAX_CONTEXT_CONCURRENCY,
) -> list[DocumentGroup]:
    """
    Contextual RAG：并发为所有 DocumentGroup 的 chunk 生成上下文描述，拼接到 chunk 文本前方。

    拼接后格式：
        "<上下文描述一句话>\\n\\n<原始 chunk 文本>"

    回到 build_pipeline 后一律做嵌入（embed_chunks），向量同时编码"在哪里"和"说了什么"两层信息。

    Args:
        groups:        split_markdown 返回的 DocumentGroup 列表
        concurrency:   最大并发 LLM 请求数（默认 5，防止触发 API 限流）

    Returns:
        groups（chunks 已被就地修改）
    """
    if not groups:
        return groups

    llm = get_llm("qa")
    semaphore = asyncio.Semaphore(concurrency)

    async def _gen_context(content: str, full_text: str, source: str) -> str:
        """为单个 chunk 生成上下文描述。失败时返回空字符串，chunk 保持原始文本。"""
        async with semaphore:
            try:
                from langchain_core.messages import HumanMessage
                from langchain_core.prompts import PromptTemplate
                prompt = PromptTemplate.from_template(
                    CONTEXTUAL_CHUNK_PROMPT, template_format="jinja2"
                ).format(document_text=full_text, chunk_content=content)
                resp = await llm.ainvoke([HumanMessage(content=prompt)])
                return resp.text.strip()
            except Exception as e:
                logger.warning(f"上下文生成失败 [{source}]：{e}")
                return ""

    # 全局并发：所有 chunk 任务提交给同一把 semaphore
    contexts = await asyncio.gather(*[
        _gen_context(c.content, g.raw_doc.page_content, c.source)
        for g in groups for c in g.chunks
    ])

    # 按 chunks 顺序回填
    total_enriched = 0
    idx = 0
    for g in groups:
        for chunk in g.chunks:
            ctx = contexts[idx]
            if ctx:
                chunk.content = f"{ctx}\n\n{chunk.content}"
                total_enriched += 1
            idx += 1

    logger.info(f"上下文增强完成：{total_enriched}/{idx} 个 chunk 已添加描述")
    return groups


async def write_to_milvus(doc_chunks: list[DocumentChunk], milvus_repo: MilvusRepository) -> None:
    """将 embed_chunks() 产出的 DocumentChunk 列表写入 Milvus。

    收集所有涉及到的 document_id → 批量删除 → 一次性插入。
    保证文档更新时旧 chunk 零残留。
    """
    if not doc_chunks:
        logger.warning("无 chunk 可写入，跳过")
        return

    doc_ids = {c.document_id for c in doc_chunks}
    logger.info("清理文档旧版本", count=len(doc_ids))
    await milvus_repo.delete_document_chunks(list(doc_ids))

    written_count = await milvus_repo.insert_chunks(doc_chunks)
    logger.info("写入完成", written_count=written_count)


async def build_pipeline(
    course_id: str,
    path: str = DATA_DIR,
    glob: str = "**/*.md",
    tenant_id: str = TENANT_ID,
    version: str = VERSION,
    use_context: bool = True,
) -> None:
    """
    知识库建库入口——从目录加载文档到写入 Milvus。

    Pipeline:
        load → split_markdown（产出 DocumentChunk, embedding=None）
              → add_context（可选，修改 chunk.content）
              → embed_chunks（回填 embedding / sparse_embedding）
              → write_to_milvus

    Args:
        course_id:   所属课程 ID
        path:        文档目录路径
        glob:        文件匹配模式
        tenant_id:   租户 ID
        version:     课程版本号
        use_context: 是否启用 Contextual RAG 上下文增强
    """
    loader = DirectoryLoader(
        path=path, glob=glob,
        loader_cls=load_document,
        recursive=True, show_progress=True,
    )
    docs = loader.load()
    if not docs:
        logger.warning("无文档可处理，跳过")
        return

    # Step 1：分块 → 直接产出 DocumentChunk（含 source_name，auto document_id）
    groups = split_markdown(docs, course_id, version=version, tenant_id=tenant_id)

    # Step 2：Contextual RAG——并发回填 chunk.content 前缀
    if use_context:
        await add_context(groups)

    # Step 3：集中嵌入——回填 embedding / sparse_embedding
    embedded = embed_chunks(groups)

    # Step 4：写入 Milvus（内部按 document_id 删旧插新）
    repo = MilvusRepository.from_settings()
    await write_to_milvus(embedded, repo)


if __name__ == "__main__":
    asyncio.run(build_pipeline(
        course_id="01",
        glob="sample2.md",
    ))

