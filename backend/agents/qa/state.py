# backend/agents/qa/state.py
#
# QA Agent 的 LangGraph State 与 Context 定义。
#
# QAState   — 可变的状态，节点之间传递演化。
# QAContext — 不可变的运行依赖，通过 LangGraph context API（v0.6+）注入。

from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class Intent(StrEnum):
    """Query 意图分类：是否需要 RAG 检索。"""
    GENERAL = "GENERAL"         # 闲聊/打招呼/问时间，跳过 RAG
    SPECIALIZED = "SPECIALIZED" # 课程相关问题，进入检索链


class RAGStrategy(StrEnum):
    """知识库检索策略。"""
    PRECISE = "PRECISE"  # 明确技术点 → 直接向量检索
    VAGUE   = "VAGUE"    # 模糊表达    → HyDE 语义扩充后再检索
    BROAD   = "BROAD"    # 范围宽泛    → Multi-Query 拆解后再并行检索


class QAState(TypedDict):
    """
    智能问答 Agent 的可变状态。
    所有节点通过读写此 State 进行数据传递。
    """

    # ── ① 消息历史（LangGraph 核心，add_messages reducer）────────
    messages: Annotated[list[BaseMessage], add_messages]

    # ── ② 不变字段（build_initial_state 注入）────────────────────
    enable_web_search: bool   # 可被 classify_query_node 改写

    # ── ③ Query 处理中间结果──────────────────────────────────────
    # intent：是否需要 RAG（GENERAL=跳过 / SPECIALIZED=进入检索链）
    # rag_strategy：仅 intent=SPECIALIZED 时有值，决定检索前预处理方式
    original_query:    str
    intent:            Intent
    rag_strategy:      Optional[RAGStrategy]
    rewritten_queries: list[str]
    hyde_document:     Optional[str]

    # ── ④ 检索与精排结果─────────────────────────────────────────
    ranked_chunks:      list[dict]
    confidence:         float
    is_high_confidence: bool
    web_search_results: list[dict]

    # ── ⑤ 生成结果 & 控制标记──────────────────────────────────────
    answer:            str
    sources:           list[str]
    answer_mode:       str           # "rag" / "llm_direct"
    summary:           Optional[str] # 会话摘要，checkpointer 自动持久化
    should_summarize:  bool
    fallback_used:     bool
    structured_output: Optional[dict]


class QAInput(TypedDict):
    """Graph 的输入边界。

    invoke() 只接受这三个字段。
    为保持 API 层设计独立，ChatRequest.to_graph_input() 负责翻译。
    """
    messages:         list[tuple[str, str]]  # [("user", "你好")]，add_messages 自动转 HumanMessage
    enable_web_search: bool
    original_query:   str


@dataclass
class QAContext:
    """Graph 单次运行的不可变依赖（不参与状态演化）。

    通过 LangGraph context API 注入，
    节点通过 runtime: Runtime[QAContext] 参数类型安全访问。
    """
    student_id: str
    tenant_id: str
    course_id: Optional[str] = None
