# backend/agents/qa/graph.py

import asyncio

from langgraph.graph import StateGraph, START, END

from backend.agents.qa.state import QAState, QAInput, QAContext, Intent
from backend.agents.qa.nodes import (
    classify_query_node,
    hyde_generate_node,
    multi_query_rewrite_node,
    retrieve_node,
    generate_rag_node,
    web_search_node,
    generate_direct_node,
    generate_general_node,
    enqueue_pending_node,
    save_memory_node,
)
from backend.core.memory import checkpointer
from backend.core.logger import get_logger

logger = get_logger(__name__)


def _route_after_classify(state: QAState) -> str:
    """
    classify_query 之后的路由：根据 intent + rag_strategy 和 enable_web_search 分流。

    intent=GENERAL + enable_web_search=True  → "GENERAL_WEB"：先联网再回答
    intent=GENERAL + enable_web_search=False → "GENERAL"：直接 LLM
    intent=SPECIALIZED                        → 按 rag_strategy 路由：
                                                  PRECISE → "PRECISE"
                                                  VAGUE   → "VAGUE"
                                                  BROAD   → "BROAD"
    """
    if state["intent"] == Intent.GENERAL:
        if state["enable_web_search"]:
            return "GENERAL_WEB"
        return "GENERAL"
    # SPECIALIZED：rag_strategy 决定检索预处理路径
    return state["rag_strategy"]


def _route_by_confidence(state: QAState) -> str:
    """
    retrieve 之后的路由：根据置信度和联网开关分流。

    is_high_confidence=True              → "high"：RAG 高质量回答
    is_high_confidence=False
      + enable_web_search=True           → "low_web"：先联网补充再直答
      + enable_web_search=False          → "low_direct"：直接 LLM 兜底
    """
    if state["is_high_confidence"]:
        return "high"
    if state["enable_web_search"]:
        return "low_web"
    return "low_direct"


def _route_after_web_search(state: QAState) -> str:
    """
    web_search 节点被两条路径共用，走完搜索后需要区分去向：
      - 来自 GENERAL_WEB 路径（intent=GENERAL）   → generate_general
      - 来自低置信度路径（intent=SPECIALIZED）     → generate_direct
    """
    if state["intent"] == Intent.GENERAL:
        return "generate_general"
    return "generate_direct"


# ── 图编译缓存（双重检查锁，避免 async 函数上 @cache 的问题）────
_build_lock = asyncio.Lock()
_compiled_graph = None


async def build_qa_graph():
    """
    构建并编译 QA Agent 的 LangGraph 状态图。

    首次编译时获取 AsyncPostgresSaver 并绑定到图中。
    后续调用直接返回缓存的 CompiledGraph。

    Returns:
        编译后的 CompiledGraph，供 API 层和 Orchestrator 调用。
    """
    global _compiled_graph
    if _compiled_graph is not None:
        return _compiled_graph

    async with _build_lock:
        if _compiled_graph is not None:  # 双重检查，避免并发出队
            return _compiled_graph

        assert checkpointer is not None, "init_checkpointer() must be called before build_qa_graph()"
        builder = StateGraph(QAState, context_schema=QAContext, input_schema=QAInput)

        # ── 注册节点 ──────────────────────────────────────────
        builder.add_node("classify_query",       classify_query_node)
        builder.add_node("hyde_generate",        hyde_generate_node)
        builder.add_node("multi_query_rewrite",  multi_query_rewrite_node)
        builder.add_node("retrieve",             retrieve_node)
        builder.add_node("generate_rag",         generate_rag_node)
        builder.add_node("web_search",           web_search_node)
        builder.add_node("generate_direct",      generate_direct_node)
        builder.add_node("generate_general",     generate_general_node)
        builder.add_node("enqueue_pending",      enqueue_pending_node)
        builder.add_node("save_memory",          save_memory_node)

        # ── 入口固定边 ────────────────────────────────────────
        builder.add_edge(START, "classify_query")

        # ── 条件边①：Query 类型路由 ──────────────────────────
        builder.add_conditional_edges(
            "classify_query",
            _route_after_classify,
            {
                "GENERAL":     "generate_general",
                "GENERAL_WEB": "web_search",
                "PRECISE":     "retrieve",
                "VAGUE":       "hyde_generate",
                "BROAD":       "multi_query_rewrite",
            },
        )

        # VAGUE / BROAD 预处理完成后汇入 retrieve
        builder.add_edge("hyde_generate",       "retrieve")
        builder.add_edge("multi_query_rewrite", "retrieve")

        # ── 条件边②：置信度路由 ──────────────────────────────
        builder.add_conditional_edges(
            "retrieve",
            _route_by_confidence,
            {
                "high":       "generate_rag",
                "low_web":    "web_search",
                "low_direct": "generate_direct",
            },
        )

        # ── 条件边③：web_search 出口路由 ─────────────────────
        builder.add_conditional_edges(
            "web_search",
            _route_after_web_search,
            {
                "generate_general": "generate_general",
                "generate_direct":  "generate_direct",
            },
        )

        # ── 固定边：各生成节点 → 收尾节点 ────────────────────
        builder.add_edge("generate_rag",     "save_memory")
        builder.add_edge("generate_general", "save_memory")
        builder.add_edge("generate_direct",  "enqueue_pending")
        builder.add_edge("enqueue_pending",  "save_memory")
        builder.add_edge("save_memory",      END)

        # ── 编译（绑定 AsyncPostgresSaver 实现多轮记忆持久化）──
        _compiled_graph = builder.compile(checkpointer=checkpointer)

        logger.info("graph.compiled", agent="qa")
        return _compiled_graph
