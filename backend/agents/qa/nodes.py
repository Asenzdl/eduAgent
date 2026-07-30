import asyncio
import re
from typing import Type
import uuid

from sqlalchemy import text
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.runtime import Runtime

from backend.agents.qa.state import QAState, QAContext, Intent, RAGStrategy
from backend.agents.qa.prompts import (
    HYDE_PROMPT,
    MULTI_QUERY_REWRITE_PROMPT,
    RAG_ANSWER_PROMPT,
    DIRECT_ANSWER_PROMPT,
    GENERAL_ANSWER_PROMPT,
    RAG_STRATEGY_PROMPT,
    SYSTEM_PROMPT,
)
from backend.core.llm_factory import get_llm
from backend.core.memory import (
    trim_messages_to_window,
    should_trigger_summary,
    compress_to_summary,
)
from backend.config import get_settings
from backend.core.logger import get_logger
from backend.core.query_classifier import get_query_classifier

logger = get_logger(__name__)

# ── 检索相关常量 ───────────────────────────────────────────────
MAX_BROAD_QUERIES        = 3   # BROAD 分支最多并行的子 Query 数
RECALL_TOP_K_PRECISE     = 8   # PRECISE：直接检索召回数
RECALL_TOP_K_VAGUE       = 10  # VAGUE：HyDE 语义扩充后多召回些
RECALL_TOP_K_BROAD_PER   = 4   # BROAD：每个子 Query 的召回数
RERANK_TOP_K             = 3   # 精排后保留的最终 chunk 数
_RAG_RULE_LOW_CONF_MIN_LEN = 18  # 规则判 VAGUE/BROAD 时，查询短于此值跳过 LLM（短查询 LLM 也判不准）



# ──────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────

def _format_history_for_prompt(messages: list[BaseMessage], tool_truncate_at: int | None = 300) -> str:
    """把最近几轮对话格式化为 Prompt 可用的文本（逐条委托 _format_msg）。"""
    if not messages:
        return "（无历史对话）"
    return "\n".join(_format_msg(m, tool_truncate_at=tool_truncate_at) for m in messages)


def _format_msg(m: BaseMessage, tool_truncate_at: int | None = None) -> str:
    """
    将单条消息转为文本行（含工具调用信息）。

    tool_truncate_at : 对 ToolMessage 内容截断到该长度。
                       传入 None（默认）表示不截断。其他消息类型不受此参数影响。
    """
    if isinstance(m, HumanMessage):
        return f"学员：{m.content}"
    elif isinstance(m, AIMessage):
        if m.tool_calls:
            tool_names = ", ".join(tc["name"] for tc in m.tool_calls)
            return f"AI：调用了工具 [{tool_names}]，结果如下："
        return f"AI：{m.content}"
    elif isinstance(m, ToolMessage):
        content = m.content
        if tool_truncate_at and len(content) > tool_truncate_at:
            return f"工具返回：{content[:tool_truncate_at]}…"
        return f"工具返回：{content}"
    return f"[{type(m).__name__}]：{m.content}"

def _current_datetime_str() -> str:
    """返回 ISO 格式时间字符串，供注入 Prompt 使用"""
    from datetime import datetime
    return datetime.now().isoformat(timespec="minutes")


def _build_system_content(summary: str | None = None) -> str:
    """
    构建注入了当前时间的 SystemMessage 内容，所有生成节点共用。
    summary 非空时追加历史学习摘要，帮助 LLM 保持多轮上下文连贯。
    """
    content = SYSTEM_PROMPT + f"\n\n【当前时间】{_current_datetime_str()}"
    if summary:
        content += f"\n\n【学员历史学习摘要】\n{summary}"
    return content


# ──────────────────────────────────────────────────────────────
# 规则集
# ──────────────────────────────────────────────────────────────

# ── Layer 1：精确匹配 ──────────────────────────────────────────
# 查询必须与集合中某一项完全一致才命中，无前缀/后缀容差。
# 用作高频短查询的快速短路：确定性的闲聊/打招呼/感谢。
_GENERAL_EXACT = {
    "你好", "hi", "hello", "嗨", "hey",
    "谢谢", "谢谢你", "感谢", "thanks", "thank you",
    "你是谁", "你叫什么", "你叫什么名字", "你是什么",
    "你能做什么", "你有什么功能", "你能帮我做什么",
    "再见", "拜拜", "bye",
}

# ── Layer 2：关键词子串匹配 ───────────────────────────────────
# 查询只需包含其中任一个关键词即命中（"帮我介绍一下你自己"→命中"介绍一下你自己"）。
# 作为 Layer 1 的 fallback，覆盖精确集合未穷尽的自然语言变体。
_GENERAL_KEYWORDS = (
    "你是谁", "你叫什么", "你能做什么", "你有什么功能",
    "介绍一下你自己", "自我介绍",
    "今天天气", "天气怎么样",
    "讲个笑话", "说个故事",
    "今天是", "今天几号", "今天是几号", "今天是星期",
    "现在是", "现在几点", "现在时间", "当前时间", "当前日期",
    "几月几号", "星期几", "是几月", "几号了", "日期是", "今天日期",
)

# 命中即确认为专业问题，跳过 MiniLM，直接进 Layer 2
_SPECIALIZED_KEYWORDS = (
    "课程", "实战", "项目", "案例", "老师", "章节",
    "作业", "课堂", "培训", "我们学的", "课程项目",
    "第几章", "第几节", "训练营",
)

_VAGUE_QUERY_HINTS = (
    "没懂", "不懂", "不太懂", "讲讲", "解释一下",
    "啥意思", "什么意思", "看不懂",
)
_BROAD_QUERY_HINTS = (
    "全面", "系统", "总结", "梳理", "路线",
    "对比", "区别", "全景", "有哪些",
)


def _rule_classify_general(query: str) -> bool:
    """规则层：是否为闲聊/时间/打招呼类（→ GENERAL）"""
    q = query.strip().lower()

    # Layer 1：精确命中 — O(1) 查表，确定性闲聊快速短路
    if q in _GENERAL_EXACT:
        return True

    # Layer 2：关键词 fallback — 子串包含匹配，覆盖精确集合未列出的变体
    return any(kw in q for kw in _GENERAL_KEYWORDS)


# ── 预编译正则版（备选方案）─────────────────────────────────────
# 将 _GENERAL_KEYWORDS 中所有模式用 `|` 连接为一个正则，由 re 模块
# 在编译时构建 DFA（确定性有限自动机），扫描一次 query 即可同时检测
# 所有关键词的任一处出现，无需 Python 层逐个循环。
#
# 为什么更快：关键词列表越长，优势越明显。
#   原始版      any(kw in q for kw in ...)     每个关键词独立扫描，最坏 O(K·N)
#   预编译正则  re.compile(...).search(q)      整体一次扫描 O(N)，K 个关键词合并为一个自动机
#
# 适用时机：当关键词数量增长到 50+，或 _rule_classify_general 调用频率
# 成为热点（profile 显示 >5% 时间）时，可替换原始版。
_GENERAL_PATTERN = re.compile(
    "|".join(map(re.escape, _GENERAL_KEYWORDS)),
    re.IGNORECASE,  # 传入的 query 已 .lower()，加 flags 保证一致性
)


def _rule_classify_general_regex(query: str) -> bool:
    """规则层（预编译正则版）：是否为闲聊/时间/打招呼类（→ GENERAL）

    机制
    ----
    1. 与原始版共用 _GENERAL_EXACT 精确集合（第一层短路）。
    2. 关键词匹配改用预编译的 _GENERAL_PATTERN.search()，
       一次扫描检测所有关键词，替代 Python 层 any(kw in q) 循环。

    性能
    ----
    关键词约 25 条时，吞吐约 0.3 µs/次（原始版约 0.5 µs/次）。
    关键词越多边际优势越大，但当前规模下两者均非瓶颈。
    """
    q = query.strip().lower()

    # Layer 1：精确命中 — 与原始版一致，确定性闲聊快速短路
    if q in _GENERAL_EXACT:
        return True

    # Layer 2：预编译正则 — 单次 DFA 扫描覆盖所有关键词
    return _GENERAL_PATTERN.search(q) is not None


def _rule_classify_specialized(query: str) -> bool:
    """规则层：是否含课程/项目信号词（→ 专业，跳过 MiniLM）"""
    q = query.lower()
    return any(kw in q for kw in _SPECIALIZED_KEYWORDS)


def _rule_classify(query: str) -> str:
    """规则层分类：返回 GENERAL / SPECIALIZED / UNKNOWN"""
    if _rule_classify_general(query):
        return "GENERAL"
    if _rule_classify_specialized(query):
        return "SPECIALIZED"
    return "UNKNOWN"


def _rule_rag_strategy(query: str) -> str:
    """
    规则快判 RAG 策略（< 1ms）。

    分支顺序（第一条命中即返回，后续不检查）：
      ① VAGUE   → 查询含「没懂/解释一下/啥意思」等模糊信号词。
                   排在 BROAD 之前，因为「全面总结一下归一化」既含 VAGUE
                   也含 BROAD 信号，应该优先处理"表达不清"的语义。
      ② BROAD   → 查询含「全面/系统/对比/有哪些」等系统梳理信号。
                   排在 VAGUE 之后，仅在无模糊歧义时才走宽泛改写。
      ③ PRECISE → 默认分支。无任何信号词命中时，认为查询意图明确，
                   直接精确检索，不引入重写/扩写的开销。

    注意：此函数只查信号词是否存在，不做否定推断——
    例如「"没懂"的反义词是什么」含 VAGUE 词但意图明确，
    会在 _determine_rag_strategy_llm 中被 LLM 校正为 PRECISE。
    """
    q = query.strip().lower()
    if any(kw in q for kw in _VAGUE_QUERY_HINTS):
        return "VAGUE"
    if any(kw in q for kw in _BROAD_QUERY_HINTS):
        return "BROAD"
    return "PRECISE"


async def _determine_rag_strategy_llm(query: str) -> str | None:
    """LLM 精判 RAG 策略。失败返回 None，由调用方决定兜底。"""
    try:
        llm = get_llm("qa", temperature=0)
        resp = await llm.ainvoke([
            HumanMessage(content=RAG_STRATEGY_PROMPT.format(query=query))
        ])
        label = resp.content.strip().upper()
        if label in ("PRECISE", "VAGUE", "BROAD"):
            return label
    except Exception:
        logger.warning("llm_classification_failed", query=query, exc_info=True)
    return None


async def _resolve_rag_strategy(query: str) -> str:
    """
    两阶段策略判定：
      ① 规则判 PRECISE → 直接采信（不调 LLM）
      ② 规则判 VAGUE/BROAD + 查询够长（≥18字）→ LLM 校正，LLM 失败退回规则
      ③ 规则判 VAGUE/BROAD + 查询太短 → 跳过 LLM（短查询 LLM 也判不准），采信规则
    """
    strategy = _rule_rag_strategy(query)

    if strategy == "PRECISE":
        return strategy

    if len(query.strip()) >= _RAG_RULE_LOW_CONF_MIN_LEN:
        return await _determine_rag_strategy_llm(query) or strategy

    return strategy


# ──────────────────────────────────────────────────────────────
# 节点：classify_query — Query 类型判断（含历史摘要加载）
# ──────────────────────────────────────────────────────────────

async def classify_query_node(state: QAState, runtime: Runtime[QAContext]) -> dict:
    """
    Query 分类节点，决定走哪条处理路径。

    两阶段流水线：
      Phase A — 规则快判（0 µs，确定性）
      Phase B — MiniLM 语义分类（~5ms CPU 推理，仅规则判 UNKNOWN 时触发）

    历史摘要通过 state.summary 从 checkpointer 获取，不再单独查 DB。
    original_query 由 InputSchema 注入，不经由 messages[-1] 提取。

    返回两个正交维度的字段：
      intent       — GENERAL（跳过 RAG）/ SPECIALIZED（进入检索链）
      rag_strategy — 仅 intent=SPECIALIZED 时有值：
                       PRECISE → 直接向量检索
                       VAGUE   → 先 HyDE 再检索
                       BROAD   → 先 Multi-Query 改写再并行检索
                     intent=GENERAL 时不写入，保持 None。
    """
    original_query = state["original_query"]

    # ── Phase A：规则快判（GENERAL / SPECIALIZED / UNKNOWN）──────
    stage = _rule_classify(original_query)

    # ── Phase B：规则判 UNKNOWN → MiniLM 语义分类 ──────────────
    if stage == "UNKNOWN":
        loop = asyncio.get_running_loop()
        label, confidence = await loop.run_in_executor(
            None, get_query_classifier().classify, original_query
        )
        stage = "GENERAL" if label == "general" else "SPECIALIZED"
        logger.info("query_classified", query=original_query, intent=stage,
                    classifier='MiniLM', confidence=confidence)

    # ── GENERAL：跳过 RAG，不判定 rag_strategy ─────────────────
    if stage == "GENERAL":
        return {"intent": Intent.GENERAL}

    # ── SPECIALIZED：判定 RAG 检索策略 ────────────────────────
    strategy = await _resolve_rag_strategy(original_query)
    logger.info("rag_strategy_decided", query=original_query, rag_strategy=strategy)
    return {"intent": Intent.SPECIALIZED, "rag_strategy": RAGStrategy(strategy)}


# ──────────────────────────────────────────────────────────────
# 节点：hyde_generate — HyDE 假设文档生成（VAGUE 分支）
# ──────────────────────────────────────────────────────────────

async def hyde_generate_node(state: QAState) -> dict:
    """
    针对模糊 Query，让 LLM 先生成一段假设性回答文档，
    用该文档的向量代替原始 Query 向量去检索。

    temperature=0.3：生成结果要有一定多样性（覆盖更多语义），
    但不能太随机（避免偏离主题）。
    """
    query    = state["original_query"]
    messages = state.get("messages", [])

    history_text = _format_history_for_prompt(messages[-6:])  # 最近 3 轮

    prompt = HYDE_PROMPT.format(history=history_text, query=query)

    llm = get_llm("qa", temperature=0.3)
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    hyde_doc = response.content.strip()

    logger.info(
        "hyde_generate.done",
        query=query[:50],
        hyde_doc_length=len(hyde_doc),
    )

    return {"hyde_document": hyde_doc}


# ──────────────────────────────────────────────────────────────
# 节点：multi_query_rewrite — Multi-Query 改写（BROAD 分支）
# ──────────────────────────────────────────────────────────────

def _get_last_content(messages: list[BaseMessage], type: Type[BaseMessage]) -> str:
    for i in range(len(messages)-1, -1, -1):
        if isinstance(messages[i], type):
            return messages[i].content
    return ""


async def multi_query_rewrite_node(state: QAState) -> dict:
    """
    针对宽泛/极短 Query，改写为 3-5 个具体子 Query，
    下一节 retrieve_node 会对这些子 Query 并行检索后合并去重。
    """
    query    = state["original_query"]
    messages = state.get("messages", [])

    # 取上一轮 AI 回答（推断"没懂"指的是什么）
    last_ai_text = _get_last_content(messages, AIMessage) or "（无上一轮回答）"

    prompt = MULTI_QUERY_REWRITE_PROMPT.format(
        last_answer=last_ai_text[:500],
        query=query,
    )

    llm = get_llm("qa", temperature=0.3)
    response = await llm.ainvoke([HumanMessage(content=prompt)])

    # 解析：每行一个子 Query，去掉序号前缀（"1. " "①" "- " 等）
    raw = response.content.strip()
    rewritten = [
        line.lstrip("0123456789.-）)、 ").strip()
        for line in raw.split("\n")
        if line.strip() and len(line.strip()) > 3
    ][:MAX_BROAD_QUERIES]

    if not rewritten:
        rewritten = [query]   # 改写失败时回退到原始 Query

    logger.info(
        "multi_query_rewrite.done",
        original=query,
        count=len(rewritten),
        queries=rewritten,
    )

    return {"rewritten_queries": rewritten}


# ──────────────────────────────────────────────────────────────
# 节点：retrieve — 混合召回 + 精排
# ──────────────────────────────────────────────────────────────

async def retrieve_node(state: QAState) -> dict:
    """
    调用 retrieve() Pipeline 完成检索与精排，直接输出 ranked_chunks。

    三条路径：
      PRECISE → 用 original_query 直接检索
      VAGUE   → 用 hyde_document 替代 original_query（语义扩充）
      BROAD   → 对所有 rewritten_queries 并行检索，结果合并去重

    retrieve() 是同步函数（BGE-M3 CPU 推理 + Milvus 阻塞 IO），
    必须用 run_in_executor 包装，避免阻塞 asyncio 事件循环。
    """
    from backend.core.retriever import retrieve
    from backend.core.reranker import RankedDocument

    rag_strategy   = state.get("rag_strategy", RAGStrategy.PRECISE)
    tenant_id      = state["tenant_id"]
    course_id      = state.get("course_id")
    original_query = state["original_query"]

    loop = asyncio.get_running_loop()

    # ── BROAD：并行多 Query 检索，合并去重 ───────────────────────
    if rag_strategy == RAGStrategy.BROAD and state.get("rewritten_queries"):
        broad_queries = state["rewritten_queries"][:MAX_BROAD_QUERIES]

        async def retrieve_one(sub_query: str) -> tuple[list, float]:
            return await loop.run_in_executor(
                None,
                lambda: retrieve(
                    sub_query,
                    tenant_id,
                    course_id,
                    recall_top_k=RECALL_TOP_K_BROAD_PER,
                    rerank_top_k=RECALL_TOP_K_BROAD_PER,
                ),
            )

        results = await asyncio.gather(*[retrieve_one(q) for q in broad_queries])

        # 合并去重：content 前 100 字符为 key，同一内容保留最高分
        seen: dict[str, RankedDocument] = {}
        for ranked_docs, _ in results:
            for doc in ranked_docs:
                key = doc.content[:100]
                if key not in seen or doc.score > seen[key].score:
                    seen[key] = doc

        merged = sorted(seen.values(), key=lambda x: x.score, reverse=True)[:RERANK_TOP_K]

    # ── PRECISE / VAGUE：单路检索 ─────────────────────────────────
    else:
        if rag_strategy == RAGStrategy.VAGUE and state.get("hyde_document"):
            query_text   = state["hyde_document"]
            recall_top_k = RECALL_TOP_K_VAGUE
        else:
            query_text   = original_query
            recall_top_k = RECALL_TOP_K_PRECISE

        merged, _ = await loop.run_in_executor(
            None,
            lambda: retrieve(
                query_text,
                tenant_id,
                course_id,
                recall_top_k=recall_top_k,
                rerank_top_k=RERANK_TOP_K,
            ),
        )

    # ── 转换 RankedDocument → dict，写入 State ─────────────────────
    ranked_chunks = [
        {
            "content":  doc.content,
            "score":    doc.score,
            "metadata": doc.metadata,
        }
        for doc in merged
    ]

    confidence         = ranked_chunks[0]["score"] if ranked_chunks else 0.0
    is_high_confidence = confidence >= 0.75

    logger.info(
        "retrieve.done",
        rag_strategy=rag_strategy,
        ranked=len(ranked_chunks),
        confidence=round(confidence, 4),
        is_high_confidence=is_high_confidence,
    )

    return {
        "ranked_chunks":      ranked_chunks,
        "confidence":         confidence,
        "is_high_confidence": is_high_confidence,
    }


# ──────────────────────────────────────────────────────────────
# 节点：generate_rag — 高置信度 RAG 生成
# ──────────────────────────────────────────────────────────────

async def generate_rag_node(state: QAState) -> dict:
    """
    高置信度 RAG 生成节点。

    将精排后的 Top-3 文档拼成 context，让 LLM 严格基于知识库内容回答。
    回答末尾附加 📚 参考来源，支持历史摘要注入保持多轮连贯性。
    """
    ranked_chunks = state.get("ranked_chunks", [])
    query         = state["original_query"]
    messages      = state.get("messages", [])
    summary       = state.get("summary")

    # 构建知识库上下文与来源列表
    context_parts = []
    sources = []
    for i, chunk in enumerate(ranked_chunks, 1):
        context_parts.append(f"【参考{i}】\n{chunk['content']}")
        source_name = chunk.get("metadata", {}).get("source_name", "课程文档")
        if source_name not in sources:
            sources.append(source_name)

    context_text = "\n\n".join(context_parts)

    # 消息列表：SystemMessage（含当前时间 + 历史摘要）
    llm_messages = [SystemMessage(content=_build_system_content(summary))]

    # 注入历史对话窗口（排除最后一条 HumanMessage）
    # 最后一条 HumanMessage 已拼入 RAG_ANSWER_PROMPT 的 {query}，
    # 再传一次会让问题在上下文里出现两次，影响生成质量。
    windowed = trim_messages_to_window(messages[:-1], window_size=10)
    for msg in windowed:
        if not isinstance(msg, SystemMessage):
            llm_messages.append(msg)

    rag_prompt = RAG_ANSWER_PROMPT.format(context=context_text, query=query)
    llm_messages.append(HumanMessage(content=rag_prompt))

    llm = get_llm("qa", streaming=True)
    response = await llm.ainvoke(llm_messages)
    answer_text = response.content.strip()

    sources_text = "\n".join([f"  • {s}" for s in sources])
    final_answer = f"{answer_text}\n\n📚 **参考来源**\n{sources_text}"

    logger.info(
        "generate_rag.done",
        answer_length=len(final_answer),
        sources=sources,
        confidence=round(state.get("confidence", 0), 4),
    )

    return {
        "answer":      final_answer,
        "sources":     sources,
        "answer_mode": "rag",
        "messages":    [AIMessage(content=final_answer)],
        "should_summarize": should_trigger_summary(messages),
        "structured_output": {
            "answer":      final_answer,
            "sources":     sources,
            "confidence":  state.get("confidence", 0),
            "answer_mode": "rag",
        },
    }

# ──────────────────────────────────────────────────────────────
# 节点：web_search — Web 搜索补充（低置信度分支）
# ──────────────────────────────────────────────────────────────

async def web_search_node(state: QAState) -> dict:
    """
    低置信度分支的 Web 搜索节点。

    知识库置信度不足时，调用 Web Search MCP Server 补充互联网信息。
    MCP Server 不可用时静默降级，返回空列表，不阻断后续生成节点。
    """
    from backend.mcp.search_server import call_mcp_tool

    settings = get_settings()

    if not settings.web_search_mcp_url:
        return {"web_search_results": []}

    try:
        results = await call_mcp_tool(
            server_url=settings.web_search_mcp_url,
            tool_name="web_search",
            arguments={
                "query":       state["original_query"],
                "max_results": 3,
            },
            timeout=10.0,
        )
        count = len(results or [])
        logger.info("web_search.done", count=count)
        return {"web_search_results": results or []}
    except Exception as e:
        logger.warning("web_search.failed", error=str(e))
        return {"web_search_results": []}

# ──────────────────────────────────────────────────────────────
# 节点：generate_direct — 低置信度 LLM 直答（含 Web 搜索补充）
# ──────────────────────────────────────────────────────────────

async def generate_direct_node(state: QAState) -> dict:
    """
    低置信度 LLM 直答节点。

    知识库无足够相关内容时，直接用 LLM 参数知识回答。
    有 Web 搜索结果时注入为上下文（web_augmented 模式）；
    无搜索结果时在回答末尾追加 ⚠️ 提示（llm_direct 模式）。
    """
    query    = state["original_query"]
    messages = state.get("messages", [])
    summary  = state.get("summary")

    llm_messages = [SystemMessage(content=_build_system_content(summary))]

    windowed = trim_messages_to_window(messages[:-1], window_size=10)
    for msg in windowed:
        if not isinstance(msg, SystemMessage):
            llm_messages.append(msg)

    # ── Web 搜索结果注入 ──────────────────────────────────────
    web_results = state.get("web_search_results") or []
    web_context = ""
    web_sources: list[str] = []
    if web_results:
        snippets = "\n".join(
            f"  [{i + 1}] {r.get('title', '')}（{r.get('url', '')}）\n"
            f"      {r.get('snippet', '')[:300]}"
            for i, r in enumerate(web_results)
        )
        web_context = f"\n\n【Web 搜索补充参考】\n{snippets}"
        web_sources = [r.get("url", "") for r in web_results if r.get("url")]

    direct_prompt = DIRECT_ANSWER_PROMPT.format(query=query) + web_context
    llm_messages.append(HumanMessage(content=direct_prompt))

    llm = get_llm("qa", streaming=True)
    response = await llm.ainvoke(llm_messages)
    answer_text = response.content.strip()

    if web_sources:
        # URL 通过 sources 字段传给前端，由 UI 折叠面板展示，不拼进正文
        final_answer = answer_text
        answer_mode  = "web_augmented"
    else:
        final_answer = (
            f"{answer_text}\n\n"
            f"⚠️ **说明**：以上为 AI 基于通用知识的回答，课程知识库中暂无相关内容。"
            f"建议以教师讲解为准，或联系教师补充相关资料。"
        )
        answer_mode = "llm_direct"

    logger.info(
        "generate_direct.done",
        answer_length=len(final_answer),
        confidence=round(state.get("confidence", 0), 4),
        web_sources=len(web_sources),
    )

    return {
        "answer":      final_answer,
        "sources":     web_sources,
        "answer_mode": answer_mode,
        "messages":    [AIMessage(content=final_answer)],
        "should_summarize": should_trigger_summary(messages),
        "structured_output": {
            "answer":      final_answer,
            "sources":     web_sources,
            "confidence":  state.get("confidence", 0),
            "answer_mode": answer_mode,
        },
    }


# ──────────────────────────────────────────────────────────────
# 节点：generate_general — 通用问题直答（跳过 RAG）
# ──────────────────────────────────────────────────────────────

async def generate_general_node(state: QAState) -> dict:
    """
    通用问题直答节点（intent=GENERAL）。

    适用于：打招呼、问时间、闲聊等与课程无关的问题。
    联网模式下若 web_search_results 非空，注入搜索结果提供时效性信息。
    """
    query       = state["original_query"]
    messages    = state.get("messages", [])
    web_results = state.get("web_search_results") or []

    web_context = ""
    web_sources: list[str] = []
    if web_results:
        snippets = "\n".join(
            f"  [{i + 1}] {r.get('title', '')}（{r.get('url', '')}）\n"
            f"      {r.get('snippet', '')[:300]}"
            for i, r in enumerate(web_results)
        )
        web_context = f"【Web 搜索结果】\n{snippets}\n\n"
        web_sources = [r.get("url", "") for r in web_results if r.get("url")]

    history_text = _format_history_for_prompt(messages[-6:])
    prompt = GENERAL_ANSWER_PROMPT.format(
        query=query,
        history=history_text,
        current_time=_current_datetime_str(),
        web_context=web_context,
    )

    llm = get_llm("qa", streaming=True)
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    answer_text = response.content.strip()

    answer_mode = "web_augmented" if web_sources else "general"

    logger.info(
        "generate_general.done",
        answer_length=len(answer_text),
        web_sources=len(web_sources),
    )

    return {
        "answer":      answer_text,
        "sources":     web_sources,
        "answer_mode": answer_mode,
        "messages":    [AIMessage(content=answer_text)],
        "should_summarize": should_trigger_summary(messages),
        "structured_output": {
            "answer":      answer_text,
            "sources":     web_sources,
            "confidence":  1.0,
            "answer_mode": answer_mode,
        },
    }


# ──────────────────────────────────────────────────────────────
# 节点：enqueue_pending — 低置信度问题入队（纯副作用）
# ──────────────────────────────────────────────────────────────

async def enqueue_pending_node(state: QAState) -> dict:
    """
    将低置信度问题写入 knowledge_pending_queue，供教师审查补充知识库。

    ON CONFLICT DO NOTHING：幂等写入，同一问题重复触发不会产生重复记录。
    失败静默，不影响已生成的回答。返回 {} 不修改 State。
    """
    from backend.dependencies import AsyncSessionLocal

    try:
        async with AsyncSessionLocal() as session:
            async with session.begin():
                await session.execute(
                    text("""
                        INSERT INTO knowledge_pending_queue
                            (id, tenant_id, question, student_id, confidence, status)
                        VALUES (:id, :tenant_id, :question, :student_id, :confidence, 'pending')
                        ON CONFLICT DO NOTHING
                    """),
                    {
                        "id":         str(uuid.uuid4()),
                        "tenant_id":  state["tenant_id"],
                        "question":   state["original_query"],
                        "student_id": state["student_id"],
                        "confidence": state.get("confidence", 0.0),
                    },
                )
        logger.info(
            "enqueue_pending.done",
            question=state["original_query"][:50],
            confidence=state.get("confidence", 0),
        )
    except Exception as e:
        logger.warning("enqueue_pending.failed", error=str(e))

    return {}

# ──────────────────────────────────────────────────────────────
# 节点：save_memory — 记忆保存（纯副作用）
# ──────────────────────────────────────────────────────────────

async def save_memory_node(state: QAState, runtime: Runtime[QAContext]) -> dict:
    """
    记忆保存节点：条件触发摘要压缩。

    should_summarize=True（对话超过 10 轮）时压缩历史。
    checkpointer 自动持久化 state，无需手动写 DB。
    任何步骤失败静默，不中断流程。
    """
    messages = state.get("messages", [])
    summary  = state.get("summary")

    if state.get("should_summarize", False):
        try:
            msgs_to_compress = trim_messages_to_window(messages, window_size=10)
            summary = await compress_to_summary(
                messages=msgs_to_compress,
                existing_summary=summary,
            )
            logger.info("save_memory.summary_compressed")
        except Exception as e:
            logger.warning("save_memory.compress_failed", error=str(e))

    return {"summary": summary} if summary else {}


if __name__ == "__main__":
    print(_build_system_content("你好"))