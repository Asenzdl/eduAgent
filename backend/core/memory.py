from pathlib import Path

from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from backend.core.logger import get_logger
from backend.config import get_settings

logger = get_logger(__name__)


# ── LangGraph Checkpointer ──────────────────────────────────────
# 模块级全局变量，lifespan startup 时初始化，shutdown 时关闭。
# 调用方直接 import checkpointer 使用，无需 getter 函数。

checkpointer: AsyncPostgresSaver | None = None


async def init_checkpointer() -> None:
    """应用启动时初始化 AsyncPostgresSaver + psycopg 连接池。

    · 创建全局连接池（min=1, max=5）
    · 幂等建表（checkpoints / checkpoint_blobs / checkpoint_writes）
    · 只应由 FastAPI lifespan 调用一次
    """
    global checkpointer
    if checkpointer is not None:
        return

    settings = get_settings()
    pool = AsyncConnectionPool(
        conninfo=settings.checkpointer_database_url,
        min_size=1,
        max_size=5,
        open=True,
    )
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()
    logger.info("memory.checkpointer_initialized", min=1, max=5)


async def close_checkpointer() -> None:
    """应用关闭时释放 psycopg 连接池。"""
    global checkpointer
    if checkpointer is not None and isinstance(checkpointer.conn, AsyncConnectionPool):
        await checkpointer.conn.close()
        checkpointer = None
        logger.info("memory.checkpointer_closed")


# ── 大 ToolMessage 内容归档目录 ──────────────────────────────────
_ARCHIVE_DIR = Path("data/tool_archives")


def _archive_tool_content(msg: ToolMessage) -> str:
    """将大 ToolMessage content 落盘，返回归档文件路径。"""
    _ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    file_path = _ARCHIVE_DIR / f"{msg.tool_call_id}.json"
    file_path.write_text(msg.content, encoding="utf-8")
    logger.info("memory.tool_archived", tool_call_id=msg.tool_call_id, path=str(file_path))
    return str(file_path)


def soften_tool_messages(
    messages: list[BaseMessage],
    max_chars: int = 2000,
    preserve_recent: int = 2,
) -> list[BaseMessage]:
    """
    软化大 ToolMessage，防止单个工具返回吃掉整个 token 预算。

    策略：保留最近的 preserve_recent 条 ToolMessage 不动，
          更早的 ToolMessage 如果超过 max_chars，则：
            1. 完整内容落盘归档
            2. 原位替换为截断文本 + 归档路径引用

    Args:
        messages:        消息列表
        max_chars:        单条 ToolMessage content 上限
        preserve_recent: 保留最近 N 条不动

    Returns:
        处理后的消息列表（大 ToolMessage 被替换为截断版 + 归档引用）
    """
    result = list(messages)
    recent_count = 0

    for i in range(len(result) - 1, -1, -1):
        if not isinstance(result[i], ToolMessage):
            continue
        recent_count += 1
        if recent_count <= preserve_recent:
            continue
        if len(result[i].content) <= max_chars:
            continue

        # 落盘 + 替换
        original = result[i].content
        archive_path = _archive_tool_content(result[i])
        preview = original[:max_chars]
        result[i] = ToolMessage(
            content=f"{preview}\n\n[完整数据已归档: {archive_path}]",
            tool_call_id=result[i].tool_call_id,
        )
        logger.info(
            "memory.tool_softened",
            tool_call_id=result[i].tool_call_id,
            original_length=len(original),
            archive_path=archive_path,
        )

    return result


def should_trigger_summary(
    messages: list[BaseMessage],
    threshold: int = 10,
) -> bool:
    """
    判断对话轮数是否超过阈值，决定是否触发摘要压缩。

    轮数 = HumanMessage 数量（每次提问算 1 轮），不计 SystemMessage。
    按 HumanMessage 计是因为工具轮 4 条/纯轮 2 条，按条数不反映实际轮次。

    Args:
        messages:  当前消息列表
        threshold: 触发压缩的轮数阈值，默认 10 轮

    Returns:
        True → 需要压缩
    """
    human_count = sum(1 for m in messages if isinstance(m, HumanMessage))
    return human_count >= threshold


def trim_messages_to_window(
    messages: list[BaseMessage],
    window_size: int = 10,
    max_tokens: int | None = 2000,
) -> tuple[list[BaseMessage], list[BaseMessage]]:
    """
    先按轮次裁剪，再按 token 预算裁剪。

    第一刀：以 HumanMessage 为轮次标识，保留最近 window_size 轮。
    第二刀：[可选] 用 trim_messages + "approximate" 做 token 预算裁剪。

    Args:
        messages:   当前消息列表
        window_size: 保留的最近轮数，默认 10 轮
        max_tokens:  token 预算上限，默认 2000。None 表示不做 token 裁剪。

    Returns:
        (保留的消息列表, 被裁掉的消息列表（两刀合计，用于摘要）)
    """
    from langchain.messages import trim_messages

    def _round_counter(msgs: list[BaseMessage]) -> int:
        return sum(1 for m in msgs if isinstance(m, HumanMessage))

    # ── 第一刀：按轮次裁剪 ──
    result = trim_messages(
        messages,
        max_tokens=window_size,
        token_counter=_round_counter,
        strategy="last",
        start_on="human",
        include_system=True,
    )

    # ── 第二刀：[可选] token 预算裁剪 ──
    if max_tokens is not None:
        result = trim_messages(
            result,
            max_tokens=max_tokens,
            token_counter="approximate",
            strategy="last",
            start_on="human",
            include_system=True,
        )

    # 被裁掉的部分 = 原始消息 - 最终保留消息（两刀合计）
    result_ids = {id(m) for m in result}
    to_summarize = [m for m in messages if id(m) not in result_ids]

    return result, to_summarize


async def compress_to_summary(
    messages: list[BaseMessage],
    existing_summary: str | None = None,
) -> str:
    """
    将历史对话压缩为结构化学员画像摘要。

    增量压缩：传入 existing_summary 防止已记录内容被重复写入。
    调用前应由 trim_messages_to_window 预先裁剪，本函数不再重复裁。

    Args:
        messages:         待压缩的历史消息列表（建议先经 trim_messages_to_window 处理）
        existing_summary: 上次的摘要（可选）

    Returns:
        压缩后的摘要文本
    """
    from langchain_core.messages import HumanMessage
    from backend.core.llm_factory import get_llm

    SUMMARY_PROMPT = """请将以下学员对话压缩为结构化学员画像摘要。

【压缩规则】
必须保留：学员明确不理解的知识点 / 反复出现的薄弱点 / 项目背景新增信息
选择性保留：已掌握知识点（简短标注）/ 学习进度信息
可以丢弃：已在上次摘要记录的内容 / 闲聊 / 已解决且理解的问题

【上一次摘要】
{previous_summary}

【本次新增对话】
{new_conversations}

请直接输出摘要文本，不要加任何前缀。"""

    def _format_msg(m: BaseMessage) -> str:
        """将单条消息转为摘要用的文本行（含工具调用信息）。"""
        if isinstance(m, HumanMessage):
            return f"学员：{m.content}"
        elif isinstance(m, AIMessage):
            if m.tool_calls:
                tool_names = ", ".join(tc["name"] for tc in m.tool_calls)
                return f"AI：调用了工具 [{tool_names}]，结果如下："
            return f"AI：{m.content}"
        elif isinstance(m, ToolMessage):
            return f"工具返回：{m.content}"
        return f"[{type(m).__name__}]：{m.content}"

    conversation_text = "\n".join(_format_msg(m) for m in messages)

    prompt_text = SUMMARY_PROMPT.format(
        previous_summary=existing_summary or "（无上次摘要）",
        new_conversations=conversation_text,
    )

    llm = get_llm("summarize")
    response = await llm.ainvoke([HumanMessage(content=prompt_text)])
    summary = response.content.strip()

    logger.info(
        "memory.summary_generated",
        input_messages=len(messages),
        summary_length=len(summary),
    )
    return summary


if __name__ == "__main__":
    pass
