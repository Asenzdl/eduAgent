"""通用工具函数。"""
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage


def slice_last_n_rounds(
    messages: list[BaseMessage],
    window_size: int = 10,
) -> list[BaseMessage]:
    """
    手动按 HumanMessage 数切片，保留最近 window_size 轮。

    不依赖 trim_messages，纯粹的消息列表操作。
    逻辑：找到倒数第 window_size 个 HumanMessage，保留它及之后的所有消息。
          SystemMessage（如果在 index 0）始终保留。

    Args:
        messages:   当前消息列表
        window_size: 保留的最近轮数，默认 10 轮

    Returns:
        切片后的消息列表
    """
    # 分离 SystemMessage
    system = None
    rest = list(messages)
    if rest and isinstance(rest[0], SystemMessage):
        system = rest.pop(0)

    # 找到倒数第 window_size 个 HumanMessage
    human_indices = [i for i, m in enumerate(rest) if isinstance(m, HumanMessage)]
    if len(human_indices) > window_size:
        start = human_indices[-window_size]
    else:
        start = 0

    result = rest[start:]
    if system:
        result.insert(0, system)
    return result
