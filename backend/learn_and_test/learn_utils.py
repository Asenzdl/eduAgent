"""learn_v2 的工具函数：构造对话 + 展示消息 + token 计数器。"""
import math
import unicodedata
from pathlib import Path
from tokenizers import Tokenizer
from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage,
)

# ── DeepSeek BPE tokenizer（模块级，只加载一次） ──
_bpe_tk = Tokenizer.from_file(
    str(Path(__file__).parent.parent / "deepseek_tokenizer" / "tokenizer.json")
)


def bpe_token_counter(msg: BaseMessage) -> int:
    """逐条模式：BPE 精确计数 + 角色标记 overhead。

    参数标注了 : BaseMessage → trim_messages 自动识别为逐条模式，
    内部会调 sum(bpe_token_counter(m) for m in messages)。

    计 token 逻辑：
        content 文本 + tool_calls JSON → BPE encode → len
        + 3（<｜User｜> 或 <｜Assistant｜> 的近似 overhead）
    """
    content = msg.content if isinstance(msg.content, str) else ""
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        content += "\n" + str(msg.tool_calls)
    return len(_bpe_tk.encode(content).ids) + 3


def cjk_token_counter(msg: BaseMessage) -> int:
    """逐条模式：CJK 感知的启发式计数。

    按字符类型分别用不同的倍率：
        CJK 字符（中文）: 1.5 字符 ≈ 1 token
        其他字符（英文/数字/符号）: 4 字符 ≈ 1 token
        角色标记 overhead: +3

    使用 unicodedata.category 判断 CJK（category='Lo' 覆盖所有扩展区），
    比硬编码区间更可靠。这是业界常见的折中方案——不用加载 BPE 词表，
    精度介于 approximate 和 BPE 之间。
    参考 headroomlabs EstimatingTokenCounter PR#1093 的做法。
    """
    text = msg.content if isinstance(msg.content, str) else ""
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        text += "\n" + str(msg.tool_calls)

    cjk = sum(1 for c in text if unicodedata.category(c) == 'Lo')
    other = len(text) - cjk
    return math.ceil(cjk / 1.5) + math.ceil(other / 4.0) + 3


def build_conversation() -> list[BaseMessage]:
    """含工具调用的多轮对话，内容较长让裁剪效果明显。"""
    return [
        SystemMessage(
            "你是一名专业的天气助手。你需要根据用户提问，使用天气查询工具获取数据，"
            "然后给出详细、准确的回答。所有回答请用中文。"
        ),
        # ── 第 1 轮（工具调用）──
        HumanMessage("北京今天天气怎么样？我计划去故宫玩，想知道具体温度、降雨概率和风力。"),
        AIMessage(content="", tool_calls=[
            {"name": "get_weather", "args": {"location": "北京"}, "id": "call_1"},
        ]),
        ToolMessage(
            '{"temperature": 22, "condition": "晴", "humidity": 30, '
            '"wind": "3-4级", "rain_probability": 5}',
            tool_call_id="call_1",
        ),
        AIMessage(
            "北京今天天气很好！\n\n"
            "🌡 温度：22°C，体感舒适\n"
            "☀️ 天气：晴朗，阳光充足\n"
            "💧 湿度：30%，比较干燥\n"
            "🌬 风力：3-4 级，微风\n"
            "🌧 降雨概率：5%，基本不会下雨\n\n"
            "非常适合去故宫游玩！建议带好防晒霜和饮用水。"
        ),
        # ── 第 2 轮（工具调用）──
        HumanMessage("上海呢？我明天要去上海出差，想知道那边天气如何，需不需要带伞？"),
        AIMessage(content="", tool_calls=[
            {"name": "get_weather", "args": {"location": "上海"}, "id": "call_2"},
        ]),
        ToolMessage(
            '{"temperature": 25, "condition": "多云转阴", "humidity": 65, '
            '"wind": "2-3级", "rain_probability": 40}',
            tool_call_id="call_2",
        ),
        AIMessage(
            "上海明天以多云转阴为主。\n\n"
            "🌡 温度：25°C，比较温暖\n"
            "⛅ 天气：多云转阴，下午可能转阴\n"
            "💧 湿度：65%，略潮湿\n"
            "🌬 风力：2-3 级，微风\n"
            "🌧 降雨概率：40%，有一定可能下雨\n\n"
            "建议带一把折叠伞，以备不时之需。"
        ),
        # ── 第 3 轮（纯对话）──
        HumanMessage(
            "看你刚才的天气预报，北京和上海两个城市情况很不一样。"
            "从体感舒适度来说，哪个更适合户外活动和游览？"
            "另外你提到上海降雨概率40%，这种概率算高还是算中等？"
        ),
        AIMessage(
            "从体感舒适度来看，两个城市各有特点：\n\n"
            "🏆 **北京更推荐户外活动**\n"
            "北京今天晴天、温度适中（22°C）、湿度低（30%），"
            "非常适合长时间的户外游览。故宫、天坛、颐和园都很合适。\n\n"
            "🏙 **上海适合短时间外出**\n"
            "上海 25°C 温暖但偏潮，多云转阴的天气对户外活动影响不大，"
            "只是要注意下午可能转阴，光线不太适合拍照。\n\n"
            "关于降雨概率：\n"
            "- 0-20%：基本不会下雨\n"
            "- 20-50%：可能有雨，建议带伞 ← 上海的40%属于这个范围\n"
            "- 50-80%：大概率会下雨\n"
            "- 80%+：几乎一定会下雨\n\n"
            "所以上海40%的降雨概率属于中等偏上水平，建议带伞。"
        ),
    ]


def show_msgs(title: str, msgs: list[BaseMessage]):
    """打印消息列表（关注类型和首行，工具调用额外标注）。"""
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")
    for i, m in enumerate(msgs):
        kind = type(m).__name__
        first_line = (m.content or "(空)").split("\n")[0]
        display = (first_line[:55] + "…") if len(first_line) > 55 else first_line
        extra = ""
        if hasattr(m, "tool_calls") and m.tool_calls:
            extra = f"  ⚡ {m.tool_calls[0]['name']}()"
        print(f"  [{i:>2}] {kind:<12} {display}{extra}")
    print(f"  ─── 共 {len(msgs)} 条消息 ───")






"""
trim_messages v2 学习案例
========================
核心：trim_messages 是 token 预算引擎，不感知"轮次"。
      start_on、include_system 只是格式保底，不决定保留多少。

用法：
    python backend/learn_and_test/learn_v2.py
"""
import time
from langchain.messages import trim_messages
from learn_utils import build_conversation, show_msgs, bpe_token_counter, cjk_token_counter
from langchain_core.messages import BaseMessage, HumanMessage


# ═══════════════════════════════════════════════════════════
# 场景 1：Baseline —— 只设 budget，看它怎么裁
# ═══════════════════════════════════════════════════════════
def demo_baseline():
    """用不同的 max_tokens 看 trim_messages 保留哪些消息。"""
    msgs = build_conversation()
    show_msgs("原始对话（11 条）", msgs)

    budgets = [120, 300, 600]
    for budget in budgets:
        result = trim_messages(
            msgs,
            max_tokens=budget,
            token_counter="approximate",
            strategy="last",
            start_on="human",
            include_system=True,
        )
        print(f"\n▸ max_tokens={budget}")
        show_msgs(f"保留 {len(result)} 条", result)


# ═══════════════════════════════════════════════════════════
# 场景 2：按条数裁剪 —— token_counter=len
# ═══════════════════════════════════════════════════════════
def demo_by_count():
    """用 token_counter=len 按消息条数裁剪，观察工具调用轮次是否完整保留。"""
    msgs = build_conversation()
    show_msgs("原始对话（11 条）", msgs)

    # 11 条 = System(1) + 工具轮1(4) + 工具轮2(4) + 纯轮3(2)
    msg_counts = [3, 5, 7, 9]
    for count in msg_counts:
        result = trim_messages(
            msgs,
            max_tokens=count,
            token_counter=len,       # ★ 按条数计数！
            strategy="last",
            start_on="human",
            include_system=True,
        )
        print(f"\n▸ max_tokens={count}（最多保留 {count} 条）")
        show_msgs(f"保留 {len(result)} 条", result)


# ═══════════════════════════════════════════════════════════
# 场景 3：自定义 token_counter —— BPE 精确计数 vs approximate
# ═══════════════════════════════════════════════════════════
def demo_custom():
    """用 DeepSeek BPE 做精确 token 计数，对比 approximate 的差异（含耗时）。"""
    msgs = build_conversation()
    show_msgs("原始对话（11 条）", msgs)

    budget = 300
    kwargs = dict(strategy="last", start_on="human", include_system=True)

    # ── 预热（消除模块加载、缓存 Miss 的影响） ──
    print("\n⏳ 预热中 ...")
    for _ in range(3):
        trim_messages(msgs, max_tokens=50, token_counter=bpe_token_counter, **kwargs)

    # ── 计时对比 ──
    N = 50
    counters = [
        ("approximate", "approximate"),
        ("CJK 感知",   cjk_token_counter),
        ("BPE 精确",   bpe_token_counter),
    ]
    for name, counter in counters:
        start = time.perf_counter()
        for _ in range(N):
            result = trim_messages(msgs, max_tokens=budget, token_counter=counter, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"\n▸ max_tokens={budget}, token_counter={name}")
        print(f"   耗时: {elapsed/N*1000:.2f} ms / 次（共 {N} 次）")
        show_msgs(f"保留 {len(result)} 条", result)


# ═══════════════════════════════════════════════════════════
# 场景 4：自定义计量单位 —— 用 HumanMessage 数当"token"
# ═══════════════════════════════════════════════════════════
def demo_round_counter():
    """用 HumanMessage 数量做计量单位，让 max_tokens 直接对应"保留几轮"。

    原理：自定义 token_counter 不一定要算 token，它可以重新定义
     trim_messages 的"计量单位"。把"轮次"作为单位传给 max_tokens。
    """
    def round_counter(msgs: list[BaseMessage]) -> int:
        """批量模式：以 HumanMessage 数量为计量单位。

        1 个 HumanMessage = 1 轮，SystemMessage 计 0。
        这样 max_tokens=3 → 保留最近 3 轮，不管工具调用用了几条消息。
        """
        return sum(1 for m in msgs if isinstance(m, HumanMessage))

    msgs = build_conversation()
    show_msgs("原始对话（11 条，3 轮）", msgs)

    for rounds in [1, 2, 3]:
        result = trim_messages(
            msgs,
            max_tokens=rounds,
            token_counter=round_counter,  # ★ 计量单位变成了"轮"！
            strategy="last",
            start_on="human",
            include_system=True,
        )
        print(f"\n▸ max_tokens={rounds}（保留 {rounds} 轮）")
        human_count = sum(1 for m in result if isinstance(m, HumanMessage))
        show_msgs(f"保留 {len(result)} 条（{human_count} 轮）", result)


if __name__ == "__main__":
    demo_round_counter()
