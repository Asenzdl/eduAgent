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
