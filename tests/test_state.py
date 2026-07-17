"""
演示：LangGraph Pydantic State + Reducer 自动合并

核心：用 Annotated + Reducer 函数，节点只需返回增量，不用手动拼接。
"""

from typing import Annotated
import operator
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


# ── 带 Reducer 的 Pydantic State ───────────────────────────

class AgentState(BaseModel):
    """
    Reducer 让节点只需返回"增量"：

        messages: Annotated[list[str], operator.add]
            → 节点 return {"messages": ["新消息"]}
            → Reducer 自动追加到已有列表，不用写 state.messages + [...]

        step: int   ← 普通字段（覆盖赋值）
            → 节点 return {"step": 3}
            → 直接覆盖旧值

    对比：
        无 reducer: return {"messages": state.messages + ["新消息"]}
        有 reducer: return {"messages": ["新消息"]}            ← 只需增量
    """
    messages: Annotated[list[str], operator.add] = Field(default_factory=list)
    step: int = 0
    confidence: float = 0.0


# ── 节点（只需返回增量） ─────────────────────────────────

def first_node(state: AgentState):
    print(f"  [first_node]  step={state.step}, msgs_count={len(state.messages)}")
    return {
        "messages": ["第一节点处理完成"],    # ← 不用 state.messages + [...]！
        "step": state.step + 1,
    }


def second_node(state: AgentState):
    confidence = min(1.0, state.confidence + 0.5)
    print(f"  [second_node] step={state.step}, msgs_count={len(state.messages)}")
    return {
        "messages": [f"第二节点: 置信度 {confidence:.1f}"],  # ← 只返回增量
        "step": state.step + 1,
        "confidence": confidence,
    }


def third_node(state: AgentState):
    """可以追加多条消息，一条 return 搞定。"""
    print(f"  [third_node]  step={state.step}, msgs_count={len(state.messages)}")
    return {
        "messages": [
            "第三节点开始",
            "第三节点结束",
        ],
        "step": state.step + 1,
    }


# ── 运行 ──────────────────────────────────────────────────

def main():
    builder = StateGraph(AgentState)
    builder.add_node("first", first_node)
    builder.add_node("second", second_node)
    builder.add_node("third", third_node)
    builder.add_edge(START, "first")
    builder.add_edge("first", "second")
    builder.add_edge("second", "third")
    builder.add_edge("third", END)

    graph = builder.compile()

    print("=" * 55)
    print("Reducer 自动合并演示")
    print("每个节点只返回单条消息，Reducer 自动追加")
    print("=" * 55)
    result = graph.invoke({})

    print("\n■ 最终输出:")
    for i, msg in enumerate(result["messages"], 1):
        print(f"   {i}. {msg}")
    print(f"   step = {result['step']}, confidence = {result['confidence']}")

    # ── 对比：没有 Reducer 的写法 ───────────────────────
    print("\n" + "=" * 55)
    print("对比：没有 Reducer 的话……")
    print("=" * 55)
    print('''
  # 无 Reducer:
  def node(state):
      return {"messages": state.messages + ["新消息"]}
                          ╰──── 手动拼接 ────╯

  # 有 Reducer (operator.add):
  def node(state):
      return {"messages": ["新消息"]}
                          ╰─ 只返回增量 ─╯
    ''')

    print("Reducer 支持的类型：")
    print("  - list + operator.add   → 自动合并列表")
    print("  - int + operator.add    → 自动累加数值")
    print("  - AnyMessage + add_messages → LangChain 消息去重合并")
    print("  - 自定义函数             → 任意合并逻辑")


if __name__ == "__main__":
    main()
