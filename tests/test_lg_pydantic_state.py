"""
测试：LangGraph 能否自动处理 Pydantic BaseModel 作为 State？

测试结论（LangGraph 1.2.7 + Pydantic 2.13）：

  可以通过 ── Pydantic BaseModel 作为 StateGraph 的 schema ✓
  可以通过 ── State 字段的 default / default_factory ✓
  可以通过 ── Nested Pydantic Model ✓
  可以通过 ── Reducer（Annotated + add_messages） ✓
  不能通过 ─ invoke() 返回 dict 而非 Pydantic model（不是 bug，是设计）
  不能通过 ─ 节点参数是 Pydantic 实例，不支持 state["key"] 语法（不是 bug，是特性）
  不能通过 ─ Pydantic field_validator 在节点返回时不触发（Pydantic v2 的 model_copy 行为）
  不能通过 ─ 对 state 对象的原地修改不会体现在输出中

详细分析见每个测试的注释。
"""

import pytest
from typing import Annotated
from datetime import datetime, timezone
from pydantic import BaseModel, Field, field_validator
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import add_messages
from langchain_core.messages import AnyMessage, HumanMessage


# ═══════════════════════════════════════════════════════════════════════
# 测试 1-6：Pydantic State 能做什么
# ═══════════════════════════════════════════════════════════════════════


# ── 1. 基础读写 ───────────────────────────────────────────────────────

def test_pydantic_state_basic_read_write():
    """
    Node 接收 Pydantic 模型实例，用 .attribute 访问字段。
    节点返回 dict → LangGraph 用 model_copy(update=...) 合并到内部 state。
    invoke() 返回的是 dict，不是 Pydantic 模型。
    """
    class S(BaseModel):
        count: int = 0
        tag: str = ""

    def inc(state: S):
        # state 是 Pydantic 模型实例，用 . 访问
        return {"count": state.count + 1}

    graph = StateGraph(S).add_node("inc", inc).add_edge(START, "inc").add_edge("inc", END).compile()

    result = graph.invoke({"count": 10, "tag": "hi"})

    assert result["count"] == 11
    assert result["tag"] == "hi"

    # ★ 关键：invoke() 返回 dict，不是 Pydantic 模型
    assert isinstance(result, dict)
    assert not isinstance(result, S)


def test_pydantic_state_node_receives_model_instance():
    """
    节点参数是 Pydantic 模型实例，不是 dict。
    state["key"] 语法会抛出 TypeError。
    """
    class S(BaseModel):
        count: int = 0

    def node_access_with_brackets(state):
        # 如果 state 是 dict，这能工作；但 Pydantic 模型不可订阅
        return {"count": state["count"] + 1}

    graph = StateGraph(S).add_node("n", node_access_with_brackets).add_edge(START, "n").add_edge("n", END).compile()

    with pytest.raises(TypeError, match="not subscriptable"):
        graph.invoke({})


# ── 2. 默认值 factory ───────────────────────────────────────────────

def test_pydantic_output_only_contains_returned_fields():
    """
    ★ 重要发现：invoke() 输出的 dict 中，只有节点显式返回的字段
    和初始输入中提供的字段会出现。Pydantic 模型的默认值字段
    如果没有被任何节点返回，也不会出现在输出中。
    """
    class S(BaseModel):
        count: int = 0
        items: list[str] = Field(default_factory=list)
        kept: str = "always"

    def add_one(state: S):
        state.items.append("LOST")  # 原地修改——不会体现在输出
        return {"count": state.count + 1}

    graph = StateGraph(S).add_node("a", add_one).add_edge(START, "a").add_edge("a", END).compile()
    # 输入 {} → Pydantic 构造全默认值对象 → 节点返回 {"count": 1}
    # → 输出 dict 只有 {"count": 1}
    result = graph.invoke({})

    assert result["count"] == 1
    assert "items" not in result  # 默认字段未在输出中
    assert "kept" not in result   # 默认字段未在输出中

    # 但如果输入中提供了，就会在输出中
    result2 = graph.invoke({"items": ["x"], "kept": "yes"})
    assert result2["items"] == ["x"]
    assert result2["kept"] == "yes"


# ── 3. Pydantic 校验只在初始输入时触发 ──────────────────────────

def test_pydantic_validation_on_initial_input():
    """
    初始输入通过 Pydantic 的 __init__ 构造 → 校验生效。
    """
    class S(BaseModel):
        score: float = 0.0

        @field_validator("score")
        @classmethod
        def nonneg(cls, v: float) -> float:
            if v < 0:
                raise ValueError("score must be >= 0")
            return v

    def noop(state: S):
        return {}

    graph = StateGraph(S).add_node("n", noop).add_edge(START, "n").add_edge("n", END).compile()

    # 合法输入通过
    graph.invoke({"score": 10})

    # 非法输入在初始构造时抛出
    with pytest.raises(ValueError, match="score must be >= 0"):
        graph.invoke({"score": -1})


def test_pydantic_validator_does_not_fire_on_node_return():
    """
    ★ 重要发现：Pydantic field_validator 不会在节点返回时触发。

    原因：LangGraph 内部用 Pydantic model_copy(update=node_output) 合并状态，
    而 Pydantic v2 的 model_copy(update=...) 默认不运行校验器。
    （__init__ 会校验，直接赋值 + validate_assignment=True 也会校验，
    但 model_copy 不会。）
    """
    class S(BaseModel):
        score: float = 0.0

        @field_validator("score")
        @classmethod
        def nonneg(cls, v: float) -> float:
            if v < 0:
                raise ValueError("score must be >= 0")
            return v

    def bad_node(state: S):
        # 返回非法值 —— 校验不会触发，不会报错！
        return {"score": -1}

    graph = StateGraph(S).add_node("bad", bad_node).add_edge(START, "bad").add_edge("bad", END).compile()

    # 节点返回了非法值，但不会报错——model_copy 不校验
    result = graph.invoke({"score": 10})
    assert result["score"] == -1  # 非法值被接受了


# ── 4. Nested Pydantic Model ────────────────────────────────────────

def test_nested_pydantic_models():
    """Pydantic 嵌套 model 作为 State 字段能正常读写。"""
    class Address(BaseModel):
        city: str = ""
        zip_code: str = ""

    class Profile(BaseModel):
        name: str = ""
        age: int = 0

    class State(BaseModel):
        address: Address = Field(default_factory=Address)
        profile: Profile = Field(default_factory=Profile)

    def upd(state: State):
        return {
            "address": Address(city="Tokyo", zip_code="100-0001"),
            "profile": Profile(name="Alice", age=30),
        }

    graph = StateGraph(State).add_node("u", upd).add_edge(START, "u").add_edge("u", END).compile()
    result = graph.invoke({})

    # ★ 嵌套 Pydantic model 以对象形式出现在输出 dict 中
    assert result["address"].city == "Tokyo"
    assert result["profile"].name == "Alice"


# ── 5. Reducer ───────────────────────────────────────────────────────

def test_reducer_on_pydantic_field():
    """
    Annotated[list, add_messages] reducer 在 Pydantic 字段上正常工作。
    """
    class S(BaseModel):
        messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
        total: int = 0

    def talk(state: S):
        return {"messages": [HumanMessage(content="hi")]}

    graph = StateGraph(S).add_node("t", talk).add_edge(START, "t").add_edge("t", END).compile()
    result = graph.invoke({"messages": [HumanMessage(content="start")]})

    assert len(result["messages"]) == 2
    assert result["messages"][0].content == "start"
    assert result["messages"][1].content == "hi"


# ── 6. Checkpointer ─────────────────────────────────────────────────

def test_pydantic_state_with_checkpointer():
    """
    Checkpointer（MemorySaver）可以存 Pydantic State。
    get_state() 返回 dict（不是 Pydantic 模型）。

    ★ 注意：invoke(None, config) 对已完成的图是 replay，不是 continue。
    如果要继续执行，需要向图中增加状态或使用 update_state。
    """
    class S(BaseModel):
        count: int = 0

    def add(state: S):
        return {"count": state.count + 1}

    graph = StateGraph(S).add_node("add", add).add_edge(START, "add").add_edge("add", END).compile(
        checkpointer=MemorySaver()
    )

    config = {"configurable": {"thread_id": "t1"}}
    result = graph.invoke({"count": 10}, config)
    assert result["count"] == 11

    # 获取 checkpoint state
    snap = graph.get_state(config)
    assert snap.values["count"] == 11
    assert isinstance(snap.values, dict)

    # invoke(None) replay 产生相同结果
    replayed = graph.invoke(None, config)
    assert replayed["count"] == 11  # replay，不是再 +1


# ═══════════════════════════════════════════════════════════════════════
# 测试 7-8：对比 TypedDict 接口差异
# ═══════════════════════════════════════════════════════════════════════

# ── 7. 接口对比 ───────────────────────────────────────────────────

def test_typeddict_vs_pydantic_access_pattern():
    """
    TypedDict State → 节点参数是 dict → state["key"]
    Pydantic State → 节点参数是 Pydantic 实例 → state.key

    这是两者最主要的接口差异。
    """
    from typing_extensions import TypedDict

    class TDState(TypedDict):
        count: int

    class PDState(BaseModel):
        count: int = 0

    # TypedDict: 用 []
    def td_node(state: TDState):
        return {"count": state["count"] + 1}  # ✓ dict 访问

    td_graph = StateGraph(TDState).add_node("n", td_node).add_edge(START, "n").add_edge("n", END).compile()
    assert td_graph.invoke({"count": 5})["count"] == 6

    # Pydantic: 用 .
    def pd_node(state: PDState):
        return {"count": state.count + 1}  # ✓ 属性访问

    pd_graph = StateGraph(PDState).add_node("n", pd_node).add_edge(START, "n").add_edge("n", END).compile()
    assert pd_graph.invoke({"count": 5})["count"] == 6


# ── 8. 序列化往返 ────────────────────────────────────────────────────

def test_manual_serialization_with_pydantic():
    """
    invoke() 返回 dict，所以用 Pydantic 的 model_validate 可以从 dict 重建。
    """
    class S(BaseModel):
        tags: list[str] = Field(default_factory=list)
        created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def add_tag(state: S):
        return {"tags": ["test"]}

    graph = StateGraph(S).add_node("t", add_tag).add_edge(START, "t").add_edge("t", END).compile()
    result = graph.invoke({})

    # result 是 dict → 用 Pydantic 重建
    restored = S.model_validate(result)
    assert restored.tags == ["test"]
    assert isinstance(restored.created_at, datetime)


# ═══════════════════════════════════════════════════════════════════════
# 汇总说明
# ═══════════════════════════════════════════════════════════════════════
#
# LangGraph 1.2.7 + Pydantic 2.13 的行为汇总：
#
# 1. ✅ Pydantic BaseModel 可以作为 StateGraph(schema) — 但返回 dict
# 2. ✅ default / default_factory — 工作正常
# 3. ✅ Nested Pydantic — 工作正常
# 4. ✅ Reducer 注解（Annotated + add_messages）— 工作正常
# 5. ✅ 初始输入校验（__init__ 触发 field_validator）
# 6. ❌ invoke() 不返回 Pydantic 模型，返回 dict
# 7. ❌ 节点参数不支持 state["key"]，要用 state.key
# 8. ❌ field_validator 不触发在节点返回的值上（model_copy 不校验）
# 9. ❌ 原地修改 state.list.append(x) 丢失在输出中
# 10. ⚠️ get_state().values 返回 dict
# 11. ⚠️ model.model_validate(dict) 可以手动重建 Pydantic 对象
#
# 根本原因：
#   - LangGraph 内部通过 model_copy(update=node_output) 合并状态
#   - Pydantic v2 的 model_copy(update=...) 不做校验
#   - invoke() 将最终状态转为纯 dict 返回
#   - node 参数是真正的 Pydantic 实例，不是 dict
