# backend/schemas/qa.py
#
# QA Agent 的请求／响应模型。
# 放在 backend/schemas/qa.py 而非 api/v1/qa.py 的原因是：
#   agents/qa/input.py 和 api/v1/qa.py 两个层都需要引用，
#   放在共享层避免 agents → api 的反向依赖。

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id:        str        = Field(..., description="会话 ID")
    course_id:         str | None = Field(None, description="课程 ID（可选，限定检索范围）")
    message:           str        = Field(..., min_length=1, max_length=2000)
    enable_web_search: bool       = Field(False, description="低置信度时是否先走 Web Search 再给 LLM")

    def to_graph_input(self) -> dict:
        """翻译自身为 LangGraph 初始输入。

        返回的 dict 结构与 QAInput 对齐，
        可直接传给 graph.ainvoke()。
        """
        return {
            "messages":           [("user", self.message)],
            "enable_web_search":  self.enable_web_search,
            "original_query":     self.message,
        }


class ChatResponse(BaseModel):
    session_id:    str
    answer:        str
    answer_mode:   str        # "rag" / "web_augmented" / "llm_direct" / "general"
    confidence:    float
    sources:       list[str]
    fallback_used: bool


class SessionMessage(BaseModel):
    role:       str   # "user" / "assistant"
    content:    str
    created_at: str


class HistoryResponse(BaseModel):
    session_id:  str
    messages:    list[SessionMessage]
    summary:     str | None
    total_turns: int
