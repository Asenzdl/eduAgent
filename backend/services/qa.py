# backend/services/qa.py
#
# QA Agent 服务层：组合 Graph 构建、输入/输出翻译、异常映射。
#
# 路由层通过 run_qa() 间接调用 Graph，不直接接触 CompiledGraph。
# Graph 实例通过 @cache 缓存，避免每次请求重新编译图。

from fastapi import HTTPException, status

from backend.schemas.qa import ChatRequest, ChatResponse
from backend.dependencies import UserContext
from backend.agents.qa.graph import build_qa_graph
from backend.agents.qa.state import QAContext
from backend.core.logger import get_logger

logger = get_logger(__name__)


# ── LangGraph 线程管理 ──────────────────────────────────────

def build_thread_id(student_id: str, session_id: str) -> str:
    """构造 LangGraph Checkpointer 使用的 thread_id。

    格式：student_{student_id}:session_{session_id}
    同一学员的不同会话有独立的历史，互不干扰。
    """
    return f"student_{student_id}:session_{session_id}"


def build_config(student_id: str, session_id: str) -> dict:
    """构造 LangGraph 调用所需的 config 字典。

    Returns:
        {"configurable": {"thread_id": "student_xxx:session_yyy"}}
    """
    return {
        "configurable": {
            "thread_id": build_thread_id(student_id, session_id),
        }
    }


def build_chat_response(session_id: str, state: dict) -> ChatResponse:
    """从 Graph 最终 State 提取响应字段。

    Args:
        session_id:  会话 ID（原样回传）。
        state:       Graph 执行完毕后的最终 QAState 字典。

    Returns:
        序列化就绪的 API 响应对象。
    """
    return ChatResponse(
        session_id=session_id,
        answer=state.get("answer", ""),
        answer_mode=state.get("answer_mode", "llm_direct"),
        confidence=state.get("confidence", 0.0),
        sources=state.get("sources", []),
        fallback_used=state.get("fallback_used", False),
    )


# ── 主入口 ──────────────────────────────────────────────────

async def run_qa(req: ChatRequest, user: UserContext) -> ChatResponse:
    """执行一次 QA 对话流程：构建输入 → 调用 Graph → 翻译输出。

    Args:
        req:  聊天请求（消息内容、会话 ID、课程限定等）。
        user: 经 JWT 鉴权的用户上下文。

    Returns:
        包含回答、置信度、来源等完整信息的响应对象。

    Raises:
        HTTPException 500: Graph 执行过程中出现未预期异常。
    """
    graph = await build_qa_graph()
    config = build_config(user.user_id, req.session_id)
    context = QAContext(
        student_id=user.user_id,
        tenant_id=user.tenant_id,
        course_id=req.course_id,
    )

    try:
        result = await graph.ainvoke(
            req.to_graph_input(),
            config=config,
            context=context,
        )
    except Exception as e:
        logger.error("qa_service.chat_error", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "AGENT_ERROR", "message": str(e)},
        )

    return build_chat_response(req.session_id, result)
