# backend/core/retry.py
"""统一重试机制：基于 tenacity 的三层兜底（自动重试 → Agent 降级 → 系统兜底）。

设计要点：
1. 异常分类谓词 _is_retryable：尊重 RetryableError / NonRetryableError 标记基类，
   并遍历 __cause__ 链识别 status_code，兼容 LangChain 对底层 SDK 异常的包装。
2. 指数退避 + 抖动（wait_exponential_jitter）：避免惊群效应，应对持续故障。
3. 单次超时与重试解耦：asyncio.wait_for 包在内部函数，tenacity 只管重试决策。
4. 三层兜底：重试耗尽 → Agent 级降级 → 系统级兜底，保证用户始终能收到响应。
"""

import asyncio
from functools import wraps
from typing import Any, Callable, TypeVar

from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from backend.core.exceptions import NonRetryableError, RetryableError
from backend.core.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

# 默认重试参数
DEFAULT_MAX_ATTEMPTS = 3            # 最大尝试次数（含首次调用）
DEFAULT_INITIAL_WAIT = 1.0          # 指数退避起点（首次重试等待秒数）
DEFAULT_MAX_WAIT = 10.0             # 单次重试最大等待秒数
DEFAULT_TIMEOUT_PER_ATTEMPT = 30.0  # 单次调用超时秒数


def _is_retryable(exc: BaseException) -> bool:
    """判断异常是否可重试（tenacity retry 谓词）。

    判断策略（按优先级短路求值）：
        1. asyncio.CancelledError → 不可重试（协程取消语义，重试无意义）
        2. NonRetryableError 子类 → 不可重试（显式声明，最高优先级）
        3. RetryableError 子类 → 可重试（显式声明）
        4. 遍历 __cause__ 链（最多 5 层，防无限循环）查 status_code：
           - 429（Rate Limit）→ 可重试
           - 5xx（服务器错误）→ 可重试
           - 4xx（客户端错误，非 429）→ 不可重试
        5. 异常类名含 Timeout / Connect / Connection → 可重试
        6. 默认 → 不可重试（安全默认，避免无意义重试浪费配额）

    为什么遍历 __cause__ 链而非只看当前异常：
        LangChain 会把底层 SDK 异常（如 openai.RateLimitError）包装成
        自己的异常类型，原始 status_code 保存在 __cause__ 链中。

    为什么用类名匹配而非 isinstance：
        避免在 core 层导入特定 SDK 的异常类（依赖倒置）。
    """
    # 1. 协程取消不重试
    if isinstance(exc, asyncio.CancelledError):
        return False
    # 2/3. 标记基类
    if isinstance(exc, NonRetryableError):
        return False
    if isinstance(exc, RetryableError):
        return True

    # 4. 遍历 __cause__ 链查 status_code
    current: BaseException | None = exc
    depth = 0
    while current is not None and depth < 5:
        status_code = getattr(current, "status_code", None)
        if status_code is None:
            status_code = getattr(current, "http_status", None)
        if status_code is not None:
            if status_code == 429 or 500 <= status_code < 600:
                return True
            if 400 <= status_code < 500:
                return False
        current = current.__cause__
        depth += 1

    # 5. 类名匹配超时/连接错误
    exc_type_name = type(exc).__name__
    if any(kw in exc_type_name for kw in ("Timeout", "Connect", "Connection")):
        return True

    # 6. 安全默认
    return False


def _make_before_sleep(agent_type: str = "") -> Callable[[RetryCallState], None]:
    """创建 tenacity before_sleep 回调：记录重试日志。

    before_sleep 在「决定重试后、开始等待前」触发，此时：
        - attempt_number：已尝试次数（含本次失败）
        - next_action.sleep：即将等待的秒数
        - outcome.exception()：触发重试的异常
    """
    def _before_sleep(retry_state: RetryCallState) -> None:
        exc = retry_state.outcome.exception()
        wait_seconds = retry_state.next_action.sleep
        logger.warning(
            "retry.attempt_failed",
            agent_type=agent_type,
            attempt=retry_state.attempt_number,
            wait_seconds=round(wait_seconds, 1),
            error=str(exc),
            error_type=type(exc).__name__,
        )
    return _before_sleep


def create_retry_decorator(
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    initial_wait: float = DEFAULT_INITIAL_WAIT,
    max_wait: float = DEFAULT_MAX_WAIT,
    agent_type: str = "",
) -> Callable[[Callable], Callable]:
    """创建配置好的 tenacity 装饰器（工厂函数）。

    用于「只需重试、不需降级」的场景（如 RAGChain 内部的 LLM 调用）。
    封装 tenacity 配置细节，调用方只需指定业务参数。

    Args:
        max_attempts: 最大尝试次数（含首次调用），默认 3
        initial_wait: 指数退避起点（首次重试等待秒数），默认 1
        max_wait: 单次重试最大等待秒数，默认 10
        agent_type: Agent 类型，用于日志标识

    Returns:
        装饰器函数，可直接 @装饰 目标函数

    用法：
        decorator = create_retry_decorator(max_attempts=3)

        @decorator
        async def call_llm(): ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            return await AsyncRetrying(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential_jitter(initial=initial_wait, max=max_wait),
                retry=retry_if_exception(_is_retryable),
                before_sleep=_make_before_sleep(agent_type),
                reraise=True,
            )(func)(*args, **kwargs)
        return wrapper
    return decorator


def retry(
    func: Callable[..., T],
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    initial_wait: float = DEFAULT_INITIAL_WAIT,
    max_wait: float = DEFAULT_MAX_WAIT,
    agent_type: str = "",
) -> Callable[..., T]:
    """便捷函数：用重试逻辑包装给定函数（运行时动态包装场景）。

    用于在 __init__ 中包装实例方法（装饰器在类定义时绑定，
    无法用于运行时才创建的方法）。

    用法：
        self._retryable_invoke = retry(self._prompt_llm_chain.invoke, max_attempts=3)
    """
    return create_retry_decorator(
        max_attempts=max_attempts,
        initial_wait=initial_wait,
        max_wait=max_wait,
        agent_type=agent_type,
    )(func)


def with_retry(agent_type: str = ""):
    """三层兜底装饰器工厂：自动重试 → Agent 降级 → 系统兜底。

    用于 Agent 主入口（如 graph.ainvoke），失败时按 agent_type 降级。

    用法：
        @with_retry(agent_type="qa")
        async def _invoke():
            return await graph.ainvoke(state, config=config)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:

            async def _call_with_timeout() -> Any:
                # 单次超时与重试解耦：超时抛 TimeoutError → 谓词判断可重试
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=DEFAULT_TIMEOUT_PER_ATTEMPT,
                )

            # ── 第一层：tenacity 自动重试 ──────────────────────
            last_error: Exception | None = None
            try:
                return await AsyncRetrying(
                    stop=stop_after_attempt(DEFAULT_MAX_ATTEMPTS),
                    wait=wait_exponential_jitter(
                        initial=DEFAULT_INITIAL_WAIT, max=DEFAULT_MAX_WAIT
                    ),
                    retry=retry_if_exception(_is_retryable),
                    before_sleep=_make_before_sleep(agent_type),
                    reraise=True,
                )(_call_with_timeout)
            except NonRetryableError as e:
                # 不可重试异常：立即抛出，不进降级
                logger.warning("retry.non_retryable_error", agent_type=agent_type, error=str(e))
                raise
            except Exception as e:
                # 重试耗尽：记录后进入降级
                last_error = e
                logger.error("retry.all_attempts_failed", agent_type=agent_type, error=str(e))

            # ── 第二层：Agent 级降级 ──────────────────────────
            try:
                fallback_result = await AgentFallbackHandler.handle(
                    agent_type=agent_type, original_error=last_error,
                    func=func, args=args, kwargs=kwargs,
                )
                logger.info("retry.fallback_succeeded", agent_type=agent_type)
                return fallback_result
            except Exception as fallback_error:
                logger.error("retry.fallback_failed", agent_type=agent_type, error=str(fallback_error))

            # ── 第三层：系统级兜底 ────────────────────────────
            logger.error("retry.system_fallback", agent_type=agent_type, original_error=str(last_error))
            return _system_fallback_response(agent_type)

        return wrapper
    return decorator


class AgentFallbackHandler:
    """第二层降级：各 Agent 的专项降级策略（尽量保留核心功能，退化为更简单的实现）。"""

    @classmethod
    async def handle(cls, agent_type, original_error, func, args, kwargs) -> Any:
        """根据 agent_type 选择对应的降级策略。"""
        fallback_map = {                              # 类型 → 降级方法 的映射表
            "qa":               cls._qa_fallback,
            "exam_code":        cls._exam_code_fallback,
            "exam_subjective":  cls._exam_subjective_fallback,
            "resume":           cls._resume_fallback,
            "interview":        cls._interview_fallback,
        }
        handler = fallback_map.get(agent_type)        # 查表
        if handler:
            return await handler(original_error, func, args, kwargs)
        raise original_error                          # 没有对应降级策略，原样抛出（交给系统兜底）

    @classmethod
    async def _qa_fallback(cls, error, func, args, kwargs) -> dict:
        """问答降级：跳过 RAG 检索，直接用大模型的参数知识回答，并加 ⚠️ 提示。"""
        from backend.core.llm_factory import get_llm   # 局部 import，避免循环依赖
        from langchain_core.messages import AIMessage
        logger.info("fallback.qa_direct_llm")

        # 尝试从调用参数里取出 state（直接传参场景）；闭包场景取不到则为空
        state = args[0] if args else kwargs.get("state", {})
        messages = state.get("messages", [])
        if not messages:                              # 取不到对话消息，无法有效降级
            raise ValueError("qa_fallback: messages unavailable, escalating to system fallback")

        llm = get_llm("qa")
        response = await llm.ainvoke(messages)        # 直接问大模型（不查知识库）
        fallback_content = (
            "⚠️ 知识库检索暂时不可用，以下为 AI 直接生成的回答，仅供参考，建议与教师确认：\n\n"
            + (response.content if hasattr(response, "content") else str(response))
        )
        return {
            "messages": messages + [AIMessage(content=fallback_content)],
            "fallback_used": True,                    # 标记：本次走了降级
            "structured_output": None,
        }

    @classmethod
    async def _exam_code_fallback(cls, error, func, args, kwargs) -> dict:
        """代码批改降级：评分服务暂时不可用时，标记需教师人工复核。"""
        logger.info("fallback.exam_code_basic")
        state = args[0] if args else kwargs.get("state", {})
        return {
            **state,                                  # ** 把原 state 展开合并进来
            "fallback_used": True,
            "needs_teacher_review": True,
            "fallback_note": "代码评分服务暂时不可用，已标记为需教师人工复核。",
        }

    @classmethod
    async def _exam_subjective_fallback(cls, error, func, args, kwargs) -> dict:
        """简答题批改降级：标记需教师复核。"""
        logger.info("fallback.exam_subjective_basic")
        state = args[0] if args else kwargs.get("state", {})
        return {
            **state,
            "fallback_used": True,
            "needs_teacher_review": True,
            "fallback_note": "AI 评分服务暂时不可用，已标记为需教师人工批改。",
        }

    @classmethod
    async def _resume_fallback(cls, error, func, args, kwargs) -> dict:
        """简历审查降级：提示服务不可用 / 检查文件。"""
        logger.info("fallback.resume_service_unavailable")
        return {
            "fallback_used": True,
            "content": "简历审查服务暂时不可用，请稍后重试。如持续失败，请检查上传的 PDF 文件是否完整。",
            "structured_output": None,
        }

    @classmethod
    async def _interview_fallback(cls, error, func, args, kwargs) -> dict:
        """面试降级：跳过深度分析，返回基础反馈。"""
        logger.info("fallback.interview_basic_feedback")
        return {
            "fallback_used": True,
            "content": "面试评估服务暂时不可用，已记录本次面试对话，请稍后查看报告。",
            "structured_output": None,
        }


def _system_fallback_response(agent_type: str) -> dict:
    """第三层：系统级兜底。所有降级都失败后返回它，保证用户始终能收到响应。"""
    messages = {                                      # 按 agent_type 给不同的友好提示
        "qa":        "非常抱歉，智能问答服务暂时不可用，请稍后再试，或直接联系教师提问。",
        "exam":      "非常抱歉，试卷批改服务暂时不可用，您的提交已保存，待服务恢复后将自动处理。",
        "resume":    "非常抱歉，简历审查服务暂时不可用，请稍后重新上传。",
        "interview": "非常抱歉，模拟面试服务暂时不可用，请稍后重新开始。",
    }
    content = messages.get(agent_type, "服务暂时不可用，请稍后再试。")  # 找不到就用通用提示
    return {
        "messages": [],
        "content": content,
        "fallback_used": True,
        "system_fallback": True,                      # 标记：走到了最后一层系统兜底
        "structured_output": None,
    }


__all__ = [
    "create_retry_decorator",
    "retry",
    "with_retry",
]
