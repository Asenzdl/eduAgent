# backend/core/retry_v2.py
"""统一重试机制：tenacity 重试编排 + 可注入降级 + 系统兜底。

职责边界：
    core 仅提供「重试 → 降级 → 兜底」编排骨架，不内置任何 agent 业务。
    agent 级降级策略由外部注入（with_retry_fallback 的 fallback 参数）。
    系统级兜底（纯文案）留在 core，是与业务无关的最后防线。

依赖方向：agent → core（agent 用 core 框架），而非 core → agent。

API 分工：
    with_retry            — 纯重试（内部调用用，耗尽即抛）
    with_retry_fallback   — 重试 + 降级 + 兜底（Agent 入口用）
    register_fallback     — 注册 agent 级降级策略（agent 模块启动时调用）
"""

import asyncio
from functools import wraps
from typing import Any, Awaitable, Callable, TypeVar

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

# 降级函数签名：(state, error) -> result。由 agent 模块定义并注入。
FallbackFn = Callable[[dict | None, BaseException], Awaitable[Any]]

# 默认重试参数
DEFAULT_MAX_ATTEMPTS = 3            # 最大尝试次数（含首次调用）
DEFAULT_INITIAL_WAIT = 1.0          # 指数退避起点（首次重试等待秒数）
DEFAULT_MAX_WAIT = 10.0             # 单次重试最大等待秒数
DEFAULT_TIMEOUT = 30.0              # 单次调用超时秒数


# ── 异常分类谓词 ──────────────────────────────────────────────

def _is_retryable(exc: BaseException) -> bool:
    """判断异常是否可重试（tenacity retry 谓词）。

    判断策略（按优先级短路）：
        1. CancelledError → 不可重试（协程取消语义）
        2. NonRetryableError 子类 → 不可重试（显式声明，最高优先级）
        3. RetryableError 子类 → 可重试
        4. 遍历 __cause__ 链（≤5 层）查 status_code：
           429 / 5xx → 可重试；4xx（非 429）→ 不可重试
        5. 类名含 Timeout / Connect / Connection → 可重试
        6. 默认 → 不可重试（安全默认）

    遍历 __cause__ 链是因为 LangChain 会包装底层 SDK 异常，
    原始 status_code 保存在 __cause__ 链中。
    """
    if isinstance(exc, asyncio.CancelledError):
        return False
    if isinstance(exc, NonRetryableError):
        return False
    if isinstance(exc, RetryableError):
        return True

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

    if any(kw in type(exc).__name__ for kw in ("Timeout", "Connect", "Connection")):
        return True

    return False


# ── tenacity 配置单一来源 ─────────────────────────────────────

def _make_before_sleep(agent_type: str) -> Callable[[RetryCallState], None]:
    """before_sleep 回调：重试前记录日志（attempt / wait / error）。"""
    def _before_sleep(retry_state: RetryCallState) -> None:
        exc = retry_state.outcome.exception()
        logger.warning(
            "retry.attempt_failed",
            agent_type=agent_type,
            attempt=retry_state.attempt_number,
            wait_seconds=round(retry_state.next_action.sleep, 1),
            error=str(exc),
            error_type=type(exc).__name__,
        )
    return _before_sleep


def _retrying(
    agent_type: str,
    *,
    max_attempts: int,
    initial_wait: float,
    max_wait: float,
) -> AsyncRetrying:
    """tenacity 配置的唯一来源。所有重试路径共享，改策略只改一处。"""
    return AsyncRetrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(initial=initial_wait, max=max_wait),
        retry=retry_if_exception(_is_retryable),
        before_sleep=_make_before_sleep(agent_type),
        reraise=True,
    )


def _make_call(func, timeout):
    """构造被重试的内部函数：超时与重试解耦，timeout=None 时不套 wait_for。"""
    async def _call(*args, **kwargs) -> Any:
        if timeout is None:
            return await func(*args, **kwargs)
        return await asyncio.wait_for(func(*args, **kwargs), timeout)
    return _call


def _extract_state(args: tuple, kwargs: dict) -> dict | None:
    """从调用参数提取 state（降级策略需要）。约定：首个位置参数或 kwargs['state']。"""
    if args:
        return args[0]
    return kwargs.get("state")


# ── 系统级兜底（业务无关，留 core）──────────────────────────────

async def _system_fallback(agent_type: str, error: BaseException | None) -> dict:
    """系统级兜底：纯文案，保证用户始终能收到响应。与 agent 业务无关。"""
    messages = {
        "qa":        "非常抱歉，智能问答服务暂时不可用，请稍后再试，或直接联系教师提问。",
        "exam":      "非常抱歉，试卷批改服务暂时不可用，您的提交已保存，待服务恢复后将自动处理。",
        "resume":    "非常抱歉，简历审查服务暂时不可用，请稍后重新上传。",
        "interview": "非常抱歉，模拟面试服务暂时不可用，请稍后重新开始。",
    }
    content = messages.get(agent_type, "服务暂时不可用，请稍后再试。")
    logger.error("retry.system_fallback", agent_type=agent_type, original_error=str(error))
    return {
        "messages": [],
        "content": content,
        "fallback_used": True,
        "system_fallback": True,
        "structured_output": None,
    }


# ── 对外 API ─────────────────────────────────────────────────

def with_retry(
    agent_type: str = "",
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    initial_wait: float = DEFAULT_INITIAL_WAIT,
    max_wait: float = DEFAULT_MAX_WAIT,
    timeout: float | None = DEFAULT_TIMEOUT,
):
    """纯重试装饰器工厂（内部调用用，无降级无兜底，耗尽即抛）。

    用于 LLM / DB 等内部调用场景：只需重试，不需要向用户返回兜底文案。

    Args:
        agent_type: Agent 类型标识，用于日志。
        max_attempts: 最大尝试次数（含首次），默认 3。
        initial_wait: 指数退避起点（秒），默认 1。
        max_wait: 单次重试最大等待（秒），默认 10。
        timeout: 单次调用超时（秒），None 表示不超时（SDK 自管），默认 30。

    用法：
        @with_retry(max_attempts=3)
        async def call_llm(): ...

        # 运行时包装（实例方法）
        self._retryable_invoke = with_retry(max_attempts=3)(self._chain.invoke)
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        _call = _make_call(func, timeout)
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            return await _retrying(
                agent_type,
                max_attempts=max_attempts,
                initial_wait=initial_wait,
                max_wait=max_wait,
            )(_call, *args, **kwargs)
        return wrapper
    return decorator


def with_retry_fallback(
    agent_type: str = "",
    *,
    fallback: FallbackFn | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    initial_wait: float = DEFAULT_INITIAL_WAIT,
    max_wait: float = DEFAULT_MAX_WAIT,
    timeout: float | None = DEFAULT_TIMEOUT,
):
    """重试 + 降级 + 兜底 装饰器工厂（Agent 入口用）。

    三层：tenacity 自动重试 → 外部注入的 agent 降级 → 系统兜底。

    Args:
        agent_type: Agent 类型标识，用于日志和系统兜底文案。
        fallback: Agent 级降级函数 async (state, error) -> result，由 agent 模块注入。
            为 None 时重试耗尽直接走系统兜底。
        max_attempts: 最大尝试次数（含首次），默认 3。
        initial_wait: 指数退避起点（秒），默认 1。
        max_wait: 单次重试最大等待（秒），默认 10。
        timeout: 单次调用超时（秒），None 表示不超时（SDK 自管），默认 30。

    用法：
        # agent 模块定义降级策略
        async def qa_fallback(state, error):
            from backend.core.llm_factory import get_llm
            ...
            return {"messages": [...], "fallback_used": True}

        # 注入给 with_retry_fallback
        @with_retry_fallback("qa", fallback=qa_fallback)
        async def invoke(state, config): ...
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        _call = _make_call(func, timeout)
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # ── 第一层：tenacity 自动重试 ──
            try:
                return await _retrying(
                    agent_type,
                    max_attempts=max_attempts,
                    initial_wait=initial_wait,
                    max_wait=max_wait,
                )(_call, *args, **kwargs)
            except NonRetryableError as e:
                # 不可重试异常：立即抛出，不降级（请求本身有问题，兜底无意义）
                logger.warning("retry.non_retryable_error", agent_type=agent_type, error=str(e))
                raise
            except Exception as e:
                last_error = e
                logger.error("retry.all_attempts_failed", agent_type=agent_type, error=str(e))

            # ── 第二层：Agent 级降级（外部注入）──
            if fallback is not None:
                try:
                    return await fallback(_extract_state(args, kwargs), last_error)
                except Exception as fb_error:
                    logger.error("retry.fallback_failed", agent_type=agent_type, error=str(fb_error))

            # ── 第三层：系统级兜底 ──
            return await _system_fallback(agent_type, last_error)

        return wrapper
    return decorator


__all__ = ["with_retry", "with_retry_fallback"]
