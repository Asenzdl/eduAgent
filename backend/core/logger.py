"""结构化日志配置模块。

设计意图：
    基于 structlog 实现生产级结构化日志，核心关注点：
    1. JSON 格式输出：便于 ELK/Loki 采集和分析
    2. request_id 上下文绑定：单次请求全链路追踪
    3. 敏感信息脱敏：日志中不暴露 API Key 等敏感数据
    4. 开发/生产双模式：开发用 ConsoleRenderer，生产用 JSONRenderer

为什么用 structlog 而非标准库 logging：
    1. 原生支持键值对日志（logger.info("事件", key=value)），
       标准 logging 需要格式化字符串或 loguru 风格的占位符
    2. 处理器链架构：每个处理器关注一个维度（时间戳、脱敏、格式化），
       符合单一职责原则
    3. contextvars 原生支持：零侵入的 request_id 传播
    4. LangGraph 生态已内置 structlog

为什么 structlog 的 stdlib 模式：
    stdlib 模式让 structlog 和标准库 logging 协作：
    - 第三方库（如 httpx、chromadb）使用 logging 的日志也会
      经过 structlog 的处理器链，格式统一
    - structlog.get_logger() 返回的 logger 同时支持两种风格
"""

import logging
import sys
import uuid

import structlog

# 敏感字段名集合（全部小写，匹配时忽略大小写）
# 为什么用 frozenset：不可变，查找 O(1)，语义表明这是常量
_SENSITIVE_KEYS: frozenset[str] = frozenset({
    "api_key", "apikey", "key",        # API 密钥
    "password", "passwd", "pwd",       # 密码
    "token", "access_token",           # 认证令牌
    "secret", "api_secret",            # 密钥
    "authorization",                   # HTTP Authorization 头
})


def _sanitize_processor(
    logger, method_name: str, event_dict: dict
) -> dict:
    """structlog 处理器：自动脱敏敏感字段。

    设计意图：
        在日志输出前的最后一道防线，确保任何包含敏感字段名的
        键值对都被遮盖，无论业务代码是否主动脱敏。

    为什么作为处理器而非在业务代码中手动脱敏：
        1. 统一保证：即使业务代码遗漏，处理器兜底
        2. 零侵入：业务代码正常传递完整值，脱敏在输出层完成
        3. 可独立测试：处理器可单独测试，无需覆盖每个调用点

    脱敏策略：
        - 字段值长度 > 4：保留前2后2，中间 ****
          示例："sk-d2874cb013704833982de93d9387701f" → "sk****1f"
        - 字段值长度 ≤ 4：整体 ****
          示例："key1" → "****"

    注意：
        此处理器应在 JSONRenderer 之前执行，确保脱敏后的值
        进入最终输出。在 setup_logging 的处理器链中，
        _sanitize_processor 排在 renderer 之前。

    Args:
        logger: structlog 内部 logger 对象（不使用）
        method_name: 日志方法名（不使用）
        event_dict: 事件字典，包含所有键值对日志数据

    Returns:
        脱敏后的事件字典（原地修改，返回引用）
    """
    # 步骤 1：遍历 event_dict 的所有键
    for key in list(event_dict.keys()):
        # 步骤 2：键名转小写后检查是否在敏感字段集合中
        if key.lower() in _SENSITIVE_KEYS:
            # 步骤 3：获取原始值并转为字符串
            value = str(event_dict[key])
            # 步骤 4：根据长度选择脱敏方式
            if len(value) > 4:
                # 保留前2后2，中间用 **** 替换
                event_dict[key] = value[:2] + "****" + value[-2:]
            else:
                # 过短则全部遮盖
                event_dict[key] = "****"
    # 步骤 5：返回修改后的事件字典
    return event_dict

from backend.config import get_settings

_LOGGING_CONFIGURED: bool = False

def configure_logging() -> None:
    """配置全局结构化日志。

    设计意图：
        一次调用完成 structlog 和标准库 logging 的双重配置，
        确保两者协作一致（structlog 使用 stdlib 模式）。

    调用时机：
        - CLI 入口（Task 1.8）：程序启动时调用一次
        - FastAPI 入口（Task 5.1）：app.on_event("startup") 中调用
        - 测试：不需要调用（使用 structlog 默认配置）

    Args:
        level: 日志级别，默认 "INFO"
            "DEBUG"：开发调试，输出所有日志
            "INFO"：生产默认，输出 info/warning/error
            "WARNING"：精简模式，仅输出 warning/error
        json_format: 是否输出 JSON 格式，默认 True
            True → 生产环境，便于 ELK/Loki 采集
            False → 开发环境，使用 ConsoleRenderer（带颜色、对齐）

    配置后的日志输出示例（JSON 模式）：
        {
            "event": "LLM 调用失败，准备重试",
            "request_id": "a1b2c3d4e5f6",
            "timestamp": "2026-04-17T10:30:00.123456+08:00",
            "level": "warning",
            "logger_name": "src.utils.retry",
            "attempt": 2,
            "wait_seconds": 4.0,
            "error": "Rate limit exceeded",
            "error_type": "RateLimitError"
        }
    """
    
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    
    # 步骤 1：配置标准库 logging
    # 为什么需要 force=True：防止已被其他库配置过的 logging 设置残留
    # format="%(message)s"：让 structlog 控制格式，标准库不额外包装
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        # stream=sys.stderr,
        level=level,
        force=True,
    )

    # 步骤 2：设置第三方库的日志级别
    # 为什么需要设置：httpx、chromadb 等库的日志级别默认 INFO，
    # 在生产环境中过于嘈杂，设置为 WARNING 仅显示警告以上
    for noisy_logger in ("httpx", "httpcore", "sqlalchemy.engine", "sqlalchemy.pool"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    # 步骤 3：构建 structlog 处理器链
    # 处理器按顺序执行，每个处理器接收并返回事件字典
    processors = [
        # 3a. 合并 contextvars 中的异步上下文变量（如 request_id）
        structlog.contextvars.merge_contextvars,
        # 3b. 根据日志级别过滤（与标准库 logging 协作）
        structlog.stdlib.filter_by_level,
        # 3c. 添加 logger_name 字段（来自 structlog.get_logger(__name__)）
        structlog.stdlib.add_logger_name,
        # 3d. 添加 level 字段（info/warning/error 等）
        structlog.stdlib.add_log_level,
        # 3e. 添加 timestamp 字段（ISO 8601 格式）
        # 为什么用 ISO 格式：便于 ELK/Loki 自动解析为时间类型
        # structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.TimeStamper(fmt="iso", utc=False),
        # 3f. 渲染堆栈信息（异常发生时自动附加）
        structlog.processors.StackInfoRenderer(),
        # 3g. 格式化异常信息（将 exc_info 转为可读字符串）
        structlog.processors.format_exc_info,
        # 3h. Unicode 解码（确保中文日志正确输出）
        structlog.processors.UnicodeDecoder(),
        # 3i. 敏感信息脱敏（在渲染前最后处理）
        _sanitize_processor,
    ]

    # 步骤 4：选择渲染器
    if settings.json_format:
        # JSON 渲染器：生产环境，输出 JSON Lines（每行一个 JSON 对象）
        renderer = structlog.processors.JSONRenderer()
    else:
        # 控制台渲染器：开发环境，带颜色和对齐
        renderer = structlog.dev.ConsoleRenderer(pad_event_to=20)

    # 步骤 5：配置 structlog
    # wrapper_class=BoundLogger：让 structlog.get_logger() 返回的 logger
    #   支持 info/warning/error 等标准方法
    # context_class=dict：上下文数据存储为普通字典（简洁够用）
    # logger_factory=PrintLoggerFactory：输出到 stdout（便于 Docker 采集）
    # cache_logger_on_first_use=True：首次使用后缓存配置（性能优化）
    structlog.configure(
        processors=processors + [renderer],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _LOGGING_CONFIGURED = True

def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """对外的工厂函数：每个模块用 get_logger(__name__) 拿到自己的日志器。
    返回的 BoundLogger 原生支持「事件名 + 任意键值对」的结构化写法。"""
    return structlog.get_logger(name)



def bind_request_id(request_id: str | None = None) -> str:
    """绑定 request_id 到日志上下文。

    设计意图：
        为当前协程/线程的所有日志自动注入 request_id，
        无需在每个 logger.info() 调用中手动传递。

    为什么用 structlog.contextvars 而非手动传递：
        手动传递需要在每个函数签名中添加 request_id 参数（侵入式），
        contextvars 在协程间自动隔离，零侵入。

    Args:
        request_id: 若不提供则自动生成 12 位十六进制 UUID
            为什么是 12 位：UUID4 共 32 位十六进制字符，
            12 位提供约 2^48 = 281 万亿种组合，碰撞概率极低，
            且在日志中不会过长影响可读性

    Returns:
        绑定的 request_id（便于调用方记录或传递给下游服务）
    """
    # 步骤 1：生成 request_id（若未提供）
    if request_id is None:
        # uuid4().hex 生成 32 位十六进制，取前 12 位
        request_id = uuid.uuid4().hex[:12]

    # 步骤 2：清除旧上下文（防止上一个请求的上下文残留）
    # structlog.contextvars.clear_contextvars()

    # 步骤 3：绑定 request_id 到上下文
    # 绑定后，后续所有 structlog.get_logger() 产出的日志
    # 都会自动包含 request_id 字段
    structlog.contextvars.bind_contextvars(request_id=request_id)

    # 步骤 4：返回 request_id
    return request_id


def unbind_request_id() -> None:
    """清除日志上下文中的 request_id。

    为什么需要显式清除：
        contextvars 在协程复用时（如 FastAPI 的 async 路由）
        可能残留上一个请求的 request_id，导致日志错乱。
        在请求处理完成后调用此函数，确保上下文干净。

    调用时机：
        - CLI：每次问答结束后
        - FastAPI：每个请求的 after_request 钩子中
    """
    # structlog.contextvars.clear_contextvars()
    structlog.contextvars.unbind_contextvars("request_id")

# 使用
# from fastapi import Request, Response
# from starlette.middleware.base import BaseHTTPMiddleware
# import structlog

# class StructlogMiddleware(BaseHTTPMiddleware):
#     async def dispatch(self, request: Request, call_next) -> Response:
#         # 1. 每一个异步请求进来，生成并绑定 request_id
#         # contextvars 会将这个隔离在当前请求的 Task 空间内
#         request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
        
#         # 使用 structlog 自带的异步上下文管理器，请求结束自动清理，绝对不会污染
#         with structlog.contextvars.bound_contextvars(request_id=request_id):
#             response = await call_next(request)
#             response.headers["X-Request-ID"] = request_id
#             return response


__all__ = [
    "bind_request_id",
    "configure_logging",
    "get_logger",
    "unbind_request_id",
]


