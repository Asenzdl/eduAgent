# backend/core/exceptions.py
# EduAgent 统一异常体系：所有自定义异常都继承同一个基类，
# 便于「一次捕获全部」以及按「可重试 / 不可重试」分类处理。
#
# 设计要点：
#   RetryableError / NonRetryableError 是标记基类，由具体异常继承。
#   这样 tenacity 的 retry_if_exception 只需检查标记基类（isinstance），
#   无需在重试层维护「异常元组」，新增异常时只需选对父类即可。


class EduAgentBaseError(Exception):
    """所有 EduAgent 自定义异常的基类（继承 Python 内置的 Exception）。
    比普通异常多带两样上下文：是哪个 Agent 出的错、以及任意细节字典。"""
    def __init__(self, message: str, agent_type: str = "", details: dict = None):
        # message：错误描述文字；agent_type：哪个 Agent（qa/exam/...）；details：额外细节
        super().__init__(message)          # 调用父类 Exception 的初始化，把错误消息存好
        self.agent_type = agent_type       # 记录出错的 Agent 类型，方便日志与排查
        self.details = details or {}       # 传 None 时用空字典兜底，避免后续访问报错


class RetryableError(EduAgentBaseError):
    """可重试错误标记基类。

    为什么用标记基类而非 is_retryable 属性：
        1. tenacity 的 retry_if_exception 基于 isinstance 判断，
           继承关系天然支持，无需在每次捕获时检查属性
        2. 异常类型本身携带语义——看到 RetryableError 就知道可重试
        3. 避免在异常实例上添加属性后遗忘设置，导致重试逻辑失效

    使用方式：
        class RateLimitExceeded(RetryableError):
            pass

        # tenacity 自动识别
        @retry(retry=retry_if_exception_type(RetryableError))
        def call_llm(): ...
    """
    pass


class NonRetryableError(EduAgentBaseError):
    """不可重试错误标记基类。

    什么时候应该继承此类：
        1. 认证/授权失败（401/403）— 重试不会让错误的 Key 变正确
        2. 请求格式错误（400）— 重试同样的错误请求无意义
        3. 配额耗尽（402/422）— 重试只会浪费更多配额
        4. 业务逻辑错误 — 如输入非法，重试结果相同
    """
    pass


class LLMAPIError(RetryableError):
    """大模型 API 调用失败（超时 / 限流 / 网络错误）。属于【可重试】异常。"""
    pass


class AgentExecutionError(EduAgentBaseError):
    """Agent 业务逻辑执行失败。"""
    pass


class PipelineError(EduAgentBaseError):
    """多 Agent Pipeline（流水线编排）失败。"""
    pass


class IntentRouteError(EduAgentBaseError):
    """意图识别路由失败（没判断出该交给哪个 Agent）。"""
    pass


class MilvusConnectionError(RetryableError):
    """Milvus 向量库连接失败。属于【可重试】异常。"""
    pass


class FileParseError(EduAgentBaseError):
    """文件解析失败（Word / PDF）。"""
    pass


class InvalidInputError(NonRetryableError):
    """用户输入不合法。属于【不可重试】异常（重试也不会变合法）。"""
    pass


class AuthenticationError(NonRetryableError):
    """认证失败。属于【不可重试】异常。"""
    pass


__all__ = [
    "EduAgentBaseError",
    "RetryableError",
    "NonRetryableError",
    "LLMAPIError",
    "AgentExecutionError",
    "PipelineError",
    "IntentRouteError",
    "MilvusConnectionError",
    "FileParseError",
    "InvalidInputError",
    "AuthenticationError",
]
