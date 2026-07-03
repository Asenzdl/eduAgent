from backend.core import EduAgentBaseError, LLMAPIError

# 抛出一个带上下文的异常，并用基类捕获
try:
    raise LLMAPIError("DeepSeek 超时", agent_type="qa", details={"timeout": 30})
except EduAgentBaseError as e:
    print("捕获:", type(e).__name__, "| msg:", e, "| agent:", e.agent_type, "| details:", e.details)

print("LLMAPIError 是 EduAgentBaseError 子类:", issubclass(LLMAPIError, EduAgentBaseError))