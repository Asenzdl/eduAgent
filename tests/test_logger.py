import asyncio
from backend.core import configure_logging, get_logger
import structlog

# =====================================================================
# 1. 初始化全局日志配置
# =====================================================================
configure_logging()

# =====================================================================
# 2. 同步业务日志 & 脱敏功能测试
# =====================================================================
print("\n" + "="*70)
print(" 🛠️  第一部分：同步日志、异常堆栈与敏感词脱敏测试".center(60))
print("="*70)

sync_logger = get_logger("backend.sync_test")

# 测试普通业务字段与长敏感词脱敏 (保留前2后2)
sync_logger.info(
    "user.login_attempt", 
    user_id="user_9527", 
    api_key="sk-prod-authentication-key-xyz123"
)

# 测试短敏感词脱敏 (小于4位全部变 ****)
sync_logger.warning(
    "credential.weak_detected", 
    user="admin", 
    pwd="123"
)

# 测试异常自动捕获与堆栈格式化 (exc_info=True)
try:
    print("\n[模拟业务触发异常] 正在尝试除以 0...")
    1 / 0
except ZeroDivisionError:
    sync_logger.error("math.calculation_failed", task_id="calc_job_01", exc_info=True)


# =====================================================================
# 3. 异步多并发 / 智能体隔离测试
# =====================================================================
print("\n" + "="*70)
print(" 🚀 第二部分：异步并发日志测试（观察 trace_id 是否保持隔离，绝不串流）".center(60))
print("="*70)

async_logger = get_logger("backend.async_test")

async def worker(worker_id: str, delay: float, sensitive_token: str):
    # 绑定当前协程上下文的变量（常用于 Multi-Agent 的会话隔离或请求 Trace 追踪）
    structlog.contextvars.bind_contextvars(trace_id=f"req-{worker_id}")
    
    # 打印启动日志，附带一个敏感字段用于顺便做异步脱敏测试
    async_logger.info("task.started", stage="init", token=sensitive_token)
    
    # 模拟异步 I/O 挂起。此时单线程事件循环会切去执行另一个并发的 worker
    await asyncio.sleep(delay)
    
    # 跨越 await 异步流切回后，structlog 依然能精准认出当前协程上下文绑定的 trace_id
    async_logger.info("task.completed", status="success")

async def main_async():
    # 并发执行两个不同延迟的任务
    # Worker A 耗时更长，意味着在它 sleep 期间，Worker B 会穿插打印日志
    await asyncio.gather(
        worker(worker_id="Agent_A", delay=0.3, sensitive_token="secret-aaa-111222"),
        worker(worker_id="Agent_B", delay=0.1, sensitive_token="token-bbb-333444")
    )
    print("="*70 + "\n🎉 恭喜！同步与异步日志全链路测试完成！\n")

# 启动异步事件循环
asyncio.run(main_async())