# backend/dependencies.py（数据库部分）
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from backend.config import get_settings

settings = get_settings()

# ── 创建异步引擎（连接池）──
engine = create_async_engine(
    settings.database_url,        # 来自 config.py，最终来自 .env.local
    pool_size=10,                 # 连接池基础大小
    max_overflow=20,              # 高峰时最多再额外开 20 个连接
    echo=False,                   # True 会打印所有 SQL，调试时可临时打开
)

# ── 会话工厂 ──
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI 依赖：获取异步数据库会话，自动提交 / 回滚 / 关闭"""
    async with AsyncSessionLocal() as session:
        try:
            yield session            # 把会话交给接口使用（回顾 2.1.6 / 2.5）
            await session.commit()   # 接口正常结束 → 自动提交
        except Exception:
            await session.rollback() # 出错 → 自动回滚
            raise

if __name__ == "__main__":
    import asyncio
    from sqlalchemy import text

    async def main():
        # 模拟 FastAPI 调用依赖：迭代一次拿到 session，跑一条查询
        async for db in get_db():
            r = await db.execute(text(
                "SELECT current_database(), "
                "count(*) FROM information_schema.tables WHERE table_schema = :s"
            ), {"s": "public"})
            row = r.fetchone()
            print("get_db 连到库:", row[0])
            print("public 下表数量:", row[1])

    asyncio.run(main())
    print("get_db 测试通过 ✅")
