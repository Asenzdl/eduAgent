from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from backend.core.memory import init_checkpointer, close_checkpointer
from backend.api.v1 import auth_router, resume_router, qa_router


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """FastAPI 应用生命周期：启动时初始化，关闭时释放资源。"""
    # ── 启动 ──────────────────────────────────────────────────
    # 初始化 AsyncPostgresSaver 连接池 + 幂等建表。
    # 在这之后任何位置 import saver 都可安全使用。
    await init_checkpointer()
    yield
    # ── 关闭 ──────────────────────────────────────────────────
    await close_checkpointer()


app = FastAPI(lifespan=lifespan)
app.include_router(auth_router, prefix='/api/v1/auth')
# app.include_router(resume_router, prefix='/api/v1/resume')
app.include_router(qa_router, prefix='/api/v1/qa')


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
