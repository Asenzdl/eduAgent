# scripts/seed_data.py
# 执行：python scripts/seed_data.py
# 用途：灌入本地开发测试账号

import asyncio
import uuid
import os
import bcrypt                         # 直接使用 bcrypt 库
from dotenv import load_dotenv
import asyncpg

load_dotenv(".env.local")            # 从项目根目录的 .env.local 读配置

# 用环境变量拼出 asyncpg 的连接串
DB_DSN = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', 5432)}"
    f"/{os.getenv('DB_NAME', 'eduagent')}"
)

TENANT_ID = "tenant_default"


async def seed_users():
    """灌入 4 个测试账号（已存在则跳过）。"""
    conn = await asyncpg.connect(DB_DSN)
    print("✅ 数据库连接成功，开始灌入测试账号...")
    try:
        users = [
            {"username": "admin",     "email": "admin@eduagent.local",     "pwd": "Admin@123456",   "role": "admin"},
            {"username": "teacher01", "email": "teacher01@eduagent.local", "pwd": "Teacher@123456", "role": "teacher"},
            {"username": "student01", "email": "student01@eduagent.local", "pwd": "Student@123456", "role": "student"},
            {"username": "student02", "email": "student02@eduagent.local", "pwd": "Student@123456", "role": "student"},
        ]
        for u in users:
            # 使用 bcrypt 生成哈希（需编码为 bytes，结果转为 str 存储）
            password_bytes = u["pwd"].encode("utf-8")
            hashed = bcrypt.hashpw(password_bytes, bcrypt.gensalt()).decode("utf-8")

            await conn.execute(
                """
                INSERT INTO users (id, tenant_id, username, email, password_hash, role)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (tenant_id, email) DO NOTHING
                """,
                str(uuid.uuid4()), TENANT_ID, u["username"], u["email"],
                hashed,
                u["role"],
            )
        print(f"✅ 测试账号灌入完成（{len(users)} 个，已存在则跳过）：")
        print("   admin@eduagent.local      / Admin@123456")
        print("   teacher01@eduagent.local  / Teacher@123456")
        print("   student01@eduagent.local  / Student@123456")
        print("   student02@eduagent.local  / Student@123456")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(seed_users())