# backend/core/llm_factory_v2.py
# LLM Factory v2：统一封装大模型调用，按 Agent 类型路由。
# 规矩：所有 Agent 必须通过此模块获取模型，禁止直接调用 init_chat_model。
#
# v2 变更：
#   - 引入 LLMConfig，支持用户自定义模型（前端存 Key，每次请求携带）
#   - 用 lru_cache 替代手动字典缓存（线程安全 + 自动缓存键 + 命中率统计）
#   - 系统默认走缓存，用户自定义不缓存（Key 每次可能不同，创建成本极低）
#   - 所有提供商均为 OpenAI 兼容接口，参数形状统一

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Type

import httpx
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from backend.config import get_settings
from backend.core.logger import get_logger

logger = get_logger(__name__)

# ── HTTP 客户端（绕过系统代理）──────────────────────────────
_HTTP_ASYNC_CLIENT = httpx.AsyncClient(
    trust_env=False,
    timeout=httpx.Timeout(120.0, connect=15.0),
)
_HTTP_SYNC_CLIENT = httpx.Client(
    trust_env=False,
    timeout=httpx.Timeout(120.0, connect=15.0),
)

# ── Agent 类型 → 模型标识符 的路由表 ──────────────────────────
_AGENT_MODEL_ROUTING: dict[str, str] = {
    "qa":               "deepseek",
    "exam_subjective":  "deepseek",
    "exam_code":        "deepseek",
    "resume":           "deepseek",
    "interview":        "deepseek",
    "intent":           "deepseek",
    "summarize":        "deepseek",
}

# 模型标识符 → API 实际接受的 model 名称
_MODEL_ID_MAP: dict[str, str] = {
    "deepseek": "deepseek-chat",
}


# ═══════════════════════════════════════════════════════════
# 用户自定义配置
# ═══════════════════════════════════════════════════════════

@dataclass(frozen=True)
class LLMConfig:
    """一次 LLM 调用的连接信息。
    frozen=True → 不可变，可做字典键，可安全传递。
    所有字段都是 str → 来自用户输入/API 请求，不做隐式转换。
    前端 localStorage 存储，每次请求携带，后端不持久化。"""
    model_id: str
    api_key: str
    base_url: str


class UserLLMConfig(BaseModel):
    """用户自定义模型配置（Pydantic 版，用于 FastAPI 请求体校验）。
    与 LLMConfig 分离的原因：Pydantic 负责入校验（字段必填、类型正确），
    dataclass 负责内部传递（轻量、不可变、可哈希）。各司其职。"""
    model_id: str = Field(description="模型名称，如 gpt-4o、moonshot-v1-8k")
    api_key: str = Field(description="API Key")
    base_url: str = Field(description="API Base URL，如 https://api.openai.com/v1")

    def to_llm_config(self) -> LLMConfig:
        """转为工厂内部使用的 dataclass。"""
        return LLMConfig(
            model_id=self.model_id,
            api_key=self.api_key,
            base_url=self.base_url,
        )


# ═══════════════════════════════════════════════════════════
# 模型构建（纯函数，无缓存）
# ═══════════════════════════════════════════════════════════

def _create_model(
    model_id: str,
    api_key: str,
    base_url: str,
    temperature: float,
    streaming: bool,
) -> BaseChatModel:
    """创建模型实例。所有提供商均为 OpenAI 兼容接口，参数形状统一。"""
    return init_chat_model(
        model=model_id,
        model_provider="openai",
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        max_retries=0,
        streaming=streaming,
        http_async_client=_HTTP_ASYNC_CLIENT,
        http_client=_HTTP_SYNC_CLIENT,
    )


# ═══════════════════════════════════════════════════════════
# 系统默认配置
# ═══════════════════════════════════════════════════════════

def _get_default_config(model_key: str) -> LLMConfig:
    """从 .env.local 读取系统默认配置。"""
    settings = get_settings()
    _DEFAULT_CONFIGS: dict[str, LLMConfig] = {
        "deepseek": LLMConfig(
            model_id=settings.deepseek_model_chat,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        ),
    }
    return _DEFAULT_CONFIGS[model_key]


# ═══════════════════════════════════════════════════════════
# LLMFactory
# ═══════════════════════════════════════════════════════════

class LLMFactory:
    """大模型工厂（统一获取模型的唯一入口）。

    设计决策：
        - 保留类而非模块函数：工厂有状态（缓存），类让状态归属更清晰，演进更平滑
        - 用 lru_cache 替代手动字典：线程安全 + 自动缓存键 + 命中率统计
        - 系统默认走缓存，用户自定义不缓存：Key 每次可能不同，创建成本极低
        - LLMConfig（dataclass）用于内部传递，UserLLMConfig（Pydantic）用于请求体校验

    用法：
        # 系统默认（走缓存）
        llm = LLMFactory.get_llm("qa")

        # 用户自定义（不缓存）
        llm = LLMFactory.get_llm("qa", llm_config=LLMConfig(
            model_id="gpt-4o", api_key="sk-xxx", base_url="https://api.openai.com/v1"
        ))

        # 结构化输出
        structured = LLMFactory.get_structured_llm("resume", MySchema)

        response = await llm.ainvoke(messages)
    """

    @classmethod
    @lru_cache()
    def _get_or_create_default(
        cls,
        model_key: str,
        temperature: float,
        streaming: bool,
    ) -> BaseChatModel:
        """系统默认模型的缓存获取。
        lru_cache 保证相同 (model_key, temperature, streaming) 只创建一次。
        缓存键由 lru_cache 从参数自动推导，无需手动拼字符串。"""
        config = _get_default_config(model_key)
        model = _create_model(
            model_id=config.model_id,
            api_key=config.api_key,
            base_url=config.base_url,
            temperature=temperature,
            streaming=streaming,
        )
        logger.info(
            "llm_factory.model_initialized",
            source="system_default",
            model_key=model_key,
            model_id=config.model_id,
            temperature=temperature,
            streaming=streaming,
        )
        return model

    @classmethod
    def get_llm(
        cls,
        agent_type: str,
        temperature: float = 0,
        streaming: bool = False,
        *,
        llm_config: LLMConfig | None = None,
    ) -> BaseChatModel:
        """按 Agent 类型获取模型实例。

        优先级：llm_config（用户自定义）> 系统默认路由（.env.local）

        Args:
            agent_type: Agent 类型标识，必须在 _AGENT_MODEL_ROUTING 中
            temperature: 温度，对话类可传 0.3~0.7，评分类保持 0
            streaming: 是否流式输出（问答/面试对话用）
            llm_config: 用户自定义模型配置，不传则使用系统默认
        """
        if llm_config is not None:
            return _create_model(
                model_id=llm_config.model_id,
                api_key=llm_config.api_key,
                base_url=llm_config.base_url,
                temperature=temperature,
                streaming=streaming,
            )

        model_key = _AGENT_MODEL_ROUTING.get(agent_type)
        if model_key is None:
            raise ValueError(
                f"未知 agent_type: '{agent_type}'，"
                f"可用类型：{list(_AGENT_MODEL_ROUTING.keys())}"
            )
        return cls._get_or_create_default(model_key, temperature, streaming)

    @classmethod
    def get_structured_llm(
        cls,
        agent_type: str,
        output_schema: Type[BaseModel],
        temperature: float = 0,
        *,
        llm_config: LLMConfig | None = None,
    ) -> Runnable[LanguageModelInput, dict[str, Any] | BaseModel]:
        """获取绑定了结构化输出 Schema 的模型。
        调用它的 ainvoke 后，直接返回 output_schema 类型的对象（不是文本）。"""
        llm = cls.get_llm(agent_type, temperature, llm_config=llm_config)
        return llm.with_structured_output(output_schema, method="function_calling")

    @classmethod
    def clear_cache(cls) -> None:
        """清空系统默认模型的缓存（测试时用）。"""
        cls._get_or_create_default.cache_clear()
        logger.info("llm_factory.cache_cleared")


# ── 模块级便捷函数 ──────────────────────────────────────────

def get_llm(
    agent_type: str,
    temperature: float = 0,
    streaming: bool = False,
    *,
    llm_config: LLMConfig | None = None,
) -> BaseChatModel:
    """LLMFactory.get_llm 的便捷入口。"""
    return LLMFactory.get_llm(agent_type, temperature, streaming, llm_config=llm_config)


def get_structured_llm(
    agent_type: str,
    output_schema: Type[BaseModel],
    temperature: float = 0,
    *,
    llm_config: LLMConfig | None = None,
) -> Runnable[LanguageModelInput, dict[str, Any] | BaseModel]:
    """LLMFactory.get_structured_llm 的便捷入口。"""
    return LLMFactory.get_structured_llm(agent_type, output_schema, temperature, llm_config=llm_config)


def clear_cache() -> None:
    """LLMFactory.clear_cache 的便捷入口。"""
    LLMFactory.clear_cache()


if __name__ == "__main__":
    import asyncio
    from langchain_core.messages import HumanMessage, SystemMessage

    async def main():
        class PersonInfo(BaseModel):
            name: str = Field(description="姓名")
            age: int = Field(description="年龄")

        messages = [
            SystemMessage(content="你是一个负责提取任务信息的助手"),
            HumanMessage(content="我叫小明，今年18岁"),
        ]

        structured = get_structured_llm("qa", PersonInfo)
        response = await structured.ainvoke(messages)
        print(response)

    asyncio.run(main())