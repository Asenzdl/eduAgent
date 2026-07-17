import os
import sys

import httpx
# 旧版，我不用
# from mcp.server.fastmcp import FastMCP
# 3.0+新版，我要用
from fastmcp import FastMCP

# 以下代码都有可能为过时内容，建议检查最新文档

mcp = FastMCP(
    name="EduAgent-WebSearch",
    stateless_http=True,
    json_response=True,
)


# ── 搜索后端 ──────────────────────────────────────────────────────

async def _search_tavily(query: str, max_results: int, api_key: str) -> list[dict]:
    """Tavily 搜索（结构化结果，需 API key，https://tavily.com）"""
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key":             api_key,
                "query":               query,
                "max_results":         max_results,
                "include_answer":      False,
                "include_raw_content": False,
            },
        )
        resp.raise_for_status()
    return [
        {
            "title":   r.get("title", ""),
            "url":     r.get("url", ""),
            "snippet": r.get("content", "")[:500],
            "content": r.get("content", ""),
        }
        for r in resp.json().get("results", [])
    ]


# ── MCP 工具 ──────────────────────────────────────────────────────

@mcp.tool()
async def web_search(
    query: str,
    max_results: int = 5,
) -> list[dict]:
    from backend.config import get_settings
    from backend.core.logger import get_logger

    logger = get_logger(__name__)
    settings = get_settings()

    # 优先 Tavily，失败后降级 DuckDuckGo
    if settings.tavily_api_key:
        try:
            results = await _search_tavily(query, max_results, settings.tavily_api_key)
            logger.info("web_search_mcp.tavily_done", hits=len(results))
            return results
        except Exception as e:
            logger.warning("web_search_mcp.tavily_failed", error=str(e))

    else:
        return [{"error": "Tavily API key not set"}]

# ── 独立运行入口 ──────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    from dotenv import load_dotenv

    sys.path.insert(0, str(__file__).split("/backend/")[0])
    load_dotenv(".env.local")

    port = int(os.getenv("WEB_SEARCH_MCP_SERVER_PORT", "8002"))
    print(f"Web Search MCP Server → http://localhost:{port}/mcp")
    uvicorn.run(mcp.streamable_http_app(), host="0.0.0.0", port=port)