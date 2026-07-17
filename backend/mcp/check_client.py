import asyncio

from backend.mcp.search_client import call_mcp_tool

async def main():
    base = "http://localhost:8001"
    tool_name="search_knowledge_base"
    arguments = {"query": "商品聚合多模态大模型项目主要讲的是什么内容"}
    
    
    result = await call_mcp_tool(base, tool_name, arguments)
    print(f"命中 {len(result)} 条文档")
    print(result)
    
asyncio.run(main())