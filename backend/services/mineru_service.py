"""
MinerU PDF 解析服务封装（异步版）

调用本地 MinerU FastAPI 服务 (http://localhost:8000) 解析 PDF 文档。

用法:
    result = await MinerUService().parse_pdf("path/to/file.pdf")
    print(result.markdown)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx


@dataclass
class ParseResult:
    """MinerU 解析结果"""
    task_id: str
    status: str
    file_name: str
    markdown: str = ""
    images: dict[str, str] = field(default_factory=dict)
    """提取的图片，{文件名: base64_data_uri}"""
    error: Optional[str] = None
    raw: dict = field(default_factory=dict)


class MinerUService:
    """MinerU PDF 解析服务封装（异步）"""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> MinerUService:
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    async def parse_pdf(
        self,
        pdf_path: str | Path,
        backend: str = "pipeline",
        lang_list: Optional[list[str]] = None,
        formula_enable: bool = True,
        table_enable: bool = True,
        return_images: bool = True,
        start_page: int = 0,
        end_page: int = 99999,
        server_url: Optional[str] = None,
    ) -> ParseResult:
        """
        异步解析 PDF 文件，返回解析结果。

        Args:
            pdf_path: PDF 文件路径
            backend: 解析后端 (pipeline / hybrid-engine / vlm-engine)
            lang_list: 文档语言列表，默认 ["ch"]
            formula_enable: 是否解析公式
            table_enable: 是否解析表格
            return_images: 是否返回提取的图片（base64 格式）
            start_page: 起始页码（从 0 开始）
            end_page: 结束页码
            server_url: (仅 hybrid/vlm-http-client) OpenAI 兼容服务地址
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

        lang_list = lang_list or ["ch"]

        data: dict = {
            "return_md": "true",
            "return_images": str(return_images).lower(),
            "backend": backend,
            "lang_list": lang_list,
            "formula_enable": str(formula_enable).lower(),
            "table_enable": str(table_enable).lower(),
            "start_page_id": str(start_page),
            "end_page_id": str(end_page),
        }
        if server_url:
            data["server_url"] = server_url

        files = {"files": (pdf_path.name, pdf_path.read_bytes(), "application/pdf")}

        resp = await self.client.post("/file_parse", data=data, files=files)
        resp.raise_for_status()

        result = resp.json()
        status = result.get("status", "unknown")
        file_name = result.get("file_names", [pdf_path.stem])[0]

        markdown = ""
        images: dict[str, str] = {}
        error = result.get("error")
        results_dict = result.get("results", {})
        if file_name in results_dict:
            file_result = results_dict[file_name]
            markdown = file_result.get("md_content", "")
            images = file_result.get("images", {})
        elif results_dict:
            first_key = next(iter(results_dict))
            file_result = results_dict[first_key]
            markdown = file_result.get("md_content", "")
            images = file_result.get("images", {})

        return ParseResult(
            task_id=result.get("task_id", ""),
            status=status,
            file_name=file_name,
            markdown=markdown,
            images=images,
            error=error,
            raw=result,
        )

    async def check_health(self) -> bool:
        try:
            resp = await self.client.get("/health")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False
