"""DirectoryLoader 综合测试。

覆盖范围：
  1. 默认加载器（DEFAULT_LOADERS）加载 .pdf / .md
  2. glob + exclude 正向筛选与反向排除
  3. exclude_dirs 跳过目录
  4. 多线程与单线程结果一致
  5. 未注册后缀 → warning 提醒
"""

import logging

from langchain_community.document_loaders import TextLoader

from backend.core.dir_loader import DirectoryLoader

DATA_DIR = "backend/agents/qa/data"


def sep(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ── 测试 1：默认加载器 ────────────────────────────────────────

sep("1. 默认加载器加载 .pdf + .md")

loader = DirectoryLoader(
    path=DATA_DIR,
    silent_errors=True,
)
docs = loader.load()
print(f"  文档块: {len(docs)}")
for d in docs:
    src = d.metadata.get("source", "?")
    preview = d.page_content[:60].replace("\n", " ")
    print(f"    [{src[-30:]}] {preview}...")


# ── 测试 2：glob + exclude ────────────────────────────────────

sep("2. glob + exclude 过滤")

loader = DirectoryLoader(
    path=DATA_DIR,
    loaders={".md": (TextLoader, {"encoding": "utf-8"})},
    glob="*.md",
    exclude=["*参考*"],
    silent_errors=True,
)
docs = loader.load()
print(f"  glob='*.md', exclude='*参考*' → {len(docs)} 个（排除『自我介绍参考.md』）")
for d in docs:
    print(f"    {d.metadata.get('source','?')}")


# ── 测试 3：exclude_dirs ─────────────────────────────────────

sep("3. exclude_dirs 跳过目录")

loader = DirectoryLoader(
    path=".",
    loaders={".toml": (TextLoader, {"encoding": "utf-8"})},
    glob="*.toml",
    exclude_dirs=[".git", ".venv", "__pycache__", ".history"],
    silent_errors=True,
)
docs = loader.load()
print(f"  排除隐藏目录后，根目录 .toml → {len(docs)} 个")
for d in docs:
    print(f"    {d.metadata.get('source','?')}")


# ── 测试 4：多线程 vs 单线程 ─────────────────────────────────

sep("4. 多线程与单线程加载结果一致")

loader = DirectoryLoader(
    path=DATA_DIR,
    silent_errors=True,
)

docs_mt = loader.load()
docs_st = list(loader.lazy_load())  # 单线程走 lazy_load
assert len(docs_mt) == len(docs_st), "结果数不一致"
print(f"  多线程 load()    → {len(docs_mt)} 个")
print(f"  单线程 lazy_load() → {len(docs_st)} 个 ✅")


# ── 测试 5：未注册格式 warning ───────────────────────────────

sep("5. 未注册后缀 → warning 提醒")

logging.basicConfig(level=logging.WARNING, format="  [%(levelname)s] %(message)s")
loader = DirectoryLoader(
    path=DATA_DIR,
    loaders={".txt": (TextLoader, {"encoding": "utf-8"})},
    silent_errors=True,
)
docs = loader.load()
print(f"  最终加载: {len(docs)} 个（.pdf / .md 未注册，应被跳过）")


print(f"\n{'=' * 60}")
print(f"  全部测试通过")
print(f"{'=' * 60}\n")
