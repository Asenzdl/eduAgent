"""
从 HuggingFace 下载 DeepSeek V4 Flash 官方分词器文件。

用法：
    python backend/deepseek_tokenizer/download_tokenizer.py

下载的文件将放在本目录下：
    tokenizer.json          BPE 词表（约 7.5 MB）
    tokenizer_config.json   分词器配置
    encoding/               prompt 编码实现（encoding_dsv4.py + 测试用例）
"""

import requests
from pathlib import Path

BASE_URL = "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/main"
TARGET_DIR = Path(__file__).parent

FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
]

ENCODING_FILES = [
    "encoding/encoding_dsv4.py",
    "encoding/test_encoding_dsv4.py",
    "encoding/tests/test_input_1.json",
    "encoding/tests/test_output_1.txt",
    "encoding/tests/test_input_2.json",
    "encoding/tests/test_output_2.txt",
    "encoding/tests/test_input_3.json",
    "encoding/tests/test_output_3.txt",
    "encoding/tests/test_input_4.json",
    "encoding/tests/test_output_4.txt",
]


def download_file(relative_path: str) -> bool:
    """下载单个文件，返回是否成功。"""
    url = f"{BASE_URL}/{relative_path}"
    local_path = TARGET_DIR / relative_path

    # 创建父目录
    local_path.parent.mkdir(parents=True, exist_ok=True)

    # 跳过已存在的
    if local_path.exists():
        size = local_path.stat().st_size
        print(f"  ⏭  已存在 {relative_path} ({size:,} bytes)")
        return True

    print(f"  ↓ 下载 {relative_path} ...", end=" ", flush=True)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        local_path.write_bytes(resp.content)
        print(f"完成 ({len(resp.content):,} bytes)")
        return True
    except requests.RequestException as e:
        print(f"失败: {e}")
        return False


def verify_tokenizer():
    """验证下载的 tokenizer 能否正常处理中文。"""
    print()
    print("─" * 40)
    print("验证 tokenizer ...")

    try:
        from tokenizers import Tokenizer

        tk_path = TARGET_DIR / "tokenizer.json"
        if not tk_path.exists():
            print("  ❌ tokenizer.json 不存在，跳过验证")
            return False

        tk = Tokenizer.from_file(str(tk_path))
        tests = [
            ("Hello", 1),
            ("你好", 1),
            ("你好世界", 2),
            ("快速排序算法", 3),
        ]
        all_ok = True
        for text, expected in tests:
            count = len(tk.encode(text).ids)
            status = "✅" if count == expected else "⚠️"
            print(f"  {status} {repr(text):<20} → {count} tokens (期望 {expected})")
            if count != expected:
                all_ok = False

        if all_ok:
            print("  ✅ 全部通过！")
        else:
            print("  ⚠️ 部分结果与预期不符，但能运行")
        return all_ok

    except Exception as e:
        print(f"  ❌ 验证失败: {e}")
        return False


if __name__ == "__main__":
    print("下载 DeepSeek V4 Flash 分词器文件")
    print(f"目标目录: {TARGET_DIR}")
    print(f"来源: {BASE_URL}")
    print()

    success = True

    print("─ 核心文件 ─")
    for rel_path in FILES:
        if not download_file(rel_path):
            success = False

    print()
    print("─ 编码实现（可选） ─")
    for rel_path in ENCODING_FILES:
        if not download_file(rel_path):
            success = False

    print()
    if success:
        print("✅ 全部下载完成")
        verify_tokenizer()
    else:
        print("⚠️ 部分文件下载失败，请检查网络后重试")
