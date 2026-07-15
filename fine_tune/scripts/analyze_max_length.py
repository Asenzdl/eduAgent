"""分析原始数据的 token 长度分布，辅助确定 max_length。"""

import sys
from pathlib import Path

# 项目根目录（eduAgent/），确保可导入 fine_tune 内部的模块
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from datasets import load_dataset
from transformers import AutoTokenizer

from fine_tune.src.utils import plot_length_distribution, plot_length_cdf
from fine_tune.src.config import load_config


def main():
    cfg = load_config()

    # 1. 加载原始数据
    ds = load_dataset("json", data_files=cfg.data.raw_datapath, split="train")
    print(f"样本总数：{len(ds)}")

    # 2. tokenize，统计长度
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.bert_path)

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=False)  # 不截断，看真实长度

    ds = ds.map(tokenize_fn, batched=True)
    lengths = [len(tokens) for tokens in ds["input_ids"]]

    # 3. 基本统计
    import numpy as np

    lengths_arr = np.array(lengths)
    print(f"\n--- 长度统计 ---")
    print(f"  Min:     {lengths_arr.min()}")
    print(f"  Max:     {lengths_arr.max()}")
    print(f"  Mean:    {lengths_arr.mean():.1f}")
    print(f"  Median:  {np.median(lengths_arr):.1f}")
    print(f"  P95:     {np.percentile(lengths_arr, 95):.1f}")
    print(f"  P99:     {np.percentile(lengths_arr, 99):.1f}")
    print(f"  μ+3σ:    {np.ceil(lengths_arr.mean() + 3 * lengths_arr.std()):.0f}")

    # 4. 建议
    print(f"\n--- max_length 建议 ---")
    p99 = np.percentile(lengths_arr, 99)
    mu_3sigma = np.ceil(lengths_arr.mean() + 3 * lengths_arr.std())
    suggested = int(min(p99, mu_3sigma) + 10)  # 留 10 个 token 余量
    print(f"  建议值 (P99+10): {suggested}")
    print(f"  建议值 (μ+3σ+10): {int(mu_3sigma + 10)}")
    print(f"  当前配置: {cfg.data.max_length}")

    # 5. 可视化
    plot_length_distribution(lengths)
    plot_length_cdf(lengths)


if __name__ == "__main__":
    main()
