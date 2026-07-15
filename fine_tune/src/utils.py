from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import random
import torch

def plot_length_distribution(lengths, title="Token Length Distribution"):
    """
    绘制 token 长度直方图，叠加均值、中位数、P95、P99 以及 μ+3σ 阈值线。

    参数:
        lengths: list[int] 或 np.array，每个样本的 token 序列长度（含特殊 token）。
        title: str，图表标题。
    """
    plt.figure(figsize=(10, 5))
    plt.hist(lengths, bins=80, color='steelblue', edgecolor='white', alpha=0.8)

    # 基本统计量
    mu, sigma = np.mean(lengths), np.std(lengths)
    median = np.median(lengths)
    p95 = np.percentile(lengths, 95)
    p99 = np.percentile(lengths, 99)

    # 绘制阈值竖线
    plt.axvline(mu, color='red', linestyle='--', linewidth=2,
                label=f'Mean = {mu:.2f}')
    plt.axvline(median, color='orange', linestyle='--', linewidth=2,
                label=f'Median = {median}')
    plt.axvline(mu + 3 * sigma, color='gray', linestyle='-.', linewidth=2,
                label=f'μ+3σ = {np.ceil(mu + 3 * sigma)}')
    plt.axvline(p95, color='green', linestyle=':', linewidth=2,
                label=f'P95 = {p95}')
    plt.axvline(p99, color='purple', linestyle=':', linewidth=2,
                label=f'P99 = {p99}')

    plt.xlabel('Token Length')
    plt.ylabel('Sample Count')
    plt.title(title)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def plot_length_cdf(lengths, title="Cumulative Distribution of Token Lengths"):
    """
    绘制 token 长度的累计分布函数 (CDF)，并标注 P90、P95、P99 位置。

    参数:
        lengths: list[int] 或 np.array，每个样本的 token 序列长度（含特殊 token）。
        title: str，图表标题。
    """
    
    sorted_lengths = np.sort(lengths)
    cdf = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)

    plt.figure(figsize=(10, 5))
    plt.plot(sorted_lengths, cdf, drawstyle='steps-post', color='darkblue')
    plt.xlabel('Token Length')
    plt.ylabel('Cumulative Proportion')
    plt.title(title)
    plt.grid(True, alpha=0.3)

    # 标注常用百分位数
    for pct, color in zip([90, 95, 99], ['orange', 'green', 'red']):
        p_val = np.percentile(lengths, pct)
        plt.axhline(pct / 100, color=color, linestyle='--', alpha=0.7)
        plt.axvline(p_val, color=color, linestyle='--', alpha=0.7)
        plt.text(p_val + 1, pct / 100 + 0.02, f'P{pct}={p_val}', fontsize=9)

    plt.tight_layout()
    plt.show()
    
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


