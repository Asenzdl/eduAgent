"""分析原始数据集的标签分布，辅助判断是否需要类别权重。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from collections import Counter

from datasets import load_dataset

from fine_tune.src.config import load_config


def main():
    cfg = load_config()
    _, id2label, label_list, _ = cfg.load_label_info()

    # 1. 加载数据（不 cast label，保留字符串方便展示）
    ds = load_dataset("json", data_files=cfg.data.raw_datapath, split="train")
    print(f"样本总数：{len(ds)}")
    print()

    # 2. 标签分布
    labels_raw = ds["label"]
    counter = Counter(labels_raw)
    total = len(labels_raw)

    print(f"{'标签':<16} {'数量':>6} {'占比':>8}")
    print("-" * 32)
    for label in sorted(counter.keys(), key=lambda x: id2label.get(x, x)):
        count = counter[label]
        print(f"{label:<16} {count:>6} {(count/total)*100:>7.1f}%")

    print()
    if len(counter) >= 2:
        majority = max(counter.values())
        minority = min(counter.values())
        ratio = majority / minority
        print(f"不平衡比（多数/少数）：{ratio:.2f}")
        if ratio > 2:
            print("⚠️  不平衡比 > 2，建议使用类别权重（当前已启用）")
        else:
            print("✅ 分布较均衡，类别权重影响不大")

    # 3. 文本长度分布（按标签）
    print()
    print(f"{'标签':<16} {'样本数':>6} {'平均长度':>10} {'最短':>6} {'最长':>6}")
    print("-" * 44)
    for label in sorted(counter.keys(), key=lambda x: id2label.get(x, x)):
        texts = [r["text"] for r in ds if r["label"] == label]
        lengths = [len(t) for t in texts]
        avg = sum(lengths) / len(lengths) if lengths else 0
        print(f"{label:<16} {len(texts):>6} {avg:>8.1f}字 {min(lengths):>6} {max(lengths):>6}")

    # 4. 空文本检查
    empty = sum(1 for r in ds if not r["text"].strip())
    if empty:
        print(f"\n⚠️  有空文本：{empty} 条")
    else:
        print("\n✅ 无空文本")


if __name__ == "__main__":
    main()
