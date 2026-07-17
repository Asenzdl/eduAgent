"""HuggingFace Trainer 版训练脚本。

直接使用 HF API，不封装自定义函数。
"""
from pathlib import Path

from collections import Counter

import torch
import torch.nn as nn
from datasets import ClassLabel, Dataset, load_dataset
from sklearn.metrics import classification_report
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
import evaluate

# ── 路径设置 ──────────────────────────────────────────────


from fine_tune.src.config import load_config


# ── 评估指标 ──────────────────────────────────────────────

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)
    f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")
    return {"accuracy": acc["accuracy"], "f1": f1["f1"]}


def compute_class_weights(dataset: Dataset, num_classes: int, label_list: list[str]) -> torch.Tensor:
    """根据训练集类别分布计算逆频率权重（均值归一化到 1）。"""
    label_counts = Counter(dataset["label"])
    class_counts = [label_counts[i] for i in range(num_classes)]
    inv_freq = [1.0 / (c ** 0.5) for c in class_counts]
    mean_freq = sum(inv_freq) / len(inv_freq)
    weights = torch.tensor([f / mean_freq for f in inv_freq], dtype=torch.float32)
    print(f"类别分布：{dict(zip(label_list, class_counts))}")
    print(f"类别权重：{dict(zip(label_list, [f'{w:.3f}' for w in weights.tolist()]))}")
    return weights


# ── 数据加载 ─────────────────────────────────────────────

def load_and_split_data(
    raw_datapath: str,
    label_list: list[str],
    seed: int,
    test_ratio: float = 0.1,
) -> tuple[Dataset, Dataset, Dataset]:
    """从单 JSONL 加载数据，标签字符串→int，拆分为 train/val/test。"""
    ds = load_dataset("json", data_files=raw_datapath, split="train")
    ds = ds.cast_column("label", ClassLabel(names=label_list))

    train_val = ds.train_test_split(test_size=test_ratio, seed=seed)
    test = train_val["test"]
    train_val = train_val["train"].train_test_split(test_size=test_ratio, seed=seed)
    train_data = train_val["train"]
    val_data = train_val["test"]
    del ds, train_val

    return train_data, val_data, test

# ── 工具函数 ──────────────────────────────────────────────

def check_resume(output_dir: str | Path) -> bool:
    """检查输出目录是否有 checkpoint，用于断点续训。

    用法:
        resume = check_resume(output_dir)
        trainer.train(resume_from_checkpoint=resume)
    """
    save_dir = Path(output_dir)
    resume = save_dir.exists() and any(save_dir.glob("checkpoint-*"))
    print(f"\n开始训练...（{'从 checkpoint 恢复' if resume else '从头训练'}）")
    return resume



def print_classification_report(trainer: Trainer, test_data: Dataset, label_list: list[str]):
    """在测试集上预测并打印分类报告。"""
    preds = trainer.predict(test_data)
    y_pred = preds.predictions.argmax(axis=-1)
    y_true = preds.label_ids
    print("\n分类报告：")
    print(classification_report(y_true, y_pred, target_names=label_list))


# ── 训练层 ──────────────────────────────────────────────

def train():
    # 1. 配置
    cfg = load_config()
    label2id, id2label, label_list, num_classes = cfg.load_label_info()
    print(f"类别数：{num_classes}，标签：{label_list}")

    # 2. 加载数据 + 标签编码 + 拆分
    train_data, val_data, test_data = load_and_split_data(
        cfg.data.raw_datapath, label_list, cfg.trainer.seed
    )
    print(f"训练集：{len(train_data)}，验证集：{len(val_data)}，测试集：{len(test_data)}")
    
    # 3. Tokenize
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.bert_path)
    
    def tokenize_fn(batch):
        """batch: Dataset({
            features: ['text', 'label', 'input_ids', 'attention_mask', 'token_type_ids']
            num_rows: len(batch)
        })
        """
        return tokenizer(batch["text"], truncation=True, max_length=cfg.data.max_length)

    train_data = train_data.map(tokenize_fn, batched=True, remove_columns=["text"])
    val_data = val_data.map(tokenize_fn, batched=True, remove_columns=["text"])
    test_data = test_data.map(tokenize_fn, batched=True, remove_columns=["text"])
    
    # 4. 模型（全量微调）
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.bert_path,
        num_labels=num_classes,
        label2id=label2id,
        id2label=id2label,
        attn_implementation="sdpa",
    )
    
    """冻结 BERT 主体（仅训练分类头）(可选)
    可训练参数：22,713,986 / 22,713,986（100.0000%）
    可训练参数：770 / 22,713,986（0.0034%）
    classifier.weight 768
    classifier.bias 2
    """
    # trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # total = sum(p.numel() for p in model.parameters())
    # print(f"可训练参数：{trainable:,} / {total:,}（{trainable/total:.4%}）")

    # for param in model.base_model.parameters():
    #     param.requires_grad = False

    # trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # total = sum(p.numel() for p in model.parameters())
    # print(f"可训练参数：{trainable:,} / {total:,}（{trainable/total:.4%}）")

    # for name, p in model.named_parameters():
    #     if p.requires_grad:
    #         print(name, p.numel())

    # 5. 训练参数
    output_dir = Path(cfg.output.model_save_path) / "hf"
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=cfg.trainer.batch_size,
        per_device_eval_batch_size=cfg.trainer.batch_size,
        num_train_epochs=cfg.trainer.num_epochs,
        learning_rate=cfg.trainer.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.trainer.warmup_ratio,
        # warmup_steps=500,
        weight_decay=cfg.trainer.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=3,
        logging_steps=5,
        logging_first_step=True,
        dataloader_drop_last=True,
        bf16=True,
        tf32=True,
        seed=cfg.trainer.seed,
        data_seed=42,
        dataloader_pin_memory=True,
        report_to="none",
    )

    # 6. 类别权重
    class_weights = compute_class_weights(train_data, num_classes, label_list)

    def weighted_loss(outputs, labels, num_items_in_batch):
        """自定义加权损失。用 sum + 手动归一化代替默认 mean，
        确保多卡和梯度累积时 loss 梯度方向正确。"""
        logits = outputs["logits"]
        loss = nn.CrossEntropyLoss(
            weight=class_weights.to(logits.device),
            reduction="sum",
        )(logits, labels)
        return loss / num_items_in_batch

    # 7. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        compute_loss_func=weighted_loss,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    #7. 训练（有 checkpoint 自动恢复，没有则从头训练）
    trainer.train(resume_from_checkpoint=check_resume(output_dir))

    # 8. 测试集评估
    print("\n测试集评估：")
    test_results = trainer.evaluate(test_data)
    print(test_results)

    # 9. 分类报告
    print_classification_report(trainer, test_data, label_list)

    # 10. 保存
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n模型已保存到：{output_dir}")


if __name__ == "__main__":
    train()
