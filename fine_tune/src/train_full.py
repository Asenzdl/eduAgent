"""全量数据训练脚本（生产用）。

CV 确认方案可行后，在此脚本中用全部数据训最终模型。
- 仅保留 10% 验证集用于 early stopping，不再留测试集
- 全量微调
"""

from pathlib import Path

from datasets import ClassLabel, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

import evaluate

from fine_tune.src.config import load_config
from fine_tune.src.utils import set_seed


# ── 评估指标 ──────────────────────────────────────────────

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)
    f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")
    return {"accuracy": acc["accuracy"], "f1": f1["f1"]}

# ── 训练 ─────────────────────────────────────────────────

def main():
    cfg = load_config()
    set_seed(cfg.trainer.seed)

    label2id, id2label, label_list, num_classes = cfg.load_label_info()
    print(f"类别数：{num_classes}，标签：{label_list}")

    # 1. 加载全部数据，仅留 10% 验证集
    ds = load_dataset("json", data_files=cfg.data.raw_datapath, split="train")
    ds = ds.cast_column("label", ClassLabel(names=label_list))

    train_val = ds.train_test_split(test_size=0.1, seed=cfg.trainer.seed)
    train_data = train_val["train"]
    val_data = train_val["test"]
    del ds, train_val

    print(f"训练集：{len(train_data)}，验证集：{len(val_data)}")

    # 2. Tokenize
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.bert_path)

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=cfg.data.max_length)

    train_data = train_data.map(tokenize_fn, batched=True, remove_columns=["text"])
    val_data = val_data.map(tokenize_fn, batched=True, remove_columns=["text"])

    # 3. 模型（全量微调）
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.bert_path,
        num_labels=num_classes,
        label2id=label2id,
        id2label=id2label,
        attn_implementation="sdpa",
    )

    # 4. 训练参数
    output_dir = Path(cfg.output.model_save_path) / "full"
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=cfg.trainer.batch_size,
        per_device_eval_batch_size=cfg.trainer.batch_size,
        num_train_epochs=cfg.trainer.num_epochs,
        learning_rate=cfg.trainer.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.trainer.warmup_ratio,
        weight_decay=cfg.trainer.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        dataloader_drop_last=True,
        logging_steps=5,
        logging_first_step=True,
        bf16=True,
        tf32=True,
        seed=cfg.trainer.seed,
        data_seed=42,
        dataloader_pin_memory=True,
        report_to="none",
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # 6. 训练
    print("\n开始全量训练...")
    trainer.train()

    # 7. 保存最终模型
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"模型已保存到：{output_dir}")


if __name__ == "__main__":
    main()
