"""5 折交叉验证训练脚本。

流程：
  raw_data → holdout test(10%) + CV 数据(90%)
           → StratifiedKFold 分 5 折
           → 依次训 5 个模型（全量微调）
           → 软投票集成 → 评估 test
"""

from pathlib import Path

import torch
from datasets import ClassLabel, Dataset, load_dataset
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
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
    return {
        "accuracy": accuracy_metric.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"],
    }


# ── 数据加载 ──────────────────────────────────────────────

def load_data(cfg, label_list):
    """加载数据、编码标签、留出 holdout test。返回 (cv_data, test)。"""
    ds = load_dataset("json", data_files=cfg.data.raw_datapath, split="train")
    ds = ds.cast_column("label", ClassLabel(names=label_list))

    train_val, test = ds.train_test_split(test_size=0.1, seed=cfg.trainer.seed).values()
    print(f"CV 数据：{len(train_val)}，Holdout test：{len(test)}")
    return train_val, test


def generate_folds(train_val, n_folds, seed):
    """返回 list of (train_indices, val_indices)。"""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(skf.split(train_val, train_val["label"]))


# ── 单折训练 ──────────────────────────────────────────────

def train_fold(
    fold: int,
    train_data: Dataset,
    val_data: Dataset,
    cfg,
    num_classes: int,
) -> AutoModelForSequenceClassification:
    """训练单个 fold，返回训练好的模型（在 CPU 上）。"""

    # 1. Tokenize
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.bert_path)

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=cfg.data.max_length)

    train_data = train_data.map(tokenize_fn, batched=True, remove_columns=["text"])
    val_data = val_data.map(tokenize_fn, batched=True, remove_columns=["text"])

    # 2. 模型（全量微调）
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.bert_path,
        num_labels=num_classes,
        attn_implementation="sdpa",
    )

    # 3. 训练参数
    fold_dir = Path(cfg.output.model_save_path) / "cross_dev" / f"fold_{fold}"
    training_args = TrainingArguments(
        output_dir=str(fold_dir),
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
        save_total_limit=1,
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

    # 4. Trainer
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    # 5. 训练
    print(f"\n{'='*40}\nFold {fold + 1} / 5\n{'='*40}")
    trainer.train()

    # 6. 验证集评估
    val_metrics = trainer.evaluate()
    print(f"Fold {fold + 1} 验证指标：{val_metrics}")

    return trainer.model.cpu()


# ── 集成预测 ──────────────────────────────────────────────

def ensemble_predict(
    models: list[AutoModelForSequenceClassification],
    test_data: Dataset,
    tokenizer,
    max_length: int,
) -> tuple[list[int], list[list[float]]]:
    """对测试集做软投票（平均 logits）。返回 (preds, avg_logits)。"""
    test_data = test_data.map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=max_length),
        batched=True,
        remove_columns=["text"],
    )

    all_logits = []
    for i, model in enumerate(models):
        model.eval()
        trainer = Trainer(model=model, processing_class=tokenizer)
        preds = trainer.predict(test_data)
        all_logits.append(torch.tensor(preds.predictions))
        print(f"  Model {i + 1} 预测完成")

    avg_logits = torch.stack(all_logits).mean(dim=0)
    preds = avg_logits.argmax(dim=-1).tolist()
    return preds, avg_logits.tolist()


# ── 入口 ──────────────────────────────────────────────────

def main():
    cfg = load_config()
    set_seed(cfg.trainer.seed)

    _, _, label_list, num_classes = cfg.load_label_info()
    print(f"类别数：{num_classes}，标签：{label_list}")

    # 1. 加载数据 + holdout test
    cv_data, test = load_data(cfg, label_list)

    # 2. 生成 5 折索引
    folds = generate_folds(cv_data, n_folds=5, seed=cfg.trainer.seed)
    print(f"共 {len(folds)} 折")

    # 3. 依次训练每一折
    models = []
    for fold, (train_idx, val_idx) in enumerate(folds):
        model = train_fold(
            fold=fold,
            train_data=cv_data.select(train_idx),
            val_data=cv_data.select(val_idx),
            cfg=cfg,
            num_classes=num_classes,
        )
        models.append(model)

    # 4. 集成评估
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.bert_path)
    print(f"\n{'='*40}\n集成评估\n{'='*40}")
    preds, _ = ensemble_predict(models, test, tokenizer, cfg.data.max_length)

    y_true = [r["label"] for r in test]
    print(f"\n测试集分类报告（5 折集成）：")
    print(classification_report(y_true, preds, target_names=label_list))


if __name__ == "__main__":
    main()
