from pathlib import Path
from pydantic import BaseModel, field_validator
import yaml
import json

# ── 路径常量 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_YAML = PROJECT_ROOT / "configs" / "template.yaml"


# ── 路径解析工具 ──────────────────────────────────────────

def _resolve_path(v: str) -> str:
    """解析路径：绝对路径原样返回，本地存在的相对路径拼接项目根目录，
    否则视为 HuggingFace repo ID 原样返回。"""
    if not v:
        return ""
    if Path(v).is_absolute():
        return v
    # 以 "." 开头或本地存在的路径 → 拼接项目根目录
    if v.startswith(".") or (PROJECT_ROOT / v).exists():
        return str(PROJECT_ROOT / v)
    # HuggingFace repo ID（如 "bert-base-chinese"）保持原样
    return v


def _resolve_local_path(v: str) -> str:
    """本地路径专用：始终拼接项目根目录"""
    if not v:
        return ""
    return str(PROJECT_ROOT / v)


# ── 子配置 ────────────────────────────────────────────────

class TrainerConfig(BaseModel):
    num_epochs: int
    batch_size: int
    lr: float
    warmup_ratio: float
    weight_decay: float
    seed: int


class DataConfig(BaseModel):
    raw_datapath: str
    train_datapath: str
    dev_datapath: str
    test_datapath: str
    label_map_path: str
    max_length: int

    @field_validator("raw_datapath", "train_datapath", "dev_datapath", "test_datapath", "label_map_path")
    @classmethod
    def resolve_path(cls, v):
        return _resolve_local_path(v)


class ModelConfig(BaseModel):
    bert_path: str
    hidden_size: int
    model_type: str = "bert"

    @field_validator("bert_path")
    @classmethod
    def resolve_path(cls, v):
        return _resolve_path(v)


class OutputConfig(BaseModel):
    model_save_path: str
    quant_model_path: str

    @field_validator("model_save_path", "quant_model_path")
    @classmethod
    def resolve_path(cls, v):
        return _resolve_local_path(v)


# ── 主配置 ────────────────────────────────────────────────

class Config(BaseModel):
    trainer: TrainerConfig
    data: DataConfig
    model: ModelConfig
    output: OutputConfig

    def load_label_info(self) -> tuple[dict[str, int], dict[int, str], list[str], int]:
        """加载标签映射信息。返回 (label2id, id2label, label_list, num_classes)。

        label2id: {"general": 0, "specialized": 1}   → AutoModel 的 label2id
        id2label: {0: "general", 1: "specialized"}   → AutoModel 的 id2label
        label_list: ["general", "specialized"]        → ClassLabel 的 names
        num_classes: 2                                → 类别数
        """
        with open(self.data.label_map_path, "r", encoding="utf-8") as f:
            label2id: dict[str, int] = json.load(f)
        id2label = {v: k for k, v in label2id.items()}
        label_list = sorted(label2id, key=label2id.get)
        return label2id, id2label, label_list, len(label_list)


# ── 加载入口 ──────────────────────────────────────────────

def load_config(yaml_path: str | Path = DEFAULT_YAML) -> Config:
    with open(yaml_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(**raw)


if __name__ == "__main__":
    cfg = load_config()
    print(f"Epochs: {cfg.trainer.num_epochs}")
    print(f"Train path: {cfg.data.train_datapath}")
    print(f"lr: {cfg.trainer.lr} ({type(cfg.trainer.lr).__name__})")
    print(f"Model path: {cfg.model.bert_path}")
    print(f"Max length: {cfg.data.max_length}")
    print(f"model save path: {cfg.output.model_save_path}")

    if cfg.data.label_map_path:
        print(f"Label map path: {cfg.data.label_map_path}")
        _, _, label_list, num_classes = cfg.load_label_info()
        print(f"Label list: {label_list}")
        print(f"num_classes: {num_classes}")
