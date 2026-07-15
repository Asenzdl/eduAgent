from __future__ import annotations
from backend.config import get_settings
from backend.core.logger import get_logger

logger = get_logger(__name__)

LABEL2ID = {"general": 0, "specialized": 1}
ID2LABEL = {0: "general", 1: "specialized"}

# general 侧置信度阈值设为 0.85（偏高）：
# 专业问题被误判为通用问题的代价更高——LLM 会用自身知识回答，
# 可能与课程内容矛盾。宁可多走一次 RAG，不要漏掉课程相关问题。
GENERAL_CONFIDENCE_THRESHOLD = 0.85


class QueryClassifier:
    """
    QA Query 二分类器：general / specialized（推理专用）。

    用法：
        qc = QueryClassifier()                          # 加载默认微调模型
        qc = QueryClassifier("models/classifier")        # 加载指定路径的微调模型
        label, conf = qc.classify("什么是 Spring IOC？")

    训练请运行：
        python scripts/train_classifier.py --data-path backend/training_data.jsonl
    """

    _instance: "QueryClassifier" | None = None

    def __init__(self, model_path: str):
        """
        Args:
            model_path: 微调模型路径（由 get_instance 从配置读取后传入）。
        """
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        device = 'cpu'
        from transformers import pipeline as hf_pipeline

        self._pipeline = hf_pipeline(
            task="text-classification",
            model=model_path,
            device=device,
            top_k=None,
            truncation=True,
            max_length=128,
        )
        
        logger.info("query_classifier.init", model_path=model_path, device=device)

    @classmethod
    def get_instance(cls) -> QueryClassifier:
        """获取单例（首次调用时懒加载）"""
        if cls._instance is None:
            cls._instance = cls(get_settings().classifier_model_path)
        return cls._instance

    # ── 推理 ─────────────────────────────────────────────────

    def classify(self, text: str) -> tuple[str, float]:
        """
        对 query 做 general / specialized 二分类。

        Returns:
            (label, confidence)
            label:      "general" 或 "specialized"
            confidence: 对应标签的置信度 [0, 1]

        分类规则：
            P(general) >= 0.85 → ("general",      P(general))
            其余               → ("specialized",  1 - P(general))

        """
        # [
        #   {'label': 'specialized', 'score': 0.9900947213172913}, 
        #   {'label': 'general', 'score': 0.009905257262289524}
        # ]
        import time
        start = time.perf_counter()
        raw_outputs: list[dict] = self._pipeline(text)[0]
        end = time.perf_counter()
        print(f"pipeline 推理耗时：{end - start:.4f} 秒")

        # 查找 general 标签的分数（兼容大小写和 LABEL_0 格式）
        # 解析 pipeline 输出，按标签名称索引分数
        scores = {item["label"].lower(): item["score"] for item in raw_outputs}
        general_score = scores.get("general") or scores.get("label_0")

        if general_score is None:
            logger.error(
                "query_classifier.unexpected_labels",
                labels=[x["label"] for x in raw_outputs],
            )
            return "specialized", 0.5

        if general_score >= GENERAL_CONFIDENCE_THRESHOLD:
            label, confidence = "general", general_score
        else:
            label, confidence = "specialized", 1.0 - general_score

        logger.info(
            "query_classifier.result",
            text_preview=text[:50],
            label=label,
            confidence=round(confidence, 4),
        )
        return label, confidence


# ── 模块级单例（供 nodes.py import 调用）───────────────────────

_classifier: QueryClassifier | None = None


def get_query_classifier() -> QueryClassifier:
    """获取 QueryClassifier 单例"""
    global _classifier
    if _classifier is None:
        _classifier = QueryClassifier.get_instance()
    return _classifier


if __name__ == "__main__":
    qc = QueryClassifier.get_instance()
    import time
    start = time.perf_counter()
    label, confidence = qc.classify("混淆矩阵可视化代码在哪里")
    end = time.perf_counter()
    print(f"classify 推理耗时：{end - start:.4f} 秒")
    print((label, confidence))
    """
    2026-07-15 18:15:02 [info     ] query_classifier.init          device=cuda model_path=D:\wjs\program\cc\eduAgent\models\classifier\all-MiniLM-L6-v2-finetuned
    pipeline 推理耗时：0.2453 秒
    2026-07-15 18:15:02 [info     ] query_classifier.result        confidence=0.9901 label=specialized text_preview=混淆矩阵可视化代码在哪里
    classify 推理耗时：0.2456 秒
    ('specialized', 0.9900947427377105)
    
    2026-07-15 18:15:27 [info     ] query_classifier.init          device=cpu model_path=D:\wjs\program\cc\eduAgent\models\classifier\all-MiniLM-L6-v2-finetuned
    pipeline 推理耗时：0.0658 秒
    2026-07-15 18:15:27 [info     ] query_classifier.result        confidence=0.9901 label=specialized text_preview=混淆矩阵可视化代码在哪里
    classify 推理耗时：0.0661 秒
    ('specialized', 0.9900947334244847)
    """