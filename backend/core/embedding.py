# backend/core/embedding.py
# BGE-M3 本地嵌入模型（进程内单例，dense + sparse 双输出）

import threading
from typing import ClassVar, Optional

from backend.config import get_settings
from backend.core.logger import get_logger
from rich.pretty import pprint


logger = get_logger(__name__)


class BGEMEmbedder:
    """
    BGE-M3 本地嵌入模型单例。

    一次推理同时输出：
      - dense 向量（1024 维浮点数组，用于语义相似度检索）
      - sparse 向量（{token_id: weight} 字典，用于关键词精确检索）

    进程内单例：首次调用 get_instance() 时加载模型（约5-15秒），
    后续调用直接返回同一实例，不重复加载。

    用法：
        embedder = BGEMEmbedder.get_instance()
        dense, sparse = embedder.encode_query("什么是 Spring IOC？")

    注意事项（使用单例时的易错点）：
      1. 始终通过 get_instance() 获取实例，不要直接调 __init__()。
         误调 BGEMEmbedder("path") 虽不会重复加载模型（入口守卫会借
         用已有 _model），但传入不同的 model_path 会被静默忽略。
      2. _instance 和 _lock 是类变量（ClassVar），不是实例变量。
         所有实例共享同一把锁，这是「单例锁」正确工作的前提。
      3. get_instance() 不是线程安全的吗？—— 是的。
         双重检查锁定（double-checked locking）保证了多线程首次
         并发调用时，只有一个线程加载模型，其余线程等待后复用。
      4. 多线程共享 _model 并发推理，在 PyTorch 纯推理路径下是
         安全的（权重只读），GPU kernel 执行会被隐式串行化。
    """

    _instance: Optional["BGEMEmbedder"] = None   # 类级别单例持有
    _lock: ClassVar = threading.Lock()           # 类级别线程锁（ClassVar 确保锁不被实例化复制）
    # ⚠️ _instance 和 _lock 都是类变量，不是实例变量，
    #    所有实例共享同一份，这是「单例锁」的关键——如果每个实例都有一把自己的锁，就锁不住了。

    def __init__(self, model_path: str):
        """
        初始化 BGE-M3 嵌入模型。

        Args:
            model_path: BGE-M3 模型权重的本地目录路径，
                        通常由 settings.bge_m3_model_path 提供。
        """
        if self.__class__._instance is not None:
            # ── 单例已存在时的入口守卫 ──
            # 防止误调 BGEMEmbedder("...") 导致重复加载模型权重。
            # 直接借用已有实例的 _model，不加载第二次。
            # ⚠️ 隐患：如果传入不同的 model_path，会被静默忽略，
            #    实例永远指向首次加载的模型。这是有意为之——单例只有「一份」。
            self._model = self.__class__._instance._model
            return

        from FlagEmbedding import BGEM3FlagModel

        self._model = BGEM3FlagModel(model_name_or_path=model_path)
        logger.info("bge_m3.loaded", model_path=model_path)

    @classmethod
    def get_instance(cls) -> "BGEMEmbedder":
        """获取单例（首次调用时加载模型，后续复用）"""
        # ── 第一次检查（无锁）──
        # 绝大多数调用走此路径：实例已存在，直接返回，零锁开销。
        if cls._instance is None:
            with cls._lock:
                # ── 第二次检查（持锁）──
                # 加锁后再次确认，防止两个线程同时通过第一次检查，
                # 都等在锁上，一个释放后另一个又创建一遍。
                if cls._instance is None:
                    cls._instance = BGEMEmbedder(get_settings().bge_m3_model_path)

        return cls._instance


    def encode(
        self,
        texts: list[str],
        batch_size: int = 12,
    ) -> tuple[list[list[float]], list[dict]]:
        """
        批量编码文本，同时返回 dense 和 sparse 两种向量。

        Args:
            texts:      待编码的文本列表
            batch_size: 单次推理批大小，越大速度越快但显存占用越多；
                        12 是 16GB 显存 / 统一内存下的经验值

        Returns:
            (dense_vecs, sparse_vecs)
              dense_vecs:  list of 1024-dim float 向量，每项对应 texts[i]
              sparse_vecs: list of {token_id: weight} 字典，每项对应 texts[i]
        """
        """
        {
            'dense_vecs': array([[-0.03076,  0.0363 , -0.0574 , ..., -0.0292 ,  0.01178, -0.02486]],
            dtype=float16),
            'lexical_weights': [defaultdict(<class 'int'>, {'87': 0.244, '29065': 0.3445, '33848': 0.2258, '32': 0.1489})],
            'colbert_vecs': None
        }
        """
        output = self._model.encode(
            texts,
            batch_size=batch_size,
            max_length=8192,            # BGE-M3 支持最长 8192 token，覆盖大多数 chunk
            return_dense=True,          # 输出稠密语义向量
            return_sparse=True,         # 输出稀疏关键词向量
            return_colbert_vecs=False,  # ColBERT 多向量表示，本项目不用
        )

        
        dense_vecs = output["dense_vecs"].tolist()   # numpy → Python list

        # 双重类型转换（必须，否则下游序列化会炸）：
        #   int(k)  — FlagEmbedding 把 token ID 存成了 str("123")，
        #             但 Milvus sparse 索引要求 int 键
        #   float(v) — 权重是 numpy.float16/32，pymilvus 和 langgraph msgpack
        #             都不支持 numpy 原生类型，必须转 Python float
        sparse_vecs = [
            {int(k): float(v) for k, v in d.items()}
            for d in output["lexical_weights"]
        ]
        return dense_vecs, sparse_vecs

    def encode_query(self, text: str) -> tuple[list[float], dict]:
        """
        编码单条查询，返回 (dense_vec, sparse_vec)。

        查询时调用此方法（而非 encode），batch_size=1 避免不必要的 padding。

        Returns:
            (dense_vec, sparse_vec)
              dense_vec:  1024-dim float 列表
              sparse_vec: {token_id: weight} 字典
        """
        dense_list, sparse_list = self.encode([text], batch_size=1)
        return dense_list[0], sparse_list[0]


if __name__ == "__main__":
    
    # =================================
    # 测试编码
    # =================================
    bge = BGEMEmbedder.get_instance()
    text = "IOC是什么？"
    dense_vec, sparse_vec = bge.encode_query(text)
    pprint(dense_vec)
    pprint(sparse_vec)
