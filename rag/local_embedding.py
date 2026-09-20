"""本地 Embedding —— 与 rag/local_reranker.py 同一套懒加载模式。

为什么不用现成的包装：
  1. 不为这一个功能多装依赖（`langchain_huggingface` 没装；`langchain_community` 那版
     已被官方标记 sunset，跑测试时一直在警告）；
  2. 必须自己控制 **query / document 的前缀不对称**：bge-zh 系列要求 query 加
     "为这个句子生成表示以用于检索相关文章："、document 不加。**加错边不会报错**，
     只会让召回悄悄变差（典型的静默劣化）；
  3. `SentenceTransformer.encode` 不是并发安全的，而检索跑在 `asyncio.to_thread`
     的默认线程池（20 worker）里 → 这里统一加锁。

实测（bge-small-zh-v1.5，本机 GTX 1660 Ti）：
  强制 HF_HUB_OFFLINE=1 下加载 0.6s、维度 512、批量 64 条 0.02s、语义排序正常。
"""

from __future__ import annotations

import threading

from langchain_core.embeddings import Embeddings

import config


class LocalEmbeddings(Embeddings):
    """本地 sentence-transformers embedding（懒加载 + 双检锁 + 推理加锁）。"""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        query_prefix: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.model_name = model_name or config.EMBEDDING_MODEL_PATH
        self.device = device or config.EMBEDDING_DEVICE
        self.query_prefix = (
            config.EMBEDDING_QUERY_PREFIX if query_prefix is None else query_prefix
        )
        self.batch_size = batch_size or config.EMBEDDING_BATCH_SIZE
        self.model = None
        self.load_error: str | None = None  # 终态失败只报一次，别每个请求刷一遍
        # ⚠️ 必须用**两把**锁：加载锁与编码锁。
        # 曾经只用一把 threading.Lock → _encode 持锁后再调 load_model()，
        # 而 load_model 又要拿同一把非重入锁 → **自锁死**（表现为调用 embed_query 永久卡住、
        # 没有任何输出与异常）。两把锁的获取顺序固定为 encode → load，不成环。
        self._load_lock = threading.Lock()
        self._encode_lock = threading.Lock()

    # ── 加载 ────────────────────────────────────────────────
    def load_model(self):
        """幂等 + 线程安全；cuda 不可用时自动退 cpu。任何情况下返回模型实例。"""
        if self.model is None:
            with self._load_lock:
                if self.model is None:
                    from sentence_transformers import SentenceTransformer

                    try:
                        self.model = SentenceTransformer(self.model_name, device=self.device)
                    except Exception as e:  # noqa: BLE001
                        if self.device != "cpu":
                            print(
                                f"[LocalEmbeddings] {self.device} 加载失败，回退 CPU: {e!r}"
                            )
                            self.model = SentenceTransformer(self.model_name, device="cpu")
                        else:
                            self.load_error = f"{type(e).__name__}: {e}"
                            raise
                    print(
                        f"[LocalEmbeddings] 模型就绪：{self.model_name} "
                        f"@ {getattr(self.model, 'device', self.device)} "
                        f"维度={self.model.get_sentence_embedding_dimension()}"
                    )
        return self.model

    # ── 编码 ────────────────────────────────────────────────
    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with self._encode_lock:  # encode 不是并发安全的（与加载锁分开，见 __init__ 注释）
            vecs = self.load_model().encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,  # 与集合的 cosine 配套
                show_progress_bar=False,
            )
        return [v.tolist() for v in vecs]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """文档侧：**不加**指令前缀。"""
        return self._encode(list(texts))

    def embed_query(self, text: str) -> list[float]:
        """查询侧：加指令前缀（bge-zh 系列要求；bge-m3 留空即可）。"""
        return self._encode([self.query_prefix + (text or "")])[0]

    @property
    def dimension(self) -> int:
        return int(self.load_model().get_sentence_embedding_dimension())


# ── 工厂：四处调用点必须共用同一个实例 ───────────────────────
# 不共用的话，每处各 new 一个 → 模型被重复加载 N 份（小模型浪费显存，大模型直接 OOM），
# 而且读写两侧若配置不同就会落在不同向量空间 —— 不报错，只是召回静默变差。
_instances: dict[str, Embeddings] = {}
_factory_lock = threading.Lock()


def get_embeddings() -> Embeddings:
    """按 config.EMBEDDING_PROVIDER 返回 embedding 实现（进程内单例）。"""
    provider = config.EMBEDDING_PROVIDER
    with _factory_lock:
        if provider not in _instances:
            if provider == "local":
                _instances[provider] = LocalEmbeddings()
            else:
                import os

                from langchain_community.embeddings import DashScopeEmbeddings

                _instances[provider] = DashScopeEmbeddings(
                    model="text-embedding-v2",
                    dashscope_api_key=os.getenv("QIANWEN_API_KEY"),
                )
        return _instances[provider]
