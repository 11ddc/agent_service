import threading

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import config


class LocalReranker:
    """本地重排序，基于 Cross-Encoder 实现。

    懒加载：构造函数只记录参数不碰模型；首次 rerank() 时才真正加载
    （sentence_transformers 的 import 也推迟到那一刻，服务启动完全不加载 torch），
    之后所有调用复用同一实例，不再重复加载。
    """

    # 本地"F:/my-agent-api/models/ms-marco-MiniLM-L-6-v2"
    # 联网"cross-encoder/ms-marco-MiniLM-L-6-v2"
    #
    # ⚠️ 以前这里写死成 "F:/my-agent-api/Reank_models/bge-reranker-v2-m3"：别人把仓库
    # clone 到别的盘符/目录（或 Linux 上）就必然加载失败，而 rag.reordering() 里的
    # except 会把它**静默降级**成 RRF 顺序 —— 不报错，只是精排这一档（README 里
    # R@1 84.8%）悄悄消失，指标对不上还查不出原因。
    # 现在交给 config 统一解析：本地目录存在就用本地，否则退回 HF 仓库 ID 自动下载。
    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ):
        self.model_name = model_name or config.RERANK_MODEL_PATH
        self.device = device or config.RERANK_DEVICE
        self.model = None
        self._lock = threading.Lock()  # 防止并发首次调用时重复加载模型

    def load_model(self):
        """加载模型（幂等 + 线程安全），任何情况下都返回模型实例。"""
        if self.model is None:  # 快路径：已加载直接返回，不加锁
            with self._lock:
                if self.model is None:  # 双重检查：拿到锁后再确认一次
                    # 重 import 放到函数内部，启动时 import 本模块零开销
                    from sentence_transformers import CrossEncoder

                    print(
                        f"[LocalReranker] 首次调用，加载模型 {self.model_name} 到 {self.device} ..."
                    )
                    try:
                        self.model = CrossEncoder(self.model_name, device=self.device)
                    except Exception as e:  # noqa: BLE001
                        # 与 rag/local_embedding.py 同一套兜底：没独显的机器上
                        # device="cuda" 会直接抛错，而 reordering() 只会降级成
                        # "精排没跑"，从日志里根本看不出是设备选错了。退 CPU 重试。
                        if self.device != "cpu":
                            print(
                                f"[LocalReranker] {self.device} 加载失败，回退 CPU: {e!r}"
                            )
                            self.model = CrossEncoder(self.model_name, device="cpu")
                        else:
                            raise
                    print("[LocalReranker] 模型加载完成，后续调用直接复用")
        return self.model

    def _score(self, query: str, docs: list) -> list[tuple]:
        """给每条候选打分，按分数降序返回 [(doc, score), ...]。"""

        if hasattr(docs[0], "page_content"):
            contents = [doc.page_content for doc in docs]
        else:
            # 如果是字符串的话
            contents = [str(doc) for doc in docs]

        data = [[query, content] for content in contents]

        # 懒加载：首次调用时加载一次，后续直接复用
        model = self.load_model()
        # 返回分数 交叉编码
        scores = model.predict(data)
        print("分数scores：", scores)

        # 将文档和分数配对然后排序
        # zip 两个数组一一绑定 【1，2】 【1，2】zip [(1,1),(2,2)]
        # lambda x:x[1] 表示取元组中的第二个元素 也就是 scores 分数来排序  true从大到小
        doc_scores_data = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        return doc_scores_data

    def rerank_scored(self, query: str, docs: list | None) -> list[tuple]:
        """与 rerank 同一套打分，但把**分数一起返回**：[(doc, score), ...]。

        调用方（rag.rag.reordering）要靠分数做"同源限流 + 相对分数下限"的筛选，
        两个规则都离不开相对分；重排本来就对所有候选算过分了，取出来不额外花时间。
        """
        if not docs:
            return []
        return [(doc, float(score)) for doc, score in self._score(query, docs)]

    def rerank(self, query: str, docs: list | None, top_n: int | None) -> list | None:
        if not docs:
            return docs

        # 提取排序文档
        reranked_docs = [doc for doc, _ in self._score(query, docs)]

        # 截取topn

        if top_n and top_n > 0:
            reranked_docs = reranked_docs[:top_n]

        print("重排序的文档：", reranked_docs)
        return reranked_docs
