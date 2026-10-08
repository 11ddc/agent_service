"""重排序：智谱 glm-4.5-air（替代原先的本地 Cross-Encoder，旧代码见 rag/local_reranker.py）。

为什么换成 API 调用：
  · 本地 bge-reranker-v2-m3 权重 2.2GB，首次加载几十秒、常驻显存；
  · 智谱的 ZHI_PU_API_KEY 已经为工具调用模型配好了，直接复用，零额外配置。

打分口径：让 glm-4.5-air 按 0~10 的**绝对标准**给每个候选块打相关性分，
再按分数降序返回。对外接口与 LocalReranker 完全一致
（rerank / rerank_scored / load_model），所以 rag.reordering() 的调用处一行都不用改。
"""

import json
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

# 本模块只依赖环境变量（不像 config.py 那样在 import 期算常量），但 key 必须在
# **构造实例之前**读到：单独跑脚本 / 换入口时别指望调用方先 load_dotenv。
# 默认不覆盖已有环境变量，所以 tests/conftest.py 注入的占位 key 仍然优先。
load_dotenv(encoding="utf-8-sig")  # utf-8-sig:兼容带 BOM 的 .env

logger = logging.getLogger(__name__)

# 全部可以用环境变量覆盖，默认值对齐 .env 里已经有的那套（key 名、base_url 与 tool_llm.py 一致）
GLM_RERANK_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"


def _content(doc) -> str:
    """候选块取正文：Document 用 page_content，别的（字符串等）就 str 一下。"""
    text = getattr(doc, "page_content", None)
    return str(text) if text is not None else str(doc)


_SYSTEM_PROMPT = (
    "你是检索相关性打分器。给你一个用户问题和若干候选片段，"
    "请为每个片段打一个 0~10 的整数分："
    "10=直接回答该问题；7~9=高度相关；4~6=部分相关；1~3=几乎无关；0=完全无关。"
    "只输出 JSON，不要任何解释、不要 markdown 代码块，格式："
    '{"scores":[{"index":1,"score":8},{"index":2,"score":0}]}'
)


def _user_prompt(query: str, texts: list[str], doc_chars: int) -> str:
    blocks = []
    for i, text in enumerate(texts, 1):
        # 长块只截前 doc_chars 字：打分看的是"这块讲不讲这个问题"，不需要全文
        snippet = re.sub(r"\s+", " ", str(text)).strip()[:doc_chars]
        blocks.append(f"[{i}] {snippet}")
    return (
        f"【用户问题】\n{query}\n\n【候选片段】\n"
        + "\n".join(blocks)
        + "\n\n请给每个片段打分，index 必须与上面的编号一一对应，只输出 JSON。"
    )


_JSON_RE = re.compile(r"\{.*\}", re.S)
# JSON 解析失败时的退路：直接扫 "index: 1, score: 8" 这种片段
_PAIR_RE = re.compile(
    r'"?index"?\s*[:：]\s*(\d+)[^}\n]*?"?score"?\s*[:：]\s*(-?\d+(?:\.\d+)?)'
)


def _clamp(value: float) -> float:
    """打分统一归到 0~10：模型偶尔会给百分制（85）或 0~1 的小数，这里只做上界缩放。"""
    if value > 10:
        value = value / 10.0
    return max(0.0, min(10.0, value))


def _parse_scores(text: str, n: int) -> list[float | None]:
    """从模型输出里抠出 n 个分数，抠不到的位置是 None（调用方决定怎么兜底）。"""
    scores: list[float | None] = [None] * n
    raw = (text or "").strip()

    items = None
    match = _JSON_RE.search(raw)
    if match:
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            for key in ("scores", "results", "data", "items"):
                if isinstance(data.get(key), list):
                    items = data[key]
                    break
        elif isinstance(data, list):
            items = data

    if items:
        for i, item in enumerate(items):
            if isinstance(item, dict):
                index = item.get("index", item.get("id", i + 1))
                value = item.get("score", item.get("relevance"))
            else:
                index, value = i + 1, item
            try:
                index, value = int(index), float(value)
            except (TypeError, ValueError):
                continue
            if 1 <= index <= n:
                scores[index - 1] = _clamp(value)

    if all(s is None for s in scores):
        for index_s, value_s in _PAIR_RE.findall(raw):
            index = int(index_s)
            if 1 <= index <= n:
                scores[index - 1] = _clamp(float(value_s))

    return scores


class GLMReranker:
    """用智谱 glm-4.5-air 做重排序（懒加载：构造时不建客户端、不发请求）。"""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        batch_size: int | None = None,
        doc_chars: int | None = None,
        max_workers: int | None = None,
    ):
        self.model_name = model or os.getenv("RERANK_MODEL_NAME") or "glm-4.5-air"
        self.api_key = api_key or os.getenv("ZHI_PU_API_KEY")
        self.base_url = base_url or os.getenv("ZHIPU_BASE_URL") or GLM_RERANK_BASE_URL
        self.timeout = float(timeout or os.getenv("RERANK_TIMEOUT", "30"))
        self.max_retries = int(max_retries or os.getenv("RERANK_MAX_RETRIES", "1"))
        # 一次请求评几块：块太多时模型容易漏编号，分批还能控制单次 token
        self.batch_size = int(batch_size or os.getenv("RERANK_BATCH_SIZE", "10"))
        self.doc_chars = int(doc_chars or os.getenv("RERANK_DOC_CHARS", "600"))
        # 批次并发数。实测 glm-4.5-air 单次约 5s：30 个候选分 3 批，串行要 14s，
        # 并发后约等于一批的耗时（各批独立打分，0~10 是绝对口径，跨批可比）。
        self.max_workers = int(max_workers or os.getenv("RERANK_MAX_WORKERS", "4"))

        self._client = None
        self._lock = threading.Lock()  # 防止并发首次调用时重复建客户端
        # 兼容旧调用口径 rag.reranker.model.predict([[query, doc], ...])，见 _PairScorer
        self.model = _PairScorer(self)

    # ── 客户端 ───────────────────────────────────────────────
    def _get_client(self):
        if self._client is None:
            with self._lock:
                if self._client is None:
                    from openai import OpenAI  # 放到函数里：本模块 import 时零开销

                    logger.info(
                        "[GLMReranker] 初始化智谱客户端 model=%s base_url=%s",
                        self.model_name,
                        self.base_url,
                    )
                    self._client = OpenAI(
                        api_key=self.api_key,
                        base_url=self.base_url,
                        timeout=self.timeout,
                        max_retries=self.max_retries,
                    )
        return self._client

    def load_model(self):
        """把"能不能用"提前暴露出来。

        eval/retrieval_recall.py 靠它判断精排档位是否可信（拿不到就说明重排会降级）。
        """
        if not self.api_key:
            raise RuntimeError("缺少 ZHI_PU_API_KEY：glm-4.5-air 重排序无法调用")
        self._get_client()
        return self

    # ── 打分 ─────────────────────────────────────────────────
    def score_texts(self, query: str, texts: list) -> list[float]:
        """给一组纯文本打相关性分，返回与入参等长的分数列表。"""
        if not texts:
            return []
        client = self._get_client()
        batches = [
            texts[start : start + self.batch_size]
            for start in range(0, len(texts), self.batch_size)
        ]
        if len(batches) == 1:
            return self._score_batch(client, query, batches[0])

        # 多批并发（不是并行省 token，而是别让调用方白等）：pool.map 保持入参顺序，
        # 任何一批失败都会在这里抛出 → 由 rag.reordering() 走可见的降级分支
        with ThreadPoolExecutor(max_workers=min(len(batches), self.max_workers)) as pool:
            batch_results = list(pool.map(lambda b: self._score_batch(client, query, b), batches))
        return [score for batch_scores in batch_results for score in batch_scores]

    def _score_batch(self, client, query: str, texts: list) -> list[float]:
        resp = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _user_prompt(query, texts, self.doc_chars)},
            ],
            temperature=0.0,
        )
        text = (resp.choices[0].message.content or "").strip()
        logger.debug("[GLMReranker] 打分原始输出：%s", text)

        parsed = _parse_scores(text, len(texts))
        if all(s is None for s in parsed):
            # 全都没解析出来：抛出去让 rag.reordering() 走可见的降级分支，
            # 而不是在这里悄悄返回一堆 0 分（那就变成"重排静默失效"）
            raise RuntimeError(f"glm-4.5-air 打分结果无法解析: {text[:200]!r}")
        for i, score in enumerate(parsed):
            if score is None:
                logger.warning("[GLMReranker] 第 %d 块没拿到分数，按 0 分处理", i + 1)
        return [0.0 if s is None else s for s in parsed]

    def _score(self, query: str, docs: list) -> list[tuple]:
        """给每条候选打分，按分数降序返回 [(doc, score), ...]。"""
        scores = self.score_texts(query, [_content(d) for d in docs])
        return sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    # ── 对外的两个方法（与 LocalReranker 同签名）──────────────
    def rerank_scored(self, query: str, docs: list | None) -> list[tuple]:
        """与 rerank 同一套打分，但把**分数一起返回**：[(doc, score), ...]。

        rag.rag.reordering 要靠分数做"同源限流 + 相对分数下限"的筛选。
        """
        if not docs:
            return []
        return [(doc, float(score)) for doc, score in self._score(query, docs)]

    def rerank(self, query: str, docs: list | None, top_n: int | None = None) -> list | None:
        if not docs:
            return docs
        reranked = [doc for doc, _ in self._score(query, docs)]
        if top_n and top_n > 0:
            reranked = reranked[:top_n]
        logger.info("[GLMReranker] 重排序 %d 块 → 送出 %d 块", len(docs), len(reranked))
        return reranked


class _PairScorer:
    """兼容旧调用口径：rag.reranker.model.predict([[query, doc], ...]) -> [float, ...]。

    eval/retrieval_recall.py 用它算负样本的最高精排分，判断"库里没有答案"能否被识别。
    """

    def __init__(self, owner: GLMReranker):
        self._owner = owner

    def predict(self, pairs: list) -> list[float]:
        return [self._owner.score_texts(q, [_content(d)])[0] for q, d in pairs]
