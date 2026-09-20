"""
意图识别 —— 三级漏斗分类器。

    A. 规则快速命中（零成本、零延迟）→ 命中直接短路
    B. Embedding 相似度分类（一次 embedding 调用）→ 高置信度直接判定
    C. LLM 结构化输出仲裁（仅歧义/低置信度才调用，顺带抽槽位）

对外入口：classify(query) -> IntentResult
"""

import json
import logging
import math
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from intent.examples import (
    ACK_PHRASES,
    GREETING_PREFIXES,
    INTENT_EXAMPLES,
    RULE_PATTERNS,
)
from intent.schemas import IntentName, IntentOutput, IntentReason, IntentResult
from rag.local_embedding import get_embeddings  # 与检索侧共用同一个 embedding 实例

load_dotenv(encoding="utf-8-sig")  # utf-8-sig:兼容带 BOM 的 .env

logger = logging.getLogger(__name__)

# ── 可调阈值（B/C 级）────────────────────────────────────
EMBED_HIGH_THRESHOLD = 0.60  # embedding 最高分超过它且拉开差距 → 直接判定
EMBED_LOW_THRESHOLD = 0.45  # 最高分低于它 → 视为 out_of_scope（省一次 LLM 调用）
EMBED_MARGIN = 0.05  # 与第二名的分数差下限
LLM_CONFIDENCE_THRESHOLD = 0.60  # LLM 仲裁置信度下限，低于它 → ambiguous
# C 级是同步阻塞调用，必须有上限：不设时限时 SDK 默认 600s + 2 次重试，
# 一次网络抖动就会把用户请求挂住几分钟（图跑在线程池里，事件循环不会死，
# 但那个用户一直在等）。
LLM_TIMEOUT_SECONDS = float(os.getenv("INTENT_LLM_TIMEOUT", "12"))
LLM_MAX_RETRIES = int(os.getenv("INTENT_LLM_MAX_RETRIES", "1"))

# 示例向量缓存：改了 intent/examples.py 里的 INTENT_EXAMPLES 后请 +1
# v3：embedding 从云端 DashScope text-embedding-v2（1536 维）换成本地
#     bge-small-zh-v1.5（512 维）—— 旧缓存是 1536 维向量，必须失效，
#     否则要么维度不匹配报错，要么相似度全错（静默劣化）。
CACHE_VERSION = 3
CACHE_FILE = Path(__file__).resolve().parent / "example_embeddings.json"

# ── 规则编译 ─────────────────────────────────────────────
_RULE_RE = [(intent, re.compile(pattern)) for intent, pattern in RULE_PATTERNS]
_GREETING_PREFIX_RE = re.compile(
    r"^(?:" + "|".join(GREETING_PREFIXES) + r")[,，。!！\s]*", re.IGNORECASE
)
_ACK_RE = re.compile(r"^(?:" + "|".join(ACK_PHRASES) + r")+$", re.IGNORECASE)


def _strip_greeting_prefix(query: str) -> str:
    # 替换掉招呼词 你好之类
    return _GREETING_PREFIX_RE.sub("", query, count=1).strip()


def rule_classify(query: str) -> tuple[IntentName | None, str]:
    """A 级：规则匹配。返回 (命中的意图 or None, 用于继续分类的有效 query)。"""
    q = query.strip()

    # 转人工/投诉（最高优先级）
    for intent, regex in _RULE_RE:
        # 拿问题去匹配某个意图的值
        if regex.search(q):
            return intent, q

    # 剥寒暄前缀，看剩余部分
    stripped = _strip_greeting_prefix(q)
    # 从头到尾匹配包括标点符号fullmatch。（这里用来替换结束语，谢谢之类）
    if not stripped or _ACK_RE.fullmatch(stripped):
        # 只剩寒暄/应答 → chitchat
        return IntentName.CHITCHAT, q
    # 有实质内容 → 用剥离后的文本继续走 B/C
    return None, stripped


# ==================== B 级：Embedding 分类 ====================
class EmbeddingClassifier:
    def __init__(self):
        # 与 RAG 检索侧共用同一个 embedding 实现（进程内单例，读写同源）
        self._embeddings = get_embeddings()
        # {IntentName: List[向量]}
        self._vectors: dict[IntentName, list[list[float]]] = {}
        self._load_or_build()

    def _load_or_build(self):
        # 向量化的意图例子 用来和问题匹配取相似度
        examples = INTENT_EXAMPLES
        # 拿到每一个意图的例子
        all_examples: list[str] = [e for exs in examples.values() for e in exs]
        count = len(all_examples)

        # 命中缓存直接复用，避免每次重启都请求 embedding API
        if CACHE_FILE.exists():
            try:
                # 通过旧缓存判断是否需要更新缓存
                data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
                if data.get("version") == CACHE_VERSION and data.get("count") == count:
                    self._vectors = {
                        IntentName(k): [list(v) for v in vecs]
                        for k, vecs in data["vectors"].items()
                    }
                    print(f"意图示例向量：加载缓存（{count} 条）")
                    return

            except Exception as e:
                print(f"意图示例向量缓存读取失败，将重新生成: {e}")

        # print(f"意图示例向量：请求 embedding API（{count} 条示例）...")

        # 计算需要写入的向量
        vectors = self._embeddings.embed_documents(all_examples)

        grouped: dict[str, list[list[float]]] = {}
        idx = 0
        for intent, exs in examples.items():
            grouped[intent.value] = vectors[idx : idx + len(exs)]
            idx += len(exs)
        # 写入文件
        CACHE_FILE.write_text(
            json.dumps(
                {"version": CACHE_VERSION, "count": count, "vectors": grouped},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self._vectors = {IntentName(k): v for k, v in grouped.items()}

        # 返回一个元组

    def classify(self, query: str) -> tuple[IntentName | None, float, dict[str, float]]:
        """
        返回 (意图 or None, 最高分, 各意图最高相似度)。
        None 表示歧义 → 交给 C 级 LLM 仲裁。
        """
        # 将问题向量化
        q_vec = self._embeddings.embed_query(query)
        # 规定这个字典必须是键IntentName值float
        scores: dict[IntentName, float] = {}

        for intent, vecs in self._vectors.items():
            # 计算用户提问和向量化分类模型中的相似度 取最大并保存当前意图
            # 这里是取每一个意图中相似度最高的
            scores[intent] = max(self._cosine(q_vec, v) for v in vecs)

        # print(f"embbding各意图相似度:",scores)

        # 对sorted从小到大排序 这里取了负数-x[1]
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        top_intent, top_score = ranked[0]
        second_score = ranked[1][1]
        # print(f"最高相似度和意图：",top_intent,top_score)

        if (
            top_score >= EMBED_HIGH_THRESHOLD
            and (top_score - second_score) >= EMBED_MARGIN
        ):
            return (
                top_intent,
                top_score,
                {k.value: round(v, 4) for k, v in scores.items()},
            )
        if top_score < EMBED_LOW_THRESHOLD:
            return (
                IntentName.OUT_OF_SCOPE,
                top_score,
                {k.value: round(v, 4) for k, v in scores.items()},
            )
        return None, top_score, {k.value: round(v, 4) for k, v in scores.items()}

    # 计算余弦相似度 越接近1越相似
    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0


# ==================== C 级：LLM 仲裁 ====================
_LLM_SYSTEM_PROMPT = """你是意图识别器，判断用户问题属于哪个意图，并抽取槽位。只能从以下 5 个意图中选择一个：

1. kb_question：用户询问知识库/业务相关内容（产品、售后、规则、流程、政策等），需要查资料回答。
2. chitchat：寒暄、闲聊、问候、感谢、告别、自我介绍类问题。
3. status_query：询问知识库本身的状态（有什么文档、多少资料、是否初始化）。
4. human_handoff：要求转人工、投诉、举报、找真人处理。
5. out_of_scope：与业务无关、知识库也无法回答的问题（写诗、闲聊天气、编程、翻译等）。

同时从问题中抽取槽位（可选）：
- source：用户明确提到的文件名/文档名，如"客服手册"
- keyword：检索关键词，如"退换货"
- time_range：时间范围，如"最近一周"

要求：
- 只输出 JSON，格式：{{"intent": "kb_question", "confidence": 0.9, "slots": {{"source": null, "keyword": null, "time_range": null}}}}
- confidence 是 0~1 的小数，表示你对意图判断的确信程度。
- 拿不准时 confidence 给低分（如 0.5），不要硬猜。
"""


class LLMClassifier:
    def __init__(self):
        self._llm = ChatOpenAI(
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            temperature=0,  # 分类任务要确定性
            timeout=LLM_TIMEOUT_SECONDS,  # 同步节点必须限时
            max_retries=LLM_MAX_RETRIES,
        )
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _LLM_SYSTEM_PROMPT),
                ("human", "用户问题：{query}"),
            ]
        )

    def classify(self, query: str) -> IntentOutput:
        # 方式一：json mode（DeepSeek 支持 response_format json_object）
        try:
            chain = self._prompt | self._llm.with_structured_output(
                IntentOutput, method="json_mode"
            )

            # print(f"llm：", chain.invoke({"query": query}))
            return chain.invoke({"query": query})
        except Exception as e:
            logger.info("LLM json-mode 意图仲裁失败，改走 function calling: %s", e)
        # 方式二：function calling 兜底
        chain = self._prompt | self._llm.with_structured_output(IntentOutput)
        return chain.invoke({"query": query})


# ==================== 三级漏斗 ====================
class IntentClassifier:
    def __init__(self):
        self._embed = EmbeddingClassifier()
        self._llm = LLMClassifier()

    def classify(self, query: str) -> IntentResult:
        query = (query or "").strip()
        # 空问题 → 用显式 reason 标记，而不是靠 confidence == 0 当哨兵
        if not query or not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", query):
            return IntentResult(
                intent=IntentName.AMBIGUOUS,
                confidence=0,
                method="rule",
                reason=IntentReason.EMPTY,
            )

        # rule_classify规则匹配
        intent, effective = rule_classify(query)
        if intent is not None:
            return IntentResult(intent=intent, confidence=1.0, method="rule")

        effective_query = effective or query

        # ── B 级：embedding ──
        # B 级不可用（超时/欠费/key 缺失/模型未就绪）不能让整个漏斗失效：
        # 早期这里异常会一路穿到 intent_router_node，被它的 except 吞掉后
        # intent_result=None → 所有问题都走 agent_flow，RAG 整条路被静默绕过。
        # 现在降级到 C 级 LLM 仲裁，只把原因记下来。
        embed_failed = False
        scores: dict[str, float] = {}
        try:
            intent, score, scores = self._embed.classify(effective_query)
        except Exception as e:
            embed_failed = True
            intent = None
            logger.warning("意图 embedding 分层不可用，降级到 LLM 仲裁: %s", e)

        if intent is not None:
            return IntentResult(
                intent=intent, confidence=score, method="embedding", scores=scores
            )

        # ── C 级：LLM 仲裁（embedding 歧义，或 B 级不可用）──
        try:
            out = self._llm.classify(effective_query)
        except Exception as e:
            logger.warning("LLM 意图仲裁失败: %s", e)
            return IntentResult(
                intent=IntentName.AMBIGUOUS,
                confidence=0,
                method="llm",
                scores=scores,
                reason=(
                    IntentReason.EMBEDDING_ERROR
                    if embed_failed
                    else IntentReason.LLM_ERROR
                ),
            )

        # llm自己的置信度
        if out.confidence >= LLM_CONFIDENCE_THRESHOLD:
            return IntentResult(
                intent=out.intent,
                confidence=out.confidence,
                method="llm",
                slots=out.slots,
                scores=scores,
            )
        # LLM 自己也拿不准 → ambiguous（上层降级走默认 RAG）
        return IntentResult(
            intent=IntentName.AMBIGUOUS,
            confidence=out.confidence,
            method="llm",
            slots=out.slots,
            scores=scores,
            reason=IntentReason.LOW_CONFIDENCE,
        )


# ── 模块级单例 + 对外入口 ────────────────────────────────
_classifier: IntentClassifier | None = None


def get_classifier() -> IntentClassifier:
    global _classifier
    if _classifier is None:
        _classifier = IntentClassifier()
    return _classifier


def classify(query: str) -> IntentResult:
    """意图识别对外入口：query -> IntentResult"""
    return get_classifier().classify(query)
