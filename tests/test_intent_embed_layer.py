"""B 级（embedding）判定规则测试 —— 零模型、零网络。

这里锁的是一条**曾经造成线上性事故的规则**：

    旧实现有一档 `top_score < 0.45 → out_of_scope`。
    示例集只覆盖"退换货/发票/物流"语境，而知识库里真正的语料是
    岚盾 L1/L2、云枢 S3、上门安装、故障排查 —— 这些真问题跟示例集都不像，
    余弦分普遍落在 0.30~0.44，于是被"低分即超范围"整批拒答
    （实测 eval/golden 的 174 条金标问题，125 条被判 out_of_scope）。

现在的规则：**只有最高分够高、且跟第二名拉开差距**才按示例集下结论；
低分或咬得紧 → 返回 None，交给 C 级 LLM 仲裁（它才看得懂语义）。

用例里的向量都是基向量，余弦值是精确可算的，不吃模型版本。
"""
import pytest

from intent import classifier as clf
from intent.schemas import IntentName

INTENTS = [
    IntentName.KB_QUESTION,
    IntentName.CHITCHAT,
    IntentName.STATUS_QUERY,
    IntentName.HUMAN_HANDOFF,
    IntentName.OUT_OF_SCOPE,
]


class _FakeEmbeddings:
    """把 query 直接映射成预置向量，跳过真实 embedding 模型。"""

    def __init__(self, mapping: dict[str, list[float]]):
        self._mapping = mapping

    def embed_query(self, text: str) -> list[float]:
        return self._mapping[text]


def _basis(i: int, dim: int = len(INTENTS)) -> list[float]:
    v = [0.0] * dim
    v[i] = 1.0
    return v


def _embed_classifier(query: str, q_vec: list[float]) -> clf.EmbeddingClassifier:
    """每个意图只放一条示例向量，且互为正交基 → 相似度等于分量占比。"""
    c = clf.EmbeddingClassifier.__new__(clf.EmbeddingClassifier)  # 不走 __init__（不建模型）
    c._embeddings = _FakeEmbeddings({query: q_vec})
    c._vectors = {intent: [_basis(i)] for i, intent in enumerate(INTENTS)}
    return c


def test_low_top_score_is_not_out_of_scope():
    """本次修复的核心：最高分低于旧分界线 0.45 时，不许下 out_of_scope。

    查询向量对五个意图均匀 → 每个余弦都等于 1/√5 ≈ 0.4472（低于旧阈值 0.45）。
    旧实现返回 OUT_OF_SCOPE（前端就是那句"超出我的服务范围"）；
    新实现必须返回 None，把它交给 C 级 LLM 仲裁。
    """
    c = _embed_classifier("q", [1.0, 1.0, 1.0, 1.0, 1.0])
    intent, score, scores = c.classify("q")

    assert score < 0.45, "构造出来的分数必须落在旧阈值以下，否则这条用例锁不住回归"
    assert intent is None, "低分只说明示例集没覆盖，不等于超出业务范围"
    assert scores  # 调试用分数照常带出来


def test_clear_off_topic_still_short_circuits_to_out_of_scope():
    """真正跑题的问题仍要能直接短路（省一次 LLM 调用）。"""
    q_vec = [0.2, 0.0, 0.0, 0.0, 3.0]  # 几乎与 out_of_scope 基向量同向
    c = _embed_classifier("q", q_vec)
    intent, score, _ = c.classify("q")

    assert intent == IntentName.OUT_OF_SCOPE
    assert score >= clf.EMBED_HIGH_THRESHOLD


def test_high_score_winner_passes_through():
    c = _embed_classifier("q", _basis(0))  # 与 kb_question 完全同向
    intent, score, _ = c.classify("q")

    assert intent == IntentName.KB_QUESTION
    assert score == pytest.approx(1.0)


def test_close_race_goes_to_llm_even_when_score_is_high():
    """第一第二名咬得紧 → 连"最像哪个意图"都算不上，不能硬判。"""
    q_vec = [5.0, 4.95, 0.0, 0.0, 0.0]
    c = _embed_classifier("q", q_vec)
    intent, score, _ = c.classify("q")

    assert score >= clf.EMBED_HIGH_THRESHOLD  # 分不低
    assert intent is None  # 但没拉开差距


@pytest.mark.parametrize("threshold_name", ["EMBED_LOW_THRESHOLD"])
def test_low_score_threshold_constant_is_gone(threshold_name):
    """显式删掉那档阈值：常量还在，就有人会再用它做"低分即超范围"。"""
    assert not hasattr(clf, threshold_name)
