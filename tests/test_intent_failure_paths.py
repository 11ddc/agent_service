"""意图漏斗失败路径测试 —— 零服务:embedding / LLM 全部用桩替换。

重点保证"某一层挂掉不能让整个漏斗失效":
- B 级(embedding)不可用 → 降级到 C 级 LLM 仲裁,而不是判定为"未识别";
- C 级也挂 → ambiguous + 明确的 reason(供路由层决定走 Agent 兜底);
- 空问题 → reason=empty(唯一应该被短路的 ambiguous)。
"""
import pytest

from intent import classifier as clf
from intent.schemas import IntentName, IntentOutput, IntentReason, IntentSlots


class _EmbedStub:
    def __init__(self, result=None, exc: Exception | None = None):
        self.result = result
        self.exc = exc
        self.calls: list[str] = []

    def classify(self, query):
        self.calls.append(query)
        if self.exc is not None:
            raise self.exc
        return self.result


class _LLMStub:
    def __init__(self, result=None, exc: Exception | None = None):
        self.result = result
        self.exc = exc
        self.calls: list[str] = []

    def classify(self, query):
        self.calls.append(query)
        if self.exc is not None:
            raise self.exc
        return self.result


def _classifier(embed, llm) -> clf.IntentClassifier:
    """跳过 __init__（不建 DashScope / ChatOpenAI 客户端、不联网）。"""
    c = clf.IntentClassifier.__new__(clf.IntentClassifier)
    c._embed = embed
    c._llm = llm
    return c


def _out(intent=IntentName.KB_QUESTION, confidence=0.9) -> IntentOutput:
    return IntentOutput(intent=intent, confidence=confidence, slots=IntentSlots())


# ── 空问题 ──


def test_blank_query_marked_empty_without_touching_any_layer():
    embed = _EmbedStub(exc=AssertionError("不该被调用"))
    llm = _LLMStub(exc=AssertionError("不该被调用"))
    result = _classifier(embed, llm).classify("   ")

    assert result.intent == IntentName.AMBIGUOUS
    assert result.reason == IntentReason.EMPTY
    assert result.confidence == 0
    assert embed.calls == [] and llm.calls == []


# ── B 级不可用 → 降级到 C 级（本次修复的核心）──


def test_embedding_failure_degrades_to_llm_not_to_unknown():
    embed = _EmbedStub(exc=RuntimeError("Arrearage / 模型不可用"))
    llm = _LLMStub(result=_out(IntentName.KB_QUESTION, 0.9))
    result = _classifier(embed, llm).classify("我要退货")

    assert result.intent == IntentName.KB_QUESTION
    assert result.method == "llm"
    assert result.reason is None  # 降级成功,不算兜底
    assert llm.calls == ["我要退货"]


def test_embedding_failure_does_not_leak_exception():
    embed = _EmbedStub(exc=ValueError("status_code: 400"))
    llm = _LLMStub(result=_out())
    # 旧实现:异常会穿出 classify()，被 intent_router_node 吞掉，
    # 结果所有问题都走 agent_flow，RAG 被静默绕过。
    result = _classifier(embed, llm).classify("保修期多久")

    assert result.intent == IntentName.KB_QUESTION


# ── 两层都挂 / 只有 C 级挂 ──


def test_both_layers_failed_marks_embedding_error():
    embed = _EmbedStub(exc=RuntimeError("embedding down"))
    llm = _LLMStub(exc=RuntimeError("llm down"))
    result = _classifier(embed, llm).classify("我要退货")

    assert result.intent == IntentName.AMBIGUOUS
    assert result.method == "llm"
    assert result.reason == IntentReason.EMBEDDING_ERROR
    assert result.confidence == 0


def test_only_llm_failed_marks_llm_error():
    embed = _EmbedStub(result=(None, 0.5, {"kb_question": 0.5}))  # 歧义 → 转 C 级
    llm = _LLMStub(exc=RuntimeError("llm down"))
    result = _classifier(embed, llm).classify("我要退货")

    assert result.reason == IntentReason.LLM_ERROR
    assert result.scores == {"kb_question": 0.5}  # 调试用分数仍然带出来


def test_llm_low_confidence_marks_low_confidence():
    embed = _EmbedStub(result=(None, 0.5, {}))
    llm = _LLMStub(result=_out(IntentName.KB_QUESTION, 0.3))
    result = _classifier(embed, llm).classify("有点说不清的问题")

    assert result.intent == IntentName.AMBIGUOUS
    assert result.reason == IntentReason.LOW_CONFIDENCE
    assert result.confidence == pytest.approx(0.3)


# ── 正常路径没被改坏 ──


def test_rule_hit_short_circuits_before_both_layers():
    embed = _EmbedStub(exc=AssertionError("不该被调用"))
    llm = _LLMStub(exc=AssertionError("不该被调用"))
    result = _classifier(embed, llm).classify("帮我查一下订单")

    assert result.intent == IntentName.TOOL_CALL
    assert result.method == "rule"
    assert result.reason is None


def test_embedding_high_confidence_result_passes_through():
    embed = _EmbedStub(result=(IntentName.KB_QUESTION, 0.88, {"kb_question": 0.88}))
    llm = _LLMStub(exc=AssertionError("不该被调用"))
    result = _classifier(embed, llm).classify("我要退货")

    assert result.method == "embedding"
    assert result.confidence == pytest.approx(0.88)
    assert result.reason is None
