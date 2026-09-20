"""LangGraph 纯路由逻辑测试 —— 零服务:不触发 LLM / 检索 / Redis。

覆盖 graph.py 里只读 state 做判断的节点:route_by_intent、route_after_split、
short_circuit_node、tool_continue、merge_node。意图结果用 IntentResult 手工构造,
不走 classify()。
"""
import pytest
from langchain_core.messages import AIMessage

import agent.graph as g
from intent.schemas import IntentName, IntentReason, IntentResult


def _res(intent, confidence=1.0, method="rule", reason=None):
    return IntentResult(
        intent=intent, confidence=confidence, method=method, reason=reason
    )


# ── route_by_intent:意图 → 下一节点 ──


@pytest.mark.parametrize(
    "intent,expected",
    [
        (IntentName.TOOL_CALL, "tool_agent"),
        (IntentName.KB_QUESTION, "rag_flow"),
        (IntentName.CHITCHAT, "short_circuit"),
        (IntentName.HUMAN_HANDOFF, "short_circuit"),
        (IntentName.STATUS_QUERY, "short_circuit"),
        (IntentName.OUT_OF_SCOPE, "short_circuit"),
    ],
)
def test_route_by_intent_maps_regular_intents(intent, expected):
    state = {"intent_result": _res(intent)}

    assert g.route_by_intent(state) == expected


def test_route_by_intent_none_result_falls_back_to_agent():
    assert g.route_by_intent({"intent_result": None}) == "agent_flow"


def test_route_by_intent_empty_question_short_circuits():
    state = {
        "intent_result": _res(
            IntentName.AMBIGUOUS, confidence=0.0, reason=IntentReason.EMPTY
        )
    }

    assert g.route_by_intent(state) == "short_circuit"


def test_route_by_intent_classifier_failure_goes_agent():
    """回归：分类器失败时 confidence 也是 0，不能当"空问题"短路。

    旧实现按 confidence == 0 判断空问题，于是 LLM 仲裁一失败，真实问题会被
    回一句"我没收到您的问题"。现在只有 reason=empty 才短路。
    """
    for reason in (
        IntentReason.LLM_ERROR,
        IntentReason.EMBEDDING_ERROR,
        IntentReason.LOW_CONFIDENCE,
    ):
        state = {
            "intent_result": _res(
                IntentName.AMBIGUOUS, confidence=0.0, method="llm", reason=reason
            )
        }

        assert g.route_by_intent(state) == "agent_flow", reason


def test_route_by_intent_ambiguous_low_confidence_goes_agent():
    state = {"intent_result": _res(IntentName.AMBIGUOUS, confidence=0.5)}

    assert g.route_by_intent(state) == "agent_flow"


def test_route_by_intent_ambiguous_without_reason_goes_agent():
    state = {"intent_result": _res(IntentName.AMBIGUOUS, confidence=0.0)}

    assert g.route_by_intent(state) == "agent_flow"


# ── short_circuit_node:空问题话术 vs 分类器失败 ──


def test_short_circuit_empty_question_prompts_reinput():
    out = g.short_circuit_node(
        {"intent_result": _res(IntentName.AMBIGUOUS, 0.0, "rule", IntentReason.EMPTY)}
    )

    assert "没收到您的问题" in out["answer"]


def test_short_circuit_classifier_failure_does_not_claim_empty_question():
    out = g.short_circuit_node(
        {
            "intent_result": _res(
                IntentName.AMBIGUOUS, 0.0, "llm", IntentReason.LLM_ERROR
            )
        }
    )

    assert "没收到您的问题" not in out["answer"]


# ── route_after_split:子问题数量 → 单问题图 / 多问题循环 ──


@pytest.mark.parametrize(
    "subs,expected",
    [
        (None, "question_flow"),
        ([], "question_flow"),
        (["退货流程是啥"], "question_flow"),
        (["怎么退货", "怎么换货"], "multi_loop"),
        (["a", "b", "c"], "multi_loop"),
    ],
)
def test_route_after_split(subs, expected):
    state = {"question": "q", "sub_questions": subs}

    assert g.route_after_split(state) == expected


# ── tool_continue:最后一条消息是否带工具调用 ──


def test_tool_continue_without_tool_calls_goes_data_node():
    state = {"messages": [AIMessage(content="我直接回答")]}

    assert g.tool_continue(state) == "data_node"


def test_tool_continue_with_tool_calls_goes_tool_call():
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "searchOrder", "args": {"query": "x"}, "id": "call_1"}
                ],
            )
        ]
    }

    assert g.tool_continue(state) == "tool_call"


# ── merge_node:answer + meta 组装 ──


def test_merge_node_without_intent_uses_fallback_meta():
    out = g.merge_node({"answer": "你好", "intent_result": None})

    assert out["answer"] == "你好"
    assert out["meta"] == {"intent": "unknown", "method": "fallback"}


def test_merge_node_builds_meta_from_intent_result():
    state = {
        "answer": "这是答案",
        "intent_result": _res(IntentName.KB_QUESTION, confidence=0.81234, method="embedding"),
    }
    out = g.merge_node(state)

    assert out["answer"] == "这是答案"
    assert out["meta"]["intent"] == "kb_question"
    assert out["meta"]["method"] == "embedding"
    assert out["meta"]["confidence"] == pytest.approx(0.8123)
    # 槽位全空时不输出 slots 键
    assert "slots" not in out["meta"]
    # 正常判定不输出 reason 键
    assert "reason" not in out["meta"]


def test_merge_node_exposes_fallback_reason():
    state = {
        "answer": "",
        "intent_result": _res(
            IntentName.AMBIGUOUS, 0.0, "llm", IntentReason.LLM_ERROR
        ),
    }
    out = g.merge_node(state)

    assert out["meta"]["method"] == "llm"
    assert out["meta"]["reason"] == "llm_error"
