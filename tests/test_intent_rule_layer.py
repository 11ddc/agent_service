"""意图识别 A 级(规则层)测试 —— 零服务:不触发 embedding / LLM。

规则层 rule_classify 是纯正则逻辑;本文件直接测它,再对 public 入口
classify() 验证"规则命中即短路、绝不会走到 B/C 级"。
"""
import pytest

from intent import classifier as clf
from intent.schemas import IntentName

TOOL_CALL = IntentName.TOOL_CALL
HUMAN_HANDOFF = IntentName.HUMAN_HANDOFF
CHITCHAT = IntentName.CHITCHAT
STATUS_QUERY = IntentName.STATUS_QUERY


@pytest.mark.parametrize(
    "query",
    [
        "帮我查一下订单",
        "查询我的物流",
        "看看快递到哪了",
        "我的订单到哪了",
        "查一下包裹状态",
        "订单发货了吗",
        "发货了吗",
        "查查我的单号",
    ],
)
def test_rule_hits_tool_call(query):
    intent, _ = clf.rule_classify(query)

    assert intent == TOOL_CALL


@pytest.mark.parametrize(
    "query",
    [
        "订单怎么取消",
        "发货时效是多久",
        "退换货的流程是什么",
        "物流政策有什么规定",
        "退款多久到账",
        "运费险怎么用",
    ],
)
def test_rule_avoids_tool_call_for_kb_questions(query):
    # 负向断言:这些是"问规则/知识"的说法,不该被 TOOL_CALL 规则短路
    intent, _ = clf.rule_classify(query)

    assert intent != TOOL_CALL


@pytest.mark.parametrize(
    "query",
    [
        "我要转人工",
        "转人工客服",
        "我要投诉",
        "帮我联系真人处理",
        "找专员",
        "我要举报",
    ],
)
def test_rule_hits_human_handoff(query):
    intent, _ = clf.rule_classify(query)

    assert intent == HUMAN_HANDOFF


@pytest.mark.parametrize(
    "query",
    ["你是谁", "你能做什么", "你好", "谢谢", "好的", "再见", "在吗", "哈哈"],
)
def test_rule_hits_chitchat(query):
    intent, _ = clf.rule_classify(query)

    assert intent == CHITCHAT


@pytest.mark.parametrize(
    "query",
    [
        "知识库里有什么文档",
        "目前知识库有多少资料",
        "知识库初始化了吗",
        "库里有什么内容",
    ],
)
def test_rule_hits_status_query(query):
    intent, _ = clf.rule_classify(query)

    assert intent == STATUS_QUERY


def test_rule_strips_greeting_prefix_before_returning_rest():
    # 剥掉寒暄前缀后剩余部分有实质内容 → intent 为 None,返回剥离后的文本
    intent, rest = clf.rule_classify("你好，帮我查一下退换货流程")

    assert intent is None
    assert rest == "帮我查一下退换货流程"


# ── public 入口 classify():规则命中必须短路,B/C 级不应被触碰 ──


class _ExplodingClassifier:
    """若被构造即记录;若被调用即失败(证明规则层已短路)。"""

    def __init__(self, *args, **kwargs):
        pass

    def classify(self, *args, **kwargs):
        raise AssertionError("规则命中后不应进入 embedding/LLM 分类")


@pytest.fixture
def rule_only(monkeypatch):
    # 把 B/C 两级替换成"一碰就炸"的替身
    monkeypatch.setattr(clf, "EmbeddingClassifier", _ExplodingClassifier)
    monkeypatch.setattr(clf, "LLMClassifier", _ExplodingClassifier)


def test_classify_short_circuits_on_rule_hit(rule_only):
    result = clf.classify("我的订单到哪了")

    assert result.intent == IntentName.TOOL_CALL
    assert result.method == "rule"
    assert result.confidence == 1.0


def test_classify_empty_question_is_ambiguous(rule_only):
    result = clf.classify("   ")

    assert result.intent == IntentName.AMBIGUOUS
    assert result.confidence == 0
    assert result.method == "rule"
