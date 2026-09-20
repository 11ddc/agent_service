"""查询改写门控 + 结果校验测试 —— 零服务:LLM 客户端用桩替换,不发任何请求。

分两层测:
1. needs_rewrite():门控判定,决定"要不要花一次 LLM 调用";
2. _accepts()/rewrite_query():改写结果校验,决定"敢不敢采用这个改写"。
"""
from types import SimpleNamespace

import pytest

import query_rewrite.rewriter as rw

HISTORY = [
    {"role": "user", "content": "我想问下云枢S3 Pro"},
    {"role": "assistant", "content": "云枢S3 Pro 是我们的旗舰型号,支持…"},
]


# ── 桩：假客户端（只暴露 chat.completions.create 这一条被用到的路径）──


class _FakeCompletions:
    def __init__(self, content: str = "", exc: Exception | None = None):
        self.content = content
        self.exc = exc
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc is not None:
            raise self.exc
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


class _FakeClient:
    def __init__(self, content: str = "", exc: Exception | None = None):
        self.completions = _FakeCompletions(content, exc)
        self.chat = SimpleNamespace(completions=self.completions)


def _use_client(monkeypatch, content: str = "", exc: Exception | None = None):
    client = _FakeClient(content, exc)
    monkeypatch.setattr(rw, "_get_client", lambda: client)
    return client


# ── 1. 门控：依赖上文的说法必须命中 ──


@pytest.mark.parametrize(
    "query",
    [
        "那退货呢？",  # "那…呢"：最典型的省略式追问（旧实现漏改）
        "换成红色的呢",  # 结尾"呢"
        "多久能到",  # 追问词开头 + 极短
        "为什么要这么久",
        "继续",  # 极短追问
        "再详细点",
        "还有别的吗",
        "它保修多久",  # 指代词
        "这个怎么弄",
        "那笔退款到账了吗",
    ],
)
def test_needs_rewrite_for_context_dependent(query):
    assert rw.needs_rewrite(query, HISTORY) is True


@pytest.mark.parametrize(
    "query",
    [
        "退换货的流程是什么",
        "帮我查一下订单物流进度",
        "知识库里有哪些文档",
        "云枢S3 Pro 的保修期是多久",
    ],
)
def test_self_contained_question_skips_llm(query):
    assert rw.needs_rewrite(query, HISTORY) is False


def test_without_history_never_rewrites():
    # 没有历史就无从补全:即便是"那退货呢"也不该调 LLM
    assert rw.needs_rewrite("那退货呢？", []) is False
    assert rw.needs_rewrite("那退货呢？", None) is False
    assert rw.needs_rewrite("", HISTORY) is False


# ── 2. 校验：什么改写结果敢用 / 不敢用 ──


def test_accepts_context_completion():
    assert rw._accepts("它保修多久？", "云枢S3 Pro 保修多久？") is True


def test_accepts_rewrite_that_drops_pure_pronoun_question():
    # "它呢"整句都是虚词:剥完没有实词,不能用实词规则卡它
    assert rw._accepts("它呢", "云枢S3 Pro 的保修期是多久？") is True


def test_rejects_info_loss():
    # 原问题实词"退货"在改写里消失 → 拒绝
    assert rw._accepts("那退货呢", "请问您想咨询什么问题？") is False


def test_rejects_dropped_order_number():
    # 单号被"顺手改掉"是最危险的静默错误
    assert rw._accepts("那单 12345 到哪了", "我的订单什么时候能到") is False


def test_rejects_overlong_rewrite():
    assert rw._accepts("它呢", "云枢S3 Pro " * 40) is False


def test_rejects_empty():
    assert rw._accepts("那退货呢", "") is False


def test_sanitize_takes_first_line_and_strips_decoration():
    assert rw._sanitize("改写后：云枢S3 Pro 保修多久？\n说明：补全了指代。") == (
        "云枢S3 Pro 保修多久？"
    )
    assert rw._sanitize('“云枢S3 Pro 保修多久？”') == "云枢S3 Pro 保修多久？"
    assert rw._sanitize("   \n  ") == ""


# ── 3. rewrite_query 端到端（打桩）──


def test_rewrite_applied_when_valid(monkeypatch):
    _use_client(monkeypatch, "云枢S3 Pro 保修多久？")

    assert rw.rewrite_query("它保修多久？", HISTORY) == "云枢S3 Pro 保修多久？"


def test_self_contained_question_never_calls_llm(monkeypatch):
    client = _use_client(monkeypatch, "不该被用到")

    assert rw.rewrite_query("退换货的流程是什么", HISTORY) == "退换货的流程是什么"
    assert client.completions.calls == []


def test_multiline_output_keeps_first_line(monkeypatch):
    _use_client(monkeypatch, "改写后：云枢S3 Pro保修多久？\n说明：补全了指代")

    assert rw.rewrite_query("它保修多久？", HISTORY) == "云枢S3 Pro保修多久？"


def test_hallucinated_rewrite_falls_back_to_original(monkeypatch):
    _use_client(monkeypatch, "请问您想咨询什么问题？")

    assert rw.rewrite_query("那退货呢", HISTORY) == "那退货呢"


def test_llm_exception_falls_back_to_original(monkeypatch):
    _use_client(monkeypatch, exc=RuntimeError("timeout"))

    assert rw.rewrite_query("那退货呢", HISTORY) == "那退货呢"


def test_missing_api_key_falls_back_to_original(monkeypatch):
    monkeypatch.setattr(rw, "_get_client", lambda: None)

    assert rw.rewrite_query("那退货呢", HISTORY) == "那退货呢"


def test_unchanged_rewrite_returns_original(monkeypatch):
    _use_client(monkeypatch, "那退货呢")

    assert rw.rewrite_query("那退货呢", HISTORY) == "那退货呢"
