"""glm-4.5-air 重排序的离线单测。

全程不发请求：客户端换成桩，只验"提示词 → 解析 → 排序"这条链。
（默认套件的承诺是零外部服务、零真实出网，见 tests/conftest.py。）
"""
import re
import threading

import pytest
from langchain_core.documents import Document

from rag import glm_reranker
from rag.glm_reranker import GLMReranker, _parse_scores


def _doc(text, source="a.txt"):
    return Document(page_content=text, metadata={"source": source})


def _resp(content: str):
    """把一段文本包成 openai 风格的回包（只用到 choices[0].message.content）。"""
    message = type("_Msg", (), {"content": content})()
    choice = type("_Choice", (), {"message": message})()
    return type("_Resp", (), {"choices": [choice]})()


class _FakeCompletions:
    """按预设回复**依次**作答；带锁，多批并发时也不会崩。"""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self._lock = threading.Lock()

    def create(self, **kwargs):
        with self._lock:
            self.calls.append(kwargs)
            content = self.replies.pop(0) if self.replies else ""
        return _resp(content)


class _EchoCompletions:
    """按提示词里的候选内容回分：分数 = 文档编号 + 1。

    这样无论批次怎么并发调度，最终排序都是确定的 —— 并发下也能断言。
    """

    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        prompt = kwargs["messages"][1]["content"]
        self.calls.append(kwargs)
        items = re.findall(r"\[(\d+)\] d(\d+)", prompt)
        body = ",".join('{"index":%s,"score":%d}' % (i, int(n) + 1) for i, n in items)
        return _resp('{"scores":[%s]}' % body)


class _FakeClient:
    def __init__(self, completions):
        self.chat = type("_Chat", (), {"completions": completions})()


def _reranker(replies, **kwargs):
    r = GLMReranker(api_key="test-key", **kwargs)
    r._client = _FakeClient(_FakeCompletions(replies))
    return r


# ── 解析 ─────────────────────────────────────────────────────
def test_parse_scores_reads_plain_json():
    text = '{"scores":[{"index":1,"score":9},{"index":2,"score":3}]}'

    assert _parse_scores(text, 2) == [9.0, 3.0]


def test_parse_scores_tolerates_code_fence_and_missing_entries():
    text = '好的：\n```json\n{"scores":[{"index":2,"score":7}]}\n```'

    assert _parse_scores(text, 3) == [None, 7.0, None]


def test_parse_scores_falls_back_to_regex_scan():
    text = "index: 1, score: 8\nindex: 2, score: 2"

    assert _parse_scores(text, 2) == [8.0, 2.0]


def test_parse_scores_scales_percent_style_scores():
    assert _parse_scores('{"scores":[{"index":1,"score":85}]}', 1) == [8.5]


def test_parse_scores_returns_all_none_on_garbage():
    assert _parse_scores("我认为都挺相关的", 2) == [None, None]


# ── 排序 ─────────────────────────────────────────────────────
def test_rerank_scored_sorts_by_model_scores_and_keeps_original_objects():
    d1, d2, d3 = _doc("A"), _doc("B"), _doc("C")
    r = _reranker(
        ['{"scores":[{"index":1,"score":2},{"index":2,"score":9},{"index":3,"score":5}]}']
    )

    out = r.rerank_scored("q", [d1, d2, d3])

    assert [d.page_content for d, _ in out] == ["B", "C", "A"]
    assert out[0][0] is d2  # 回填原对象，metadata 不丢
    assert [s for _, s in out] == [9.0, 5.0, 2.0]


def test_rerank_truncates_to_top_n():
    docs = [_doc(f"d{i}", f"f{i}.txt") for i in range(4)]
    r = _reranker(
        ['{"scores":[{"index":1,"score":1},{"index":2,"score":4},'
         '{"index":3,"score":3},{"index":4,"score":2}]}']
    )

    assert [d.page_content for d in r.rerank("q", docs, 2)] == ["d1", "d2"]


def test_empty_or_single_doc_short_circuits():
    r = _reranker(['{"scores":[{"index":1,"score":5}]}'])

    assert r.rerank_scored("q", []) == []
    assert r.rerank("q", [], 5) == []  # 空候选不发请求
    # 单条也走模型（与 reordering 的分支不同：那边在 len==1 时根本不调重排）
    assert len(r.rerank_scored("q", [_doc("only")])) == 1


# ── 分批与失败 ────────────────────────────────────────────────
def test_docs_are_scored_in_batches_and_order_is_reassembled():
    docs = [_doc(f"d{i}", f"f{i}.txt") for i in range(5)]
    r = GLMReranker(api_key="test-key", batch_size=2)
    r._client = _FakeClient(_EchoCompletions())

    out = r.rerank_scored("q", docs)

    assert len(r._client.chat.completions.calls) == 3  # 2 + 2 + 1
    assert [d.page_content for d, _ in out] == ["d4", "d3", "d2", "d1", "d0"]


def test_batch_failure_propagates_so_reordering_can_degrade():
    docs = [_doc(f"d{i}", f"f{i}.txt") for i in range(4)]
    r = _reranker(["", '{"scores":[{"index":1,"score":1}]}'], batch_size=2)

    with pytest.raises(RuntimeError):
        r.rerank_scored("q", docs)


def test_unparsable_reply_raises_so_reordering_can_degrade():
    r = _reranker(["我不知道怎么打分"])

    with pytest.raises(RuntimeError):
        r.rerank_scored("q", [_doc("A"), _doc("B")])


def test_missing_api_key_is_reported_by_load_model(monkeypatch):
    monkeypatch.delenv("ZHI_PU_API_KEY", raising=False)
    r = GLMReranker(api_key=None)

    with pytest.raises(RuntimeError, match="ZHI_PU_API_KEY"):
        r.load_model()


def test_prompt_contains_query_and_numbered_candidates():
    r = _reranker(['{"scores":[{"index":1,"score":1}]}'], doc_chars=20)
    r.score_texts("怎么退货", ["第一段内容" + "x" * 50])

    prompt = r._client.chat.completions.calls[0]["messages"][1]["content"]
    assert "怎么退货" in prompt
    assert "[1]" in prompt
    assert "x" * 21 not in prompt  # 超长正文按 doc_chars 截断
    assert r._client.chat.completions.calls[0]["model"] == "glm-4.5-air"


def test_default_model_and_base_url_come_from_the_zhipu_env(monkeypatch):
    monkeypatch.setenv("ZHI_PU_API_KEY", "key-from-env")
    r = GLMReranker()

    assert r.model_name == "glm-4.5-air"
    assert r.base_url == glm_reranker.GLM_RERANK_BASE_URL
    assert r.api_key == "key-from-env"


def test_model_adapter_keeps_pair_order():
    """兼容旧 eval 脚本：rag.reranker.model.predict([[query, doc], ...])。"""
    r = _reranker(['{"scores":[{"index":1,"score":6}]}', '{"scores":[{"index":1,"score":1}]}'])

    assert r.model.predict([["q", _doc("A")], ["q", _doc("B")]]) == [6.0, 1.0]
