"""rag_node 预算接线测试 —— 零服务：检索、重排、生成全部换成替身。

锁住一件事：**预检确实夹在「重排之后、生成之前」**，
且 rag_node 不再走无预算的 generate_answer。
"""
import pytest
from langchain_core.documents import Document

import agent.graph as g
from context_budget import input_budget


class _FakeGenerator:
    """替身生成器：记录调用参数；老方法被调用就直接失败。"""

    def __init__(self, answer="预算内生成的答案"):
        self.calls = []
        self._answer = answer

    def generate_answer_within_budget(
        self, query, docs, *, budget_tokens=None, emit=None
    ):
        self.calls.append(
            {
                "query": query,
                "docs": list(docs),
                "budget_tokens": budget_tokens,
                "emit": emit,
            }
        )
        return {
            "answer": self._answer,
            "used_tokens": 123,
            "kept": len(docs),
            "dropped": 0,
            "truncated": False,
            "budget_tokens": budget_tokens,
            "degraded": False,
            "retries": 0,
        }

    def generate_answer(self, query, docs):  # pragma: no cover - 被调用即测试失败
        raise AssertionError("rag_node 不应再走无预算的 generate_answer")


def test_rag_node_runs_budget_check_between_reorder_and_generate(monkeypatch):
    docs = [Document(page_content="资料一"), Document(page_content="资料二")]
    order: list[str] = []

    def _retrieve(_q):
        order.append("retrieve")
        return docs

    def _reorder(_q, _docs):
        order.append("reorder")
        return _docs

    monkeypatch.setattr(g, "retrieve_sync", _retrieve)
    monkeypatch.setattr(g, "reordering", _reorder)
    fake = _FakeGenerator()
    monkeypatch.setattr(g, "Generator", fake)

    out = g.rag_node({"question": "退货流程是什么", "session_id": "s1"})

    assert out["answer"] == "预算内生成的答案"
    assert order == ["retrieve", "reorder"]  # 顺序没被打乱
    assert len(fake.calls) == 1
    # 生成前拿到了预检预算，且传的就是重排后的资料
    assert fake.calls[0]["budget_tokens"] == g.RAG_DOC_BUDGET
    assert fake.calls[0]["query"] == "退货流程是什么"
    assert fake.calls[0]["docs"] == docs
    # 同步链路拿到的流式出口必须是 None（emit 有值就代表要边生成边推流），
    # rag_node 自己判断 current_emitter()：/chat 没设 emitter → 传 None
    assert fake.calls[0]["emit"] is None


def test_rag_node_skips_generation_when_retrieval_empty(monkeypatch):
    monkeypatch.setattr(g, "retrieve_sync", lambda _q: [])
    fake = _FakeGenerator()
    monkeypatch.setattr(g, "Generator", fake)

    out = g.rag_node({"question": "q", "session_id": "s"})

    assert "未在知识库中找到" in out["answer"]
    assert fake.calls == [], "没有资料时不该调生成模型"


def test_rag_node_degrades_on_generation_error(monkeypatch):
    """非超限异常仍由原有兜底接管，用户看到的是可读提示而不是堆栈。"""
    monkeypatch.setattr(g, "retrieve_sync", lambda _q: [Document(page_content="资料")])
    monkeypatch.setattr(g, "reordering", lambda _q, docs: docs)

    class _Boom:
        def generate_answer_within_budget(self, *a, **kw):
            raise RuntimeError("dashscope 500")

    monkeypatch.setattr(g, "Generator", _Boom())

    out = g.rag_node({"question": "q", "session_id": "s"})

    assert out["answer"] == "知识库检索失败，请稍后重试。"


def test_rag_doc_budget_reserves_room_for_prompt_template():
    """资料预算必须小于输入总预算：RAG_PROMPT 模板和用户问题也要占额度。"""
    assert 0 < g.RAG_DOC_BUDGET < input_budget()
