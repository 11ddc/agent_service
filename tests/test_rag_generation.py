"""RAG 生成侧测试 —— 零服务:不调 DashScope API。

_format_docs 是纯函数直接测;generate_answer 的 OpenAI 客户端替换成假客户端。
"""
import pytest
from langchain_core.documents import Document

from rag.generatellm import RAGGenerator, _format_docs


def test_format_docs_numbering_compatible_docs_and_strings():
    out = _format_docs(
        [Document(page_content="内容甲", metadata={}), "内容乙字符串"]
    )

    assert "[doc1] 内容甲" in out
    assert "[doc2] 内容乙字符串" in out


def test_generate_answer_returns_model_content(monkeypatch):
    class _Message:
        content = "基于资料生成的答案"

    class _Choice:
        message = _Message()

    class _Result:
        choices = [_Choice()]

    class _Completions:
        @staticmethod
        def create(**kwargs):
            return _Result()

    class _FakeClient:
        chat = type("_Chat", (), {"completions": _Completions()})()

    gen = RAGGenerator()
    monkeypatch.setattr(gen, "generate_client", _FakeClient())

    answer = gen.generate_answer("问题", [Document(page_content="资料")])

    assert answer == "基于资料生成的答案"


def test_generate_answer_re_raises_on_api_error(monkeypatch):
    class _BrokenCompletions:
        @staticmethod
        def create(**kwargs):
            raise RuntimeError("dashscope 500")

    class _FakeClient:
        chat = type("_Chat", (), {"completions": _BrokenCompletions()})()

    gen = RAGGenerator()
    monkeypatch.setattr(gen, "generate_client", _FakeClient())

    with pytest.raises(RuntimeError, match="dashscope 500"):
        gen.generate_answer("问题", [Document(page_content="资料")])
