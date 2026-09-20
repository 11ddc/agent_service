"""客服工具（`tools_agent/kb_tools.py`）测试 —— 零服务：MySQL / Chroma / Redis 全部打桩。

这些工具与普通业务代码有一点本质不同：**它们是给模型决策用的**。
所以除返回值正确之外，还要守住三条契约：

1. 每个工具的 docstring 必须说清"什么时候该用、什么时候不该用" ——
   模型没有别的信息来源，docstring 就是它的选型依据；
2. 底层不可用（MySQL 挂了 / 库为空 / 文件名不存在）时必须返回**可读的降级话术**，
   绝不能抛异常 —— 工具抛错会打断整条 tool_agent 子图；
3. 返回文本必须带**可核验的出处**（文件名 + 章节 + 页码），否则工具结果无法被
   引用，等于又造了一个不可溯源的信息源。
"""
import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

import tools_agent.kb_tools as kb
from db import MySQLUnavailable
from db.document_store import DocumentRow


# ── 桩：文档元数据 ──────────────────────────────────────────


def _row(filename, source, chunk_count=10, status="ok", err=None):
    return DocumentRow(
        doc_id="d_" + filename,
        filename=filename,
        source=source,
        uploaded_at=None,
        parsed_status=status,
        error_msg=err,
        chunk_count=chunk_count,
        parent_count=max(1, chunk_count // 3),
        chunk_schema_ver="v3-simple-1",
    )


ROWS = [
    _row("售后服务政策_v2_2025.docx", "F:/kb/售后服务政策_v2_2025.docx", 12),
    _row("备件总表.xlsx", "F:/kb/备件总表.xlsx", 40),
    _row("损坏.pdf", "F:/kb/损坏.pdf", 0, status="failed", err="PDF 已损坏，无法解析"),
]


class _FakeStore:
    def __init__(self, rows=None, exc=None):
        self._rows = ROWS if rows is None else rows
        self._exc = exc

    def all(self):
        if self._exc:
            raise self._exc
        return self._rows


@pytest.fixture
def fake_store(monkeypatch):
    def _install(rows=None, exc=None):
        store = _FakeStore(rows, exc)
        monkeypatch.setattr(kb, "_document_store", lambda: store)
        return store

    return _install


# ── 1. 全库检索 ─────────────────────────────────────────────


def test_search_knowledge_base_returns_text_with_citation(monkeypatch):
    docs = [
        Document(
            page_content="退货需在签收后 7 日内提出。",
            metadata={
                "source": "F:/kb/售后服务政策_v2_2025.docx",
                "breadcrumb": "第3章 > 3.2 退换货",
                "page_start": 5,
            },
        )
    ]
    monkeypatch.setattr(kb, "retrieve_sync", lambda q, k: docs)
    monkeypatch.setattr(kb, "reordering", lambda q, d: d)

    out = kb.search_knowledge_base.invoke({"query": "退货时效"})

    assert "退货需在签收后 7 日内提出。" in out
    # 出处三件套：文件名、章节、页码 —— 缺一个引用就核验不了
    assert "售后服务政策_v2_2025.docx" in out
    assert "第3章 > 3.2 退换货" in out
    assert "5" in out


def test_search_knowledge_base_reports_no_hit_clearly(monkeypatch):
    monkeypatch.setattr(kb, "retrieve_sync", lambda q, k: [])
    monkeypatch.setattr(kb, "reordering", lambda q, d: [])

    out = kb.search_knowledge_base.invoke({"query": "无关问题"})

    # 注意别写成 `assert "未" in out and "检索" in out or "没有" in out`：
    # and/or 优先级会让尾部那个 or 把判据放宽到几乎恒真，等于没断言。
    assert "未检索到" in out


def test_search_knowledge_base_degrades_when_kb_unavailable(monkeypatch):
    """知识库不可用要给可读话术，不能把异常抛进工具循环。"""
    from rag.rag import KnowledgeBaseError

    def _boom(q, k):
        raise KnowledgeBaseError("知识库连接失败，请检查后重试。")

    monkeypatch.setattr(kb, "retrieve_sync", _boom)

    out = kb.search_knowledge_base.invoke({"query": "退货"})

    assert "知识库" in out
    assert "失败" in out or "不可用" in out


def test_search_knowledge_base_truncates_long_snippets(monkeypatch):
    """工具结果要回灌进模型上下文，不能整篇塞进去。"""
    docs = [Document(page_content="啊" * 5000, metadata={"source": "F:/kb/a.md"})]
    monkeypatch.setattr(kb, "retrieve_sync", lambda q, k: docs)
    monkeypatch.setattr(kb, "reordering", lambda q, d: d)

    out = kb.search_knowledge_base.invoke({"query": "长文档"})

    assert len(out) < 3000


# ── 2. 限定文档检索（主 RAG 链路没有的能力）────────────────


def test_search_in_document_rejects_unknown_filename(fake_store):
    """文件名不在库里要明确说清楚，而不是静默返回空 —— 后者模型会当成"文档里没写"。"""
    fake_store()

    out = kb.search_in_document.invoke({"query": "保修", "filename": "不存在的文档.pdf"})

    assert "不存在的文档.pdf" in out
    assert "没有" in out or "未找到" in out


def test_search_in_document_filters_chroma_by_real_source(monkeypatch, fake_store):
    """限定检索必须落到 Chroma 的 source 过滤，且用**绝对路径**而不是文件名。"""
    fake_store()
    captured = {}

    class _Store:
        def similarity_search(self, query, k, filter=None):
            captured["query"] = query
            captured["filter"] = filter
            return [
                Document(
                    page_content="保修期为 24 个月。",
                    metadata={"source": "F:/kb/售后服务政策_v2_2025.docx"},
                )
            ]

    monkeypatch.setattr(kb, "_ready_store", lambda: _Store())

    out = kb.search_in_document.invoke(
        {"query": "保修多久", "filename": "售后服务政策_v2_2025.docx"}
    )

    assert captured["filter"] == {"source": "F:/kb/售后服务政策_v2_2025.docx"}
    assert "保修期为 24 个月。" in out


def test_search_in_document_reports_parse_failure(fake_store):
    """解析失败的文档要说清"内容没入库"，不能回一句"没检索到"。

    后者会让模型告诉用户"这份文档里没写"—— 把系统故障说成了业务事实。
    """
    fake_store()

    out = kb.search_in_document.invoke({"query": "保修", "filename": "损坏.pdf"})

    assert "解析失败" in out
    assert "损坏.pdf" in out


def test_search_in_document_matches_filename_by_partial_name(monkeypatch, fake_store):
    """用户说的是"售后服务政策"，文件名却是"售后服务政策_v2_2025.docx"。"""
    fake_store()
    captured = {}

    class _Store:
        def similarity_search(self, query, k, filter=None):
            captured["filter"] = filter
            return [Document(page_content="x", metadata={"source": "s"})]

    monkeypatch.setattr(kb, "_ready_store", lambda: _Store())

    out = kb.search_in_document.invoke({"query": "保修", "filename": "售后服务政策"})

    assert captured["filter"] == {"source": "F:/kb/售后服务政策_v2_2025.docx"}
    assert "没有名为" not in out


def test_search_in_document_reports_ambiguous_match(monkeypatch, fake_store):
    """子串命中多份时必须如实报歧义，绝不能随便挑一份 —— 挑错就是答错文档。"""
    fake_store(
        rows=[
            _row("备件总表_华东仓.xlsx", "F:/kb/备件总表_华东仓.xlsx"),
            _row("备件总表_华北仓.xlsx", "F:/kb/备件总表_华北仓.xlsx"),
        ]
    )

    out = kb.search_in_document.invoke({"query": "编码", "filename": "备件总表"})

    assert "多份" in out
    assert "备件总表_华东仓.xlsx" in out
    assert "备件总表_华北仓.xlsx" in out


# ── 3. 文档清单 ─────────────────────────────────────────────


def test_list_documents_lists_names_and_chunk_counts(fake_store):
    fake_store()

    out = kb.list_documents.invoke({})

    assert "售后服务政策_v2_2025.docx" in out
    assert "备件总表.xlsx" in out
    assert "12" in out and "40" in out  # 块数


def test_list_documents_marks_failed_parses(fake_store):
    """解析失败的文档必须标出来 —— 否则用户问"为什么搜不到"时无从解释。"""
    fake_store()

    out = kb.list_documents.invoke({})

    assert "损坏.pdf" in out
    assert "失败" in out


def test_list_documents_filters_by_keyword(fake_store):
    fake_store()

    out = kb.list_documents.invoke({"keyword": "备件"})

    assert "备件总表.xlsx" in out
    assert "售后服务政策_v2_2025.docx" not in out


def test_list_documents_reports_empty_library(fake_store):
    fake_store(rows=[])

    out = kb.list_documents.invoke({})

    assert "空" in out or "没有" in out


def test_list_documents_degrades_when_mysql_unavailable(fake_store):
    fake_store(exc=MySQLUnavailable("连接 MySQL 失败"))

    out = kb.list_documents.invoke({})

    assert "MySQL" in out or "不可用" in out


# ── 4. 文档章节概览 ─────────────────────────────────────────


def test_get_document_outline_lists_sections(monkeypatch, fake_store):
    fake_store()
    monkeypatch.setattr(
        kb,
        "_parent_outline",
        lambda doc_id: ["第1章 总则", "第3章 > 3.2 退换货", "第3章 > 3.3 退款"],
    )

    out = kb.get_document_outline.invoke({"filename": "售后服务政策_v2_2025.docx"})

    assert "第3章 > 3.2 退换货" in out
    assert "第1章 总则" in out


def test_get_document_outline_rejects_unknown_filename(fake_store):
    fake_store()

    out = kb.get_document_outline.invoke({"filename": "不存在.docx"})

    assert "不存在.docx" in out
    assert "没有" in out or "未找到" in out


def test_get_document_outline_degrades_when_mysql_unavailable(monkeypatch, fake_store):
    fake_store()

    def _boom(doc_id):
        raise MySQLUnavailable("连接 MySQL 失败")

    monkeypatch.setattr(kb, "_parent_outline", _boom)

    out = kb.get_document_outline.invoke({"filename": "售后服务政策_v2_2025.docx"})

    assert "不可用" in out or "MySQL" in out


# ── 5. 给模型的契约：docstring 与注册 ───────────────────────


ALL_TOOLS = [
    kb.search_knowledge_base,
    kb.search_in_document,
    kb.list_documents,
    kb.get_document_outline,
]


@pytest.mark.parametrize("tool", ALL_TOOLS, ids=lambda t: t.name)
def test_every_tool_tells_the_model_when_to_use_it(tool):
    """docstring 是模型唯一的选型依据：缺"何时用"就等于这个工具永远不会被正确调用。"""
    desc = tool.description or ""

    assert len(desc) >= 40, f"{tool.name} 的 docstring 太短，模型无法判断何时使用"
    assert any(k in desc for k in ("何时", "使用场景", "适用", "调用")), (
        f"{tool.name} 没说清什么时候该用"
    )


def test_model_sees_filename_needs_no_path():
    """Chroma 里存的是绝对路径，但模型只该给文件名 —— 映射必须由工具自己做。

    让模型拼绝对路径是坏设计：它拿不到真实路径，只会编一个出来；而"编出来的
    路径查不到"在检索侧表现为空结果，又是一次静默劣化。

    ⚠️ 实测（langchain 1.3 / langchain-core）：Google 风格的 `Args:` **不会**被
    解析成 per-parameter description —— `tool.args` 里只有 title/type/default，
    整段 docstring（含 Args）是塞进 `tool.description` 的。所以这条约束必须
    在 description 里断言，断言 args 会 KeyError 且保护不到任何东西。
    """
    desc = kb.search_in_document.description

    assert "filename:" in desc
    assert "不需要路径" in desc


# ── 6. 接线：锁住"工具写了但没挂上"的两处回归 ────────────────


def test_fallback_agent_actually_registers_tools():
    """回归：`create_agent` 曾经没传 `tools=`，两个 @tool 从未注册 ——
    兜底 Agent 查不了知识库，而它的职责恰恰是"检索/路由失败时兜底回答"。
    工具写得再好，没挂上就等于不存在。
    """
    from agent.langchina import AGENT_TOOLS

    names = {t.name for t in AGENT_TOOLS}

    assert "search_knowledge_base" in names
    assert "search_in_document" in names
    assert len(AGENT_TOOLS) == len(set(names)), "工具名不能重复，否则分派会互相覆盖"


def test_tool_call_node_invokes_real_knowledge_base_tool(monkeypatch):
    """端到端：模型发出 tool_call → tool_call_node 真的把工具跑起来并回填结果。

    回归：tool_call_node 以前只有 `mcp__*` 和两个硬编码占位工具的分支，
    知识库工具会掉进"未知工具"分支，模型永远拿不到资料。
    """
    import agent.graph as g

    monkeypatch.setattr(
        kb,
        "retrieve_sync",
        lambda q, k: [
            Document(
                page_content="退货需在签收后 7 日内提出。",
                metadata={"source": "F:/kb/售后服务政策_v2_2025.docx"},
            )
        ],
    )
    monkeypatch.setattr(kb, "reordering", lambda q, d: d)

    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_knowledge_base",
                        "args": {"query": "退货时效"},
                        "id": "c1",
                    }
                ],
            )
        ]
    }

    out = g.tool_call_node(state)["messages"]

    assert len(out) == 1
    assert "退货需在签收后 7 日内提出。" in out[0].content
    assert out[0].tool_call_id == "c1"


def test_tool_call_node_reports_tool_failure_without_breaking_loop(monkeypatch):
    """工具抛错必须回填错误信息，而不是把工具循环打断。

    模型看到"查询失败"还能换个说法或建议转人工；链路断掉则整条
    tool_agent 子图失败，用户拿到的是兜底话术。
    """
    import agent.graph as g

    class _Boom:
        name = "list_documents"

        def invoke(self, args):
            raise RuntimeError("MySQL 连接超时")

    monkeypatch.setitem(g.LOCAL_TOOL_MAP, "list_documents", _Boom())

    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"name": "list_documents", "args": {}, "id": "c9"}],
            )
        ]
    }

    out = g.tool_call_node(state)["messages"]

    assert len(out) == 1
    assert "执行失败" in out[0].content
    assert "MySQL 连接超时" in out[0].content
