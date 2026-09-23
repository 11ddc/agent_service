"""RAG 检索链路单元测试 —— 零服务:不连 Chroma / 不调 embedding API / 不加载重排模型。

策略:
- 纯函数直接测:_rrf_fuse、_clean_query、_format_docs、_guess_image_ext、find_image_references、
  _load_txt、_load_file(重活 loader 打桩);
- 撞外部边界的编排函数,把 rag.rag 模块内的全局依赖替换成内存替身:
  Chroma(→get_status)、reranker(→reordering)、_vectorstore/_sparse_search/_ensure_ready
  (→retrieve),只验证编排逻辑本身。
"""
import asyncio

import pytest
from langchain_core.documents import Document

from rag import rag
from rag.structure import Section


def _doc(content, source="a.txt", **meta):
    return Document(page_content=content, metadata={"source": source, **meta})


def _section(text):
    """结构单元。loader 在两级切分改造后统一返回 list[Section]，不再返回 Document。"""
    return Section(text=text, section_path=[], order=0)


# ── 纯函数:图片引用提取 ──


def test_find_image_references_extracts_basenames():
    content = "![示意图](images/a.png) 文本 ![x](sub/b.jpg) ![y](../c.png)"

    refs = rag.find_image_references(content)

    assert refs == ["a.png", "b.jpg", "c.png"]


def test_find_image_references_no_images():
    assert rag.find_image_references("没有任何图片的纯文本") == []


# ── 纯函数:查询清洗(与 BM25 语料对齐用) ──


def test_clean_query_removes_chinese_punct_and_space():
    assert rag._clean_query("  苹果，退货。政策？  ") == "苹果退货政策"


def test_clean_query_keeps_ascii_alnum():
    assert rag._clean_query("订单 ORD-20260831-001 到了吗") == "订单ORD-20260831-001到了吗"


# ── 纯函数:RRF 融合 ──


def test_rrf_fuse_ranks_doc_hit_by_both_paths_first_and_dedupes():
    d1 = _doc("苹果退货政策", source="a.txt")
    d2_same_content = _doc("苹果退货政策", source="b.txt")  # 内容相同 → 视为同一文档
    d3 = _doc("运费险", source="c.txt")

    out = rag._rrf_fuse([d1], [d2_same_content, d3], top_n=10)

    assert [d.page_content for d in out] == ["苹果退货政策", "运费险"]
    # 去重时保留先出现的对象
    assert out[0] is d1


def test_rrf_fuse_respects_top_n():
    docs = [_doc(f"d{i}") for i in range(5)]

    out = rag._rrf_fuse(docs[:2], docs[2:], top_n=2)

    assert len(out) == 2


def test_rrf_fuse_empty_inputs():
    assert rag._rrf_fuse([], []) == []


# ── 纯函数:结果格式化 ──


def test_format_docs_numbering_and_source_footnote():
    docs = [_doc("内容A", "手册.pdf"), _doc("内容B", "说明.docx")]

    out = rag._format_docs(docs)

    assert "[文档片段 1 — 来源: 手册.pdf]\n内容A" in out
    assert "[文档片段 2 — 来源: 说明.docx]\n内容B" in out


# ── 纯函数:docx 图片后缀推断 ──


@pytest.mark.parametrize(
    "content_type,blob,expected",
    [
        ("image/png", b"", ".png"),
        ("IMAGE/JPEG ", b"", ".jpg"),
        ("image/webp", b"", ".webp"),
        ("", b"\x89PNG\r\n\x1a\nxxxx", ".png"),
        ("", b"\xff\xd8\xffrest", ".jpg"),
        ("", b"GIF89a...", ".gif"),
        ("", b"RIFF\x00\x00\x00\x00WEBP", ".webp"),
        ("", b"II*\x00rest", ".tiff"),
        ("", b"BMrest", ".bmp"),
        ("", b"\x00\x01\x02", ".png"),  # 无法识别 → 默认 png
        ("image/jpeg", b"\x89PNG\r\n\x1a\n", ".jpg"),  # content_type 优先于魔数
    ],
)
def test_guess_image_ext(content_type, blob, expected):
    assert rag._guess_image_ext(content_type, blob) == expected


# ── 文件加载:txt(真实读文件,零服务) ──


def test_load_txt_reads_utf8(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("第一行\n第二行", encoding="utf-8")

    sections = rag._load_txt(str(f))

    # loader 现在返回结构单元（Section 协议）而不是 Document —— 切几级、
    # 怎么切交给 rag/structure.py，文件类型与切分策略解耦
    assert len(sections) == 1
    assert sections[0].text == "第一行\n第二行"
    assert sections[0].section_path == []  # 无标题 → 无路径
    assert sections[0].atomic is False
    assert sections[0].kind == "text"


def test_load_txt_blank_returns_empty(tmp_path):
    f = tmp_path / "blank.txt"
    f.write_text("  \n ", encoding="utf-8")

    assert rag._load_txt(str(f)) == []


def test_load_file_dispatches_by_suffix(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "_load_pdf", lambda path: [_section("来自pdf")])
    monkeypatch.setattr(rag, "_load_docx", lambda path: [_section("来自docx")])

    md = tmp_path / "note.md"
    md.write_text("hello", encoding="utf-8")

    assert rag._load_file(str(md))[0].text == "hello"  # .md 走 txt 读取
    assert rag._load_file(str(tmp_path / "x.pdf"))[0].text == "来自pdf"
    assert rag._load_file(str(tmp_path / "x.docx"))[0].text == "来自docx"
    assert rag._load_file(str(tmp_path / "x.unknown")) == []


# ── get_status:把模块内 Chroma 换成替身 ──


def test_get_status_returns_failed_dict_when_chroma_raises(monkeypatch):
    class _BoomChroma:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("connection refused")

    monkeypatch.setattr(rag, "Chroma", _BoomChroma)

    status = rag.get_status()

    assert status["initialized"] is False
    assert status["document_count"] == 0
    assert status["knowledge_base_dir"] == str(rag.KNOWLEDGE_BASE_DIR)
    assert "connection refused" in status["error"]


def test_get_status_returns_count_when_chroma_ok(monkeypatch):
    class _Col:
        def count(self):
            return 7

    class _FakeChroma:
        def __init__(self, *args, **kwargs):
            self._collection = _Col()

    monkeypatch.setattr(rag, "Chroma", _FakeChroma)

    status = rag.get_status()

    assert status["initialized"] is True
    assert status["document_count"] == 7
    assert status["knowledge_base_dir"] == str(rag.KNOWLEDGE_BASE_DIR)


# ── reordering:模块内 reranker 换成替身 ──


def test_reordering_empty_or_single_skips_reranker(monkeypatch):
    class _NeverCalled:
        def rerank(self, *a, **k):
            raise AssertionError("空/单文档不应触发重排模型")

        def rerank_scored(self, *a, **k):
            raise AssertionError("空/单文档不应触发重排模型")

    monkeypatch.setattr(rag, "reranker", _NeverCalled())

    assert rag.reordering("q", []) == []
    assert rag.reordering("q", [_doc("只有一条")]) == [_doc("只有一条")]


def test_reordering_uses_rerank_order_and_backfills_original_objects(monkeypatch):
    d1 = _doc("A", "a.txt", page=1)
    d2 = _doc("B", "b.txt", page=2)
    d3 = _doc("C", "c.txt")

    class _FakeReranker:
        def __init__(self):
            self.calls = []

        def rerank_scored(self, query, docs):
            self.calls.append((query, len(docs)))
            # 按 C,A,B 排，分数都远高于相对下限，不该被筛掉
            return [(docs[2], 0.9), (docs[0], 0.8), (docs[1], 0.7)]

    fake = _FakeReranker()
    monkeypatch.setattr(rag, "reranker", fake)

    out = rag.reordering("q", [d1, d2, d3])

    assert [d.page_content for d in out] == ["C", "A", "B"]
    # 回填的是原对象,元数据不丢
    assert out[1] is d1
    # 打分要覆盖**全部候选**：多样性筛选要在完整候选里挑，不能先截断再挑
    assert fake.calls == [("q", 3)]


def test_reordering_truncates_to_top_n(monkeypatch):
    # 每块来自不同文件：本用例只验证 TOP_N 截断，同源限流交给下一个用例
    docs = [_doc(f"d{i}", f"f{i}.txt") for i in range(6)]
    # 显式钉住 TOP_N：原来这个测试写死了 rag.TOP_N 的当前值，
    # TOP_N 从 4 调到 10 之后断言就自相矛盾了（len==TOP_N 与期望列表长度冲突）
    monkeypatch.setattr(rag, "TOP_N", 4)

    class _FakeReranker:
        def rerank_scored(self, query, docs):
            return [(d, 0.5) for d in reversed(docs)]

    monkeypatch.setattr(rag, "reranker", _FakeReranker())

    out = rag.reordering("q", docs)

    assert len(out) == rag.TOP_N
    assert [d.page_content for d in out] == ["d5", "d4", "d3", "d2"]


def test_select_diverse_caps_per_source_and_drops_low_scores():
    """同源限流 + 相对分数下限 —— 这次"上下文被同源近重复块灌满"的回归锁。

    真实事故：问"音箱连不上 WiFi 怎么排查？"时，同一个 xlsx 的 18 个近重复块
    分数挤在 0.60~0.674，把 TOP_N 全占满；真正写着排查原则的那块排第 19 名
    （0.092）永远进不了上下文 → 模型只能答"根据现有资料无法回答"。
    """
    same_file = [(Document(page_content=f"表{i}", metadata={"source": "排查.xlsx"}), 0.67 - i * 0.001) for i in range(8)]
    other = [(Document(page_content="排查原则", metadata={"source": "手册.md"}), 0.092)]
    junk = [(Document(page_content="无关", metadata={"source": "变更记录.md"}), 0.0)]

    kept = rag._select_diverse(same_file + other + junk, top_n=10)

    assert [d.page_content for d in kept] == ["表0", "表1", "表2", "排查原则"]  # 同源只留 3 块
    assert all(d.page_content != "无关" for d in kept)  # 低于 12% 的不再凑数


def test_select_diverse_keeps_everything_when_scores_are_unbounded():
    """最高分 <= 0（无界 logit 的 reranker）时不能启用相对下限，否则会把候选全砍光。"""
    pairs = [(Document(page_content=f"d{i}", metadata={"source": f"f{i}.txt"}), -0.5 - i) for i in range(4)]

    assert len(rag._select_diverse(pairs, top_n=10)) == 4


def test_reordering_degrades_to_original_order_on_error(monkeypatch):
    d1, d2 = _doc("A"), _doc("B")

    class _BadReranker:
        def rerank_scored(self, *a, **k):
            raise RuntimeError("模型加载失败")

    monkeypatch.setattr(rag, "reranker", _BadReranker())

    assert rag.reordering("q", [d1, d2]) == [d1, d2]


# ── retrieve / retrieve_sync:把向量库与稀疏路整体换成替身 ──


class _FakeCollection:
    def __init__(self, n):
        self._n = n

    def count(self):
        return self._n


class _FakeVectorStore:
    def __init__(self, count, search_results=None, raise_on_search=False):
        self._collection = _FakeCollection(count)
        self._results = search_results or []
        self._raise = raise_on_search

    def similarity_search(self, query, k):
        if self._raise:
            raise RuntimeError("embedding 服务不可用")
        return self._results


def test_retrieve_raises_when_knowledge_base_unavailable(monkeypatch):
    monkeypatch.setattr(rag, "_vectorstore", None)
    monkeypatch.setattr(rag, "_ensure_ready", lambda: None)  # 不让它真去连 Chroma

    with pytest.raises(rag.KnowledgeBaseError, match="连接失败"):
        asyncio.run(rag.retrieve("问题"))


def test_retrieve_raises_when_knowledge_base_empty(monkeypatch):
    monkeypatch.setattr(rag, "_vectorstore", _FakeVectorStore(count=0))

    with pytest.raises(rag.KnowledgeBaseError, match="为空"):
        asyncio.run(rag.retrieve("问题"))


def test_retrieve_fuses_dense_and_sparse(monkeypatch):
    d1 = _doc("苹果退货政策", "a.txt")
    d2 = _doc("运费险说明", "b.txt")
    d3 = _doc("保修期说明", "c.txt")
    store = _FakeVectorStore(count=3, search_results=[d1, d2])
    monkeypatch.setattr(rag, "_vectorstore", store)
    monkeypatch.setattr(rag, "_sparse_search", lambda query, k: [d2, d3])

    out = asyncio.run(rag.retrieve("运费", k=3))

    # d2 被两路同时命中(各 rank1)→ 分数最高排第一;d1 > d3(rank 更靠前)
    assert [d.page_content for d in out] == ["运费险说明", "苹果退货政策", "保修期说明"]


def test_retrieve_degrades_to_sparse_when_dense_fails(monkeypatch):
    d2 = _doc("运费险说明", "b.txt")
    d3 = _doc("保修期说明", "c.txt")
    store = _FakeVectorStore(count=3, raise_on_search=True)
    monkeypatch.setattr(rag, "_vectorstore", store)
    monkeypatch.setattr(rag, "_sparse_search", lambda query, k: [d2, d3])

    out = asyncio.run(rag.retrieve("运费", k=2))

    assert [d.page_content for d in out] == ["运费险说明", "保修期说明"]


def test_retrieve_sync_wraps_retrieve(monkeypatch):
    async def _fake_retrieve(query, k=rag.TOP_K):
        return [_doc("同步检索结果")]

    monkeypatch.setattr(rag, "retrieve", _fake_retrieve)

    assert [d.page_content for d in rag.retrieve_sync("问题")] == ["同步检索结果"]
