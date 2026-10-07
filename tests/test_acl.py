"""文档 ACL 与检索过滤测试（A4）—— 完全离线：不连 Chroma / MySQL，不出网。

这一层守的是**权限边界**，不是复述实现。核心性质有三条：

1. **两条通道的语义必须一致**：dense 走 `chroma_filter()`，sparse 走 `allows()`。
   只要两者对同一份 metadata 给出不同答案，就会出现"从 BM25 那一侧漏出去"——
   而这是 ACL 最容易漏的一处（BM25 对**全量语料**打分，不经过 Chroma）。
   下面用一段独立的 where 求值器把两者对齐着测。
2. **失败方向是拒绝**：草稿、下架、认不出的可见性、缺元数据 → 都不可见。
3. **过滤器形状合法**：Chroma 不接受 `$and` 里套 `$and`（实测会抛 ValueError），
   所以合并过滤器必须展开一层。
"""

import pytest
from langchain_core.documents import Document
from types import SimpleNamespace

import config
import rag.rag as rag_module
from db.document_store import DocumentRow
from rag import acl as rag_acl

# ══════════════════════════════════════════════════════════════
# 工具：把 Chroma 的 where 表达式在 Python 里求值，用来与 allows() 对齐
# ══════════════════════════════════════════════════════════════


def _eval_where(expr: dict, meta: dict) -> bool:
    """最小 where 求值器（只支持本项目用到的 $and/$or/$eq）。"""
    if "$and" in expr:
        return all(_eval_where(e, meta) for e in expr["$and"])
    if "$or" in expr:
        return any(_eval_where(e, meta) for e in expr["$or"])
    (field, cond), = expr.items()
    (op, value), = cond.items()
    if op == "$eq":
        return str(meta.get(field) or "") == str(value)
    raise AssertionError(f"求值器没实现的操作符: {op}")


def _meta(**over) -> dict:
    base = {
        "status": rag_acl.STATUS_PUBLISHED,
        "visibility": rag_acl.VISIBILITY_TENANT,
        "tenant_id": "default",
        "owner_id": "u_1",
    }
    base.update(over)
    return base


CASES = [
    # (说明, metadata, 是否应当可见)
    ("本租户已发布", _meta(), True),
    ("草稿", _meta(status="draft"), False),
    ("已下架", _meta(status="archived"), False),
    ("缺 status（历史块未回填）", {k: v for k, v in _meta().items() if k != "status"}, False),
    ("别的租户的 tenant 可见文档", _meta(tenant_id="other"), False),
    ("public 可见（跨租户）", _meta(visibility="public", tenant_id="other"), True),
    ("private 且是自己", _meta(visibility="private", owner_id="u_1"), True),
    ("private 但是别人的", _meta(visibility="private", owner_id="u_2"), False),
    ("认不出的可见性", _meta(visibility="weird"), False),
    ("缺 visibility", {k: v for k, v in _meta().items() if k != "visibility"}, False),
]


# ══════════════════════════════════════════════════════════════
# 1. 两条通道语义一致
# ══════════════════════════════════════════════════════════════
@pytest.mark.parametrize("desc,meta,expected", CASES)
def test_allows_matches_expected(desc, meta, expected):
    acl = rag_acl.Acl(tenant_id="default", user_id="u_1")

    assert acl.allows(meta) is expected, desc


@pytest.mark.parametrize("desc,meta,expected", CASES)
def test_chroma_filter_agrees_with_allows(desc, meta, expected):
    """⚠️ 关键：dense 的 where 与 sparse 的判据对同一份数据必须给出同一答案。

    两者只要有一处不一致，就会出现"某条通道漏出受限文档"——
    而且不会有任何报错，只是答案里混进了不该出现的内容。
    """
    acl = rag_acl.Acl(tenant_id="default", user_id="u_1")

    assert _eval_where(acl.chroma_filter(), meta) is expected, desc


def test_different_tenant_cannot_see_tenant_scoped_docs():
    acl = rag_acl.Acl(tenant_id="other", user_id="u_1")
    meta = _meta()

    assert acl.allows(meta) is False
    assert _eval_where(acl.chroma_filter(), meta) is False


def test_acl_disabled_turns_filtering_off():
    """ACL 关闭 = 退回不过滤（只应出现在本地演示，且 /health 会暴露它）。"""
    acl = rag_acl.Acl(tenant_id="other", user_id="u_1")

    config.ACL_ENABLED = False
    try:
        assert acl.chroma_filter() is None
        assert acl.allows(_meta(status="draft")) is True
    finally:
        config.ACL_ENABLED = True


# ══════════════════════════════════════════════════════════════
# 2. 过滤器形状
# ══════════════════════════════════════════════════════════════
def _nested_and(expr) -> bool:
    """是否存在 `$and` 里直接套 `$and`（Chroma 会拒绝）。"""
    if not isinstance(expr, dict):
        return False
    if "$and" in expr:
        return any(
            ("$and" in child) or _nested_and(child) for child in expr["$and"]
        )
    if "$or" in expr:
        return any(_nested_and(child) for child in expr["$or"])
    return False


def test_acl_filter_shape_is_accepted_by_chroma():
    """`$and` 里不能套 `$and`（实测 Chroma 会抛 ValueError）。"""
    assert not _nested_and(rag_acl.Acl().chroma_filter())


def test_merge_filters_flattens_nested_and():
    """合并"限定文档 + ACL"时必须展开一层，否则形状非法。"""
    acl_filter = rag_acl.Acl(tenant_id="default", user_id="u_1").chroma_filter()

    merged = rag_acl.merge_filters({"source": {"$eq": "a.pdf"}}, acl_filter)

    assert not _nested_and(merged), "合并后出现了 $and 套 $and"
    assert merged["$and"][0] == {"source": {"$eq": "a.pdf"}}
    # 语义不能变：展开后 ACL 部分仍然与 allows 一致（补上 source 以满足文档条件）
    for _desc, meta, expected in CASES:
        assert _eval_where(merged, {**meta, "source": "a.pdf"}) is expected, _desc
    # 且确实限定了文档：别的 source 不匹配
    assert _eval_where(merged, {**_meta(), "source": "b.pdf"}) is False


def test_merge_filters_handles_empty_and_single():
    acl_filter = rag_acl.Acl().chroma_filter()

    assert rag_acl.merge_filters(None, None) is None
    assert rag_acl.merge_filters({"a": {"$eq": 1}}) == {"a": {"$eq": 1}}
    assert rag_acl.merge_filters(None, {"a": {"$eq": 1}}) == {"a": {"$eq": 1}}
    assert rag_acl.merge_filters({}, acl_filter) == acl_filter


# ══════════════════════════════════════════════════════════════
# 3. acl_metadata 校验
# ══════════════════════════════════════════════════════════════
def test_acl_metadata_builds_the_four_fields():
    acl = rag_acl.Acl(tenant_id="t1", user_id="u9")

    meta = rag_acl.acl_metadata(
        acl, status=rag_acl.STATUS_DRAFT, visibility=rag_acl.VISIBILITY_PRIVATE
    )

    assert meta == {
        "tenant_id": "t1",
        "owner_id": "u9",
        "visibility": "private",
        "status": "draft",
    }


@pytest.mark.parametrize("field,bad", [("visibility", "everyone"), ("status", "live")])
def test_acl_metadata_rejects_unknown_values(field, bad):
    """拼错的值必须报错，而不是写进去变成"认不出的可见性"（那会静默变成不可见）。"""
    kwargs = {"status": rag_acl.STATUS_DRAFT, "visibility": rag_acl.VISIBILITY_TENANT}
    kwargs[field] = bad

    with pytest.raises(ValueError):
        rag_acl.acl_metadata(rag_acl.Acl(), **kwargs)


# ══════════════════════════════════════════════════════════════
# 4. ContextVar 传递（工具函数签名由 LLM 决定，只能靠上下文）
# ══════════════════════════════════════════════════════════════
def test_current_acl_defaults_to_system():
    assert rag_acl.current_acl().user_id == rag_acl.SYSTEM_USER_ID


def test_set_and_reset_acl_roundtrip():
    original = rag_acl.current_acl()
    token = rag_acl.set_acl(rag_acl.Acl(tenant_id="t9", user_id="u9"))

    assert rag_acl.current_acl().tenant_id == "t9"

    rag_acl.reset_acl(token)
    assert rag_acl.current_acl() == original


def test_acl_of_principal():
    principal = SimpleNamespace(user_id="u7", tenant_id="t7", role=SimpleNamespace(value="kb_admin"))

    acl = rag_acl.Acl.of(principal)

    assert (acl.tenant_id, acl.user_id, acl.role) == ("t7", "u7", "kb_admin")


# ══════════════════════════════════════════════════════════════
# 5. sparse 通道真的过滤了（BM25 对全量语料打分，最容易漏）
# ══════════════════════════════════════════════════════════════
class _FakeBM25:
    def __init__(self, scores):
        self._scores = scores

    def get_scores(self, _tokens):
        return list(self._scores)


class _FakeCollection:
    def __init__(self, count):
        self._count = count

    def count(self):
        return self._count


def _install_sparse(monkeypatch, docs, scores):
    monkeypatch.setattr(rag_module, "_vectorstore", SimpleNamespace(_collection=_FakeCollection(len(docs))))
    monkeypatch.setattr(rag_module, "_chunk_docs", docs)
    monkeypatch.setattr(rag_module, "_bm25_corpus", [["x"]] * len(docs))
    monkeypatch.setattr(rag_module, "_bm25", _FakeBM25(scores))


def test_sparse_search_excludes_documents_outside_the_acl(monkeypatch):
    """BM25 打分覆盖全量语料 —— 必须在返回前按 ACL 过滤。"""
    docs = [
        Document(page_content="本租户已发布", metadata=_meta()),
        Document(page_content="草稿", metadata=_meta(status="draft")),
        Document(page_content="别的租户", metadata=_meta(tenant_id="other")),
        Document(page_content="公开", metadata=_meta(visibility="public", tenant_id="other")),
    ]
    _install_sparse(monkeypatch, docs, [5.0, 9.0, 8.0, 1.0])  # 草稿分最高

    token = rag_acl.set_acl(rag_acl.Acl(tenant_id="default", user_id="u_1"))
    try:
        got = rag_module._sparse_search("q", 10)
    finally:
        rag_acl.reset_acl(token)

    contents = {d.page_content for d in got}
    assert contents == {"本租户已发布", "公开"}, "草稿/别的租户不该被召回"
    # 高分但不可见的块不能挤掉可见块的位置
    assert got[0].page_content == "本租户已发布"


def test_sparse_search_takes_top_k_after_filtering(monkeypatch):
    """顺序很重要：先按 ACL 缩小候选再取 top-k，否则可见结果会被挤掉。"""
    docs = [
        Document(page_content="draft-a", metadata=_meta(status="draft")),
        Document(page_content="draft-b", metadata=_meta(status="draft")),
        Document(page_content="visible", metadata=_meta()),
    ]
    _install_sparse(monkeypatch, docs, [9.0, 8.0, 1.0])

    token = rag_acl.set_acl(rag_acl.Acl(tenant_id="default", user_id="u_1"))
    try:
        got = rag_module._sparse_search("q", 1)
    finally:
        rag_acl.reset_acl(token)

    assert [d.page_content for d in got] == ["visible"]


def test_sparse_search_returns_nothing_for_a_foreign_tenant(monkeypatch):
    docs = [Document(page_content="a", metadata=_meta())]
    _install_sparse(monkeypatch, docs, [5.0])

    token = rag_acl.set_acl(rag_acl.Acl(tenant_id="other", user_id="u_1"))
    try:
        got = rag_module._sparse_search("q", 10)
    finally:
        rag_acl.reset_acl(token)

    assert got == []


# ══════════════════════════════════════════════════════════════
# 6. dense 通道把过滤器下推给了 Chroma
# ══════════════════════════════════════════════════════════════
def test_retrieve_pushes_the_acl_filter_to_chroma(monkeypatch):
    import asyncio

    seen = {}

    class _RecordingVectorStore:
        _collection = _FakeCollection(3)

        def similarity_search(self, query, k, filter=None):
            seen["filter"] = filter
            return []

    monkeypatch.setattr(rag_module, "_vectorstore", _RecordingVectorStore())
    monkeypatch.setattr(rag_module, "_bm25", None)
    monkeypatch.setattr(rag_module, "_chunk_docs", [])

    token = rag_acl.set_acl(rag_acl.Acl(tenant_id="default", user_id="u_1"))
    try:
        asyncio.run(rag_module.retrieve("q", k=2))
    finally:
        rag_acl.reset_acl(token)

    assert seen["filter"] == rag_acl.Acl(tenant_id="default", user_id="u_1").chroma_filter()


def test_invalidate_index_cache_forces_bm25_rebuild(monkeypatch):
    """发布/下架只改 metadata、不改块数 —— 不显式失效就会用旧的 status。"""
    monkeypatch.setattr(rag_module, "_bm25", object())

    rag_module.invalidate_index_cache()

    assert rag_module._bm25 is None


# ══════════════════════════════════════════════════════════════
# 7. update_document_acl 合并而不是覆盖 metadata
# ══════════════════════════════════════════════════════════════
def test_update_document_acl_merges_and_invalidates(monkeypatch):
    stored = {}

    class _Col:
        def __init__(self):
            self.updated = None

        def update(self, ids, metadatas):
            self.updated = (list(ids), [dict(m) for m in metadatas])
            stored["ok"] = True

    col = _Col()

    class _VS:
        _collection = col

        def get(self, where=None):
            return {
                "ids": ["c1", "c2"],
                "metadatas": [
                    {"source": "a.pdf", "parent_id": "p1", "status": "draft"},
                    {"source": "a.pdf", "parent_id": "p2", "status": "draft"},
                ],
            }

    monkeypatch.setattr(rag_module, "_vectorstore", _VS())
    monkeypatch.setattr(rag_module, "_bm25", object())

    n = rag_module.update_document_acl("a.pdf", status="published")

    assert n == 2
    ids, metas = col.updated
    assert ids == ["c1", "c2"]
    # 关键：原有字段必须保留（否则引用 source / 父子扩展会一起失效）
    assert metas[0] == {
        "source": "a.pdf",
        "parent_id": "p1",
        "status": "published",
    }
    assert rag_module._bm25 is None, "元数据变了必须让 BM25 缓存失效"


def test_update_document_acl_with_no_chunks_returns_zero(monkeypatch):
    class _VS:
        _collection = SimpleNamespace()

        def get(self, where=None):
            return {"ids": [], "metadatas": []}

    monkeypatch.setattr(rag_module, "_vectorstore", _VS())

    assert rag_module.update_document_acl("missing.pdf", status="published") == 0


# ══════════════════════════════════════════════════════════════
# 8. 回填脚本的"存量识别"是纯函数
# ══════════════════════════════════════════════════════════════
def test_backfill_identifies_only_chunks_missing_acl():
    from rag import acl_backfill

    class _Col:
        def __init__(self):
            self._metas = [
                {"source": "old.pdf"},  # 缺 ACL
                {"source": "old.pdf"},  # 缺 ACL
                {"source": "new.pdf", "status": "published", "visibility": "tenant",
                 "tenant_id": "default"},  # 已有 ACL
                {"source": "half.pdf", "status": "published"},  # 只缺一部分也算存量
            ]

        def count(self):
            return len(self._metas)

        def get(self, limit=0, offset=0, include=None):
            chunk = self._metas[offset : offset + limit]
            return {"ids": [f"c{i}" for i in range(len(chunk))], "metadatas": chunk}

    legacy = acl_backfill._legacy_sources(_Col())

    assert legacy == {"old.pdf": 2, "half.pdf": 1}


def test_document_row_keeps_backward_compatible_defaults():
    """DocumentRow 新增了 ACL 字段，但带默认值 —— 既有按位置构造的调用不受影响。"""
    row = DocumentRow("d1", "a.pdf", "/x/a.pdf", None, "ok", None, 3, 1, "v3")

    assert (row.tenant_id, row.visibility, row.status) == ("default", "tenant", "draft")
