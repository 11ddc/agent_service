"""知识库管理接口测试（A4 的审核发布侧）—— 完全离线：替掉文档存储与向量库。

守的是"审核发布"这条链路的契约：

- **角色**：普通 user 不能碰知识库管理（403），kb_admin 才行；
- **404/503 分清**：文档不存在 → 404；存储不可用 → 503（不是 404，也不是 200）；
- **发布/下架要同步到向量库**：接口必须返回 `chunks_updated`，
  否则运维无法判断"发布到底生效了没有"；
- **向量库失败必须明说**：返回 502 并提示重试，而不是假装成功；
- **可见性取值受限**：拼错的值 → 422（不能写进 metadata 变成"认不出的可见性"）。
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.kb as kb_api
from db.document_store import DocumentRow
from db.mysql import MySQLUnavailable
from rag.rag import KnowledgeBaseError


def _row(**over) -> DocumentRow:
    base = dict(
        doc_id="d_1", filename="手册.pdf", source="/kb/手册.pdf", uploaded_at=None,
        parsed_status="ok", error_msg=None, chunk_count=7, parent_count=2,
        chunk_schema_ver="v3", tenant_id="default", owner_id="u_1",
        visibility="tenant", status="draft", published_at=None, published_by=None,
    )
    base.update(over)
    return DocumentRow(**base)


class FakeDocStore:
    """内存版文档存储；用类属性保存状态，方便在用例里预置与断言。"""

    rows: dict[str, DocumentRow] = {}
    fail = False
    calls: list = []

    def __init__(self):
        pass

    def _guard(self):
        if FakeDocStore.fail:
            raise MySQLUnavailable("模拟：文档存储不可用")

    def get(self, doc_id):
        self._guard()
        return FakeDocStore.rows.get(doc_id)

    def list_by_status(self, status=None, tenant_id=None, limit=200):
        self._guard()
        return [
            r for r in FakeDocStore.rows.values()
            if (not status or r.status == status) and (not tenant_id or r.tenant_id == tenant_id)
        ]

    def set_status(self, doc_id, status, published_by=None):
        self._guard()
        FakeDocStore.calls.append(("set_status", doc_id, status, published_by))
        row = FakeDocStore.rows[doc_id]
        FakeDocStore.rows[doc_id] = _row(**{**row.__dict__, "status": status,
                                           "published_by": published_by})
        return 1

    def set_visibility(self, doc_id, visibility):
        self._guard()
        FakeDocStore.calls.append(("set_visibility", doc_id, visibility))
        row = FakeDocStore.rows[doc_id]
        FakeDocStore.rows[doc_id] = _row(**{**row.__dict__, "visibility": visibility})
        return 1


@pytest.fixture
def kb(monkeypatch):
    """最小 app + 替身存储；返回 (client, 状态容器)。"""
    FakeDocStore.rows = {"d_1": _row()}
    FakeDocStore.fail = False
    FakeDocStore.calls = []
    state = {"chunks": 7, "sync_error": None, "audit": []}

    def _fake_update(source, **fields):
        if state["sync_error"]:
            raise state["sync_error"]
        state.setdefault("synced", []).append((source, fields))
        return state["chunks"]

    async def _fake_audit(action, result, **kwargs):
        state["audit"].append((action, result))

    monkeypatch.setattr(kb_api, "DocumentStore", FakeDocStore)
    monkeypatch.setattr(kb_api, "update_document_acl", _fake_update)
    monkeypatch.setattr(kb_api.audit, "record", _fake_audit)
    return state


@pytest.fixture
def client(kb, as_role) -> TestClient:
    app = FastAPI()
    app.include_router(kb_api.router, prefix="/api/kb")
    as_role(app, role="kb_admin", user_id="u_admin", tenant_id="default")
    return TestClient(app)


@pytest.fixture
def user_client(kb, as_role) -> TestClient:
    app = FastAPI()
    app.include_router(kb_api.router, prefix="/api/kb")
    as_role(app, role="user")
    return TestClient(app)


# ── 角色边界 ─────────────────────────────────────────────────
def test_normal_user_cannot_manage_the_knowledge_base(user_client):
    assert user_client.get("/api/kb/documents").status_code == 403
    assert user_client.post("/api/kb/documents/d_1/publish").status_code == 403
    assert user_client.post("/api/kb/documents/d_1/archive").status_code == 403


def test_operator_can_view_but_not_publish(kb, as_role):
    app = FastAPI()
    app.include_router(kb_api.router, prefix="/api/kb")
    as_role(app, role="operator")
    c = TestClient(app)

    assert c.get("/api/kb/documents").status_code == 200
    assert c.post("/api/kb/documents/d_1/publish").status_code == 403


# ── 清单 ────────────────────────────────────────────────────
def test_list_documents(client):
    r = client.get("/api/kb/documents")

    assert r.status_code == 200
    assert [d["doc_id"] for d in r.json()] == ["d_1"]
    assert r.json()[0]["status"] == "draft"


def test_list_documents_filters_by_status(client):
    assert client.get("/api/kb/documents", params={"status_filter": "published"}).json() == []


def test_list_returns_503_when_store_is_down(client, kb):
    FakeDocStore.fail = True

    assert client.get("/api/kb/documents").status_code == 503


# ── 发布 ────────────────────────────────────────────────────
def test_publish_updates_both_stores(client, kb):
    r = client.post("/api/kb/documents/d_1/publish")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "published"
    assert body["chunks_updated"] == 7
    assert ("set_status", "d_1", "published", "u_admin") in FakeDocStore.calls
    assert kb["synced"] == [("/kb/手册.pdf", {"status": "published"})]
    assert ("kb.publish", "ok") in kb["audit"]


def test_publish_unknown_document_is_404(client):
    assert client.post("/api/kb/documents/nope/publish").status_code == 404


def test_publish_reports_502_when_vector_store_fails(client, kb):
    """向量库没更新 = 发布没真正生效，必须明说并让调用方重试，不能假装成功。"""
    kb["sync_error"] = KnowledgeBaseError("模拟：向量库连不上")

    r = client.post("/api/kb/documents/d_1/publish")

    assert r.status_code == 502
    assert "重试" in r.text


def test_publish_returns_503_when_store_is_down(client, kb):
    FakeDocStore.fail = True

    assert client.post("/api/kb/documents/d_1/publish").status_code == 503


def test_publish_warns_when_the_document_has_no_chunks(client, kb):
    """解析失败过的文档没有块 —— 发布成功但依然检索不到，必须提醒。"""
    kb["chunks"] = 0

    body = client.post("/api/kb/documents/d_1/publish").json()

    assert body["chunks_updated"] == 0
    assert "向量库里没有块" in body.get("warning", "")


# ── 下架 ────────────────────────────────────────────────────
def test_archive_updates_both_stores(client, kb):
    r = client.post("/api/kb/documents/d_1/archive")

    assert r.status_code == 200
    assert r.json()["status"] == "archived"
    assert kb["synced"] == [("/kb/手册.pdf", {"status": "archived"})]
    assert ("kb.archive", "ok") in kb["audit"]


def test_archive_unknown_document_is_404(client):
    assert client.post("/api/kb/documents/nope/archive").status_code == 404


# ── 可见性 ──────────────────────────────────────────────────
@pytest.mark.parametrize("visibility", ["tenant", "private", "public"])
def test_set_valid_visibility(client, kb, visibility):
    r = client.post("/api/kb/documents/d_1/visibility", json={"visibility": visibility})

    assert r.status_code == 200, r.text
    assert r.json()["visibility"] == visibility
    assert kb["synced"] == [("/kb/手册.pdf", {"visibility": visibility})]


@pytest.mark.parametrize("visibility", ["everyone", "", "  "])
def test_set_invalid_visibility_is_422(client, kb, visibility):
    r = client.post("/api/kb/documents/d_1/visibility", json={"visibility": visibility})

    assert r.status_code == 422, f"{visibility!r} 应当被拒"
    assert not kb.get("synced"), "非法值不该被写进向量库"


@pytest.mark.parametrize(
    "given,expected", [("TENANT ", "tenant"), ("Private", "private"), (" public", "public")]
)
def test_visibility_input_is_normalised(client, kb, given, expected):
    """大小写与首尾空格要容忍（但值本身必须是白名单内的）。"""
    r = client.post("/api/kb/documents/d_1/visibility", json={"visibility": given})

    assert r.status_code == 200, r.text
    assert r.json()["visibility"] == expected
    assert kb["synced"] == [("/kb/手册.pdf", {"visibility": expected})]


def test_set_visibility_unknown_document_is_404(client):
    r = client.post("/api/kb/documents/nope/visibility", json={"visibility": "public"})

    assert r.status_code == 404
