"""探针与指标接口测试（A5）—— 完全离线：替掉 MySQL/Redis 探针。

守的性质：

1. **探针不被鉴权挡住**（k8s 的探针不会带令牌），但 `/metrics` 默认要 admin；
2. **/health 暴露危险开关**：生产里 AUTH/ACL/限流 被关掉必须能被监控发现；
3. **/ready 的判定口径**：MySQL 不可用且认证开启 → 503（没人能登录）；
   Redis 不可用 → 仍就绪（只是降级）；向量库未初始化 → 仍就绪，且
   **不能因为探针去触发模型加载**（那会让探针变贵、变脆）。
"""

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import api.system as system_api
import config
from auth.deps import Principal, Role
from metrics import REGISTRY


class _FakeAuth:
    """替掉 require_user：role=None 表示没带令牌。"""

    def __init__(self, role: str | None):
        self.role = role

    async def __call__(self, credentials=None):
        if self.role is None:
            raise HTTPException(status_code=401, detail="缺少访问令牌")
        return Principal(
            user_id="u_1", username="tester", role=Role(self.role), tenant_id="default"
        )


@pytest.fixture
def client(monkeypatch) -> TestClient:
    app = FastAPI()
    app.include_router(system_api.router)

    # 探针依赖：默认都是健康的，用例里按需覆盖
    monkeypatch.setattr("db.mysql.ping", lambda: (True, "ok (MySQL 8.0.34)"))

    class _Redis:
        async def ping(self):
            return True

    monkeypatch.setattr("redis_client.redis_client", _Redis())
    monkeypatch.setattr(system_api, "require_user", _FakeAuth("admin"))
    return TestClient(app)


# ── /health ─────────────────────────────────────────────────
def test_health_is_ok_and_exposes_switches(client):
    r = client.get("/health")

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"] == config.APP_VERSION
    assert set(body["switches"]) == {
        "auth_enabled",
        "acl_enabled",
        "rate_limit_enabled",
        "metrics_require_auth",
    }


def test_health_warns_when_a_safety_switch_is_off(client, monkeypatch):
    """关掉鉴权/ACL/限流必须产生 warnings —— 让监控能直接告警。"""
    monkeypatch.setattr(config, "AUTH_ENABLED", False)
    monkeypatch.setattr(config, "ACL_ENABLED", False)

    body = client.get("/health").json()

    assert body["warnings"], "危险开关被关掉时必须告警"
    assert "auth_enabled" in body["warnings"][0]
    assert "acl_enabled" in body["warnings"][0]


def test_health_needs_no_authentication(client, monkeypatch):
    """探针不会带令牌：这条接口在任何角色下都必须是 200。"""
    monkeypatch.setattr(system_api, "require_user", _FakeAuth(None))

    assert client.get("/health").status_code == 200


# ── /ready ──────────────────────────────────────────────────
def test_ready_when_everything_is_healthy(client):
    body = client.get("/ready").json()

    assert body["ready"] is True
    assert body["checks"]["mysql"]["ok"] is True
    assert body["checks"]["redis"]["ok"] is True
    assert body["checks"]["vector_store"]["detail"] == "not_initialized"


def test_ready_is_503_when_mysql_is_down_and_auth_is_on(client, monkeypatch):
    monkeypatch.setattr("db.mysql.ping", lambda: (False, "连接被拒绝"))
    monkeypatch.setattr(config, "AUTH_ENABLED", True)

    r = client.get("/ready")

    assert r.status_code == 503
    body = r.json()
    assert body["ready"] is False
    assert body["blocker"] and "MySQL" in body["blocker"]


def test_ready_stays_200_when_mysql_is_down_but_auth_is_off(client, monkeypatch):
    """本地演示模式：不认证就用不到账号库，MySQL 挂了也能服务。"""
    monkeypatch.setattr("db.mysql.ping", lambda: (False, "连接被拒绝"))
    monkeypatch.setattr(config, "AUTH_ENABLED", False)

    r = client.get("/ready")

    assert r.status_code == 200
    assert r.json()["ready"] is True


def test_redis_failure_does_not_block_readiness(client, monkeypatch):
    """Redis 只是会话历史/计数窗口，不是问答主链路的必需项。"""

    class _Broken:
        async def ping(self):
            raise ConnectionError("redis 连不上")

    monkeypatch.setattr("redis_client.redis_client", _Broken())

    r = client.get("/ready")

    assert r.status_code == 200
    assert r.json()["checks"]["redis"]["ok"] is False
    assert "ConnectionError" in r.json()["checks"]["redis"]["detail"]


def test_ready_never_loads_the_vector_store(client, monkeypatch):
    """探针必须便宜：未初始化时只报 not_initialized，不能触发模型加载。"""
    import rag.rag as rag_module

    monkeypatch.setattr(rag_module, "_vectorstore", None)

    body = client.get("/ready").json()

    assert body["checks"]["vector_store"] == {"ok": True, "detail": "not_initialized"}


def test_ready_reports_chunk_count_when_initialized(client, monkeypatch):
    import rag.rag as rag_module

    class _Col:
        def count(self):
            return 1818

    monkeypatch.setattr(
        rag_module, "_vectorstore", type("S", (), {"_collection": _Col()})()
    )

    body = client.get("/ready").json()

    assert body["checks"]["vector_store"]["detail"] == "chunks=1818"


def test_ready_survives_a_broken_probe(client, monkeypatch):
    """探针自己绝对不能抛：它挂了会被判成"实例不健康"，反而引发重启风暴。"""
    def _boom():
        raise RuntimeError("驱动炸了")

    monkeypatch.setattr("db.mysql.ping", _boom)

    r = client.get("/ready")

    assert r.status_code in (200, 503)
    assert r.json()["checks"]["mysql"]["ok"] is False


# ── /metrics ────────────────────────────────────────────────
def test_metrics_requires_a_token_by_default(client, monkeypatch):
    monkeypatch.setattr(config, "METRICS_REQUIRE_AUTH", True)
    monkeypatch.setattr(system_api, "require_user", _FakeAuth(None))

    assert client.get("/metrics").status_code == 401


def test_metrics_rejects_non_admin(client, monkeypatch):
    monkeypatch.setattr(config, "METRICS_REQUIRE_AUTH", True)
    monkeypatch.setattr(system_api, "require_user", _FakeAuth("user"))

    r = client.get("/metrics")

    assert r.status_code == 403
    assert "admin" in r.json()["detail"]


def test_metrics_returns_prometheus_text_for_admin(client, monkeypatch):
    monkeypatch.setattr(config, "METRICS_REQUIRE_AUTH", True)
    REGISTRY.counter("test_sys_counter_total", "测试").inc()

    r = client.get("/metrics")

    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert "# TYPE test_sys_counter_total counter" in r.text
    assert "test_sys_counter_total 1" in r.text


def test_metrics_can_be_opened_for_in_cluster_scraping(client, monkeypatch):
    """集群内抓取可以关掉鉴权（此时务必只在内部网暴露）。"""
    monkeypatch.setattr(config, "METRICS_REQUIRE_AUTH", False)

    assert client.get("/metrics").status_code == 200
