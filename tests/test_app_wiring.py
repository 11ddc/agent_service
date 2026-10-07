"""整机接线冒烟测试 —— 用**真实 app**（`main.app`），但仍零外部服务。

补这个文件的原因：以前没有任何测试 import 过 `main.py`，于是下面这些"接线"
错误可以一路绿着上线：

- `router.register_routers` 里的 prefix 写错 → 接口 404，但单测全过；
- CORS 与请求体上限的**中间件顺序**错了 → 413 响应不带 CORS 头，
  前端看到的是 CORS 报错而不是"文件太大"；
- 新增的请求体上限根本没挂上 → 超限请求继续被完整缓冲。

注意：这里只发**根本不会走到业务逻辑**的请求（超限体、被拒的文件名），
所以不需要 Chroma / MySQL / Redis / 任何模型。
"""
import pytest
from fastapi.testclient import TestClient

import body_limit
import main
from body_limit import BodyLimitMiddleware


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(main.app)


# ── 1. 路由真的挂上了 ───────────────────────────────────────
def test_chat_routes_are_mounted(client):
    """只验路由存在（401/422 都行），不发真实问答请求。"""
    resp = client.post("/api/chat", json={})

    assert resp.status_code != 404, "POST /api/chat 没挂上"


def test_stream_route_is_mounted(client):
    resp = client.post("/api/chat/stream", json={})

    assert resp.status_code != 404, "POST /api/chat/stream 没挂上"


def test_upload_route_is_mounted(client):
    """不带文件 → 422（参数校验失败），但绝不能是 404。"""
    resp = client.post("/api/upload")

    assert resp.status_code != 404, "POST /api/upload 没挂上"


# ── 2. 请求体上限真的挂在 app 上，而且 CORS 在更外层 ──────────
def test_body_limit_middleware_is_registered():
    """光有 body_limit.py 不够 —— 必须真的挂到 app 上。"""
    assert any(
        m.cls is BodyLimitMiddleware for m in main.app.user_middleware
    ), "请求体上限没有挂在 app 上"


def test_oversized_body_is_rejected_and_keeps_cors_headers(client):
    """带上真实 app：超限体必须在 multipart 解析前拿到 413，且带 CORS 头。

    CORS 头这条守的是中间件**顺序**：Starlette 里后加的在最外层，所以 CORS
    必须比请求体上限后加 —— 否则"按 Content-Length 直接拒掉"的响应绕过 CORS，
    前端看到的会是 CORS 报错而不是"文件太大"。
    """
    oversized = b"x" * (body_limit.MAX_BODY_BYTES + 1024)

    resp = client.post(
        "/api/upload",
        content=oversized,
        headers={
            "content-type": "multipart/form-data; boundary=----x",
            "origin": "http://localhost:5173",
        },
    )

    assert resp.status_code == 413, resp.text
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_normal_json_body_is_not_affected_by_the_limit(client):
    """上限不能误伤正常请求：缺字段的空请求应当正常进入参数校验（422）。"""
    resp = client.post("/api/chat", json={})

    assert resp.status_code == 422, resp.text
