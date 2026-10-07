"""请求体上限（`body_limit.py`）回归测试 —— 零服务：不起网络、不调模型。

守的是一条**顺序**不变量：超限的请求必须在 multipart 解析**之前**被拒。

为什么这条顺序重要：Starlette 对**文件字段没有任何大小上限**
（`max_part_size` 只作用于非文件字段，文件部分走 SpooledTemporaryFile），
所以守在上传接口里的 `file.size > MAX_UPLOAD_BYTES` 是**事后**检查 ——
一个 10GB 的请求会先被完整缓冲到内存/临时盘，然后才拿到 413。
"""
import asyncio

import pytest
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse

import body_limit
from body_limit import BodyLimitMiddleware, RequestBodyTooLarge


def _client(limit: int, reached: list | None = None):
    """最小 app：只挂中间件和一个回显端点。"""
    from fastapi.testclient import TestClient

    app = FastAPI()
    reached = reached if reached is not None else []

    @app.post("/echo")
    async def echo(request: Request):
        reached.append(True)
        return {"size": len(await request.body())}

    async def _handle(request: Request, exc: RequestBodyTooLarge):
        return JSONResponse(status_code=413, content={"detail": "too large"})

    app.add_exception_handler(RequestBodyTooLarge, _handle)
    app.add_middleware(BodyLimitMiddleware, max_body_bytes=limit)
    return TestClient(app)


# ── 1. 正常请求不受影响 ─────────────────────────────────────
def test_body_within_limit_passes():
    reached = []
    resp = _client(limit=1024, reached=reached).post("/echo", content=b"x" * 100)

    assert resp.status_code == 200
    assert resp.json() == {"size": 100}
    assert reached == [True]


# ── 2. Content-Length 超限：一个字节都不读 ───────────────────
def test_content_length_over_limit_is_rejected_before_the_app_runs():
    reached = []
    resp = _client(limit=64, reached=reached).post("/echo", content=b"x" * 4096)

    assert resp.status_code == 413
    assert reached == [], "超限请求不该进到业务处理里"


# ── 3. 没有 Content-Length（chunked）：靠 receive 计数掐断 ────
def test_streaming_body_is_cut_off_without_content_length():
    """直接对着 ASGI 通道测：把 receive 换成"分块推送"，第二块越界必须抛错。

    这条路径是 chunked 请求的兜底 —— 没有它，`Content-Length` 缺失就等于没有上限。
    """
    consumed = {"bytes": 0}

    async def app(scope, receive, send):
        while True:
            msg = await receive()
            if msg["type"] != "http.request":
                break
            consumed["bytes"] += len(msg.get("body") or b"")
            if not msg.get("more_body"):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    middleware = BodyLimitMiddleware(app, max_body_bytes=10)
    scope = {"type": "http", "method": "POST", "path": "/echo", "headers": []}
    chunks = [b"x" * 8, b"x" * 8]

    async def receive():
        body = chunks.pop(0)
        return {"type": "http.request", "body": body, "more_body": bool(chunks)}

    async def send(message):
        pass

    with pytest.raises(RequestBodyTooLarge):
        asyncio.run(middleware(scope, receive, send))

    # 越界的那一块**不会**被交给应用：计数值停在前一块（8 字节），
    # 第二次 receive 直接抛错 —— 应用拿不到超限数据。
    assert consumed["bytes"] == 8


def test_non_http_scope_passes_through():
    """lifespan / websocket 这些非 http scope 必须原样放行。"""
    seen = []

    async def app(scope, receive, send):
        seen.append(scope["type"])

    asyncio.run(BodyLimitMiddleware(app, max_body_bytes=1)({"type": "lifespan"}, None, None))

    assert seen == ["lifespan"]


# ── 4. 上限常量之间的关系 ────────────────────────────────────
def _client_with_file_route(limit: int):
    """带 `File(...)` 参数的最小 app —— 复现**真实端点**的请求体解析路径。

    为什么必须带 `File(...)`：FastAPI 只在 `request_body_to_args` 里才有那句
        except HTTPException: raise
        except Exception: raise HTTPException(400, "There was an error parsing the body")
    所以"超限应当返回 413"这件事，只有在**声明了 body 参数**的端点上才验证得动。
    不带参数的端点会绕过那段代码，从而给出与实际部署不一致的结论。
    """
    from fastapi import FastAPI, File, UploadFile
    from fastapi.testclient import TestClient

    app = FastAPI()

    @app.post("/upload")
    async def upload(file: UploadFile = File(...)):  # noqa: ARG001
        return {"ok": True}

    app.add_middleware(BodyLimitMiddleware, max_body_bytes=limit)
    return TestClient(app)


def test_chunked_body_over_limit_returns_413_not_400():
    """⚠️ 回归：没有 Content-Length 的超限请求必须返回 **413**，而不是 400。

    上一版把 `RequestBodyTooLarge` 定义成普通 `Exception`，而 FastAPI 解析请求体时会把
    非 HTTPException 一律改写成 400 —— 真实端点上永远拿不到 413，
    客户端也就分不清"文件太大"和"请求格式错"。根因见 fastapi/routing.py。
    """
    client = _client_with_file_route(limit=1024)

    def _chunks():
        for _ in range(10):
            yield b"x" * 512  # 共 5KB，远超 1024

    resp = client.post(
        "/upload",
        content=_chunks(),  # 迭代器 → httpx 用 chunked 编码，不带 Content-Length
        headers={"content-type": "multipart/form-data; boundary=----x"},
    )

    assert resp.status_code == 413, resp.text


def test_content_length_over_limit_returns_413_on_a_real_endpoint():
    """有 Content-Length 时走中间件自己的响应路径，也必须是 413。"""
    client = _client_with_file_route(limit=1024)
    resp = client.post(
        "/upload",
        content=b"x" * 4096,
        headers={"content-type": "multipart/form-data; boundary=----x"},
    )

    assert resp.status_code == 413, resp.text


def test_uppercase_content_length_header_is_still_honoured():
    """ASGI 规范要求 header name 小写，但真实服务器/替身不保证；
    写死小写会漏掉预检（虽然 httptools/h11 都是小写，属于纵深防御）。"""
    import body_limit

    assert body_limit._content_length(
        {"headers": [(b"Content-Length", b"999999")]}
    ) == 999999


# ── 5. 上限常量之间的关系 ────────────────────────────────────
def test_body_limit_leaves_room_for_multipart_framing():
    """整个请求体的上限必须**略大于**单文件上限，否则一个刚好 50MB 的文件
    会因为 multipart 帧开销被中间件先拒掉。"""
    assert body_limit.MAX_BODY_BYTES > body_limit.MAX_UPLOAD_BYTES


def test_upload_module_shares_the_single_upload_limit():
    """两个上限不能各处维护：上传接口的常量必须就是 body_limit 那一个。"""
    from api import upload_file

    assert upload_file.MAX_UPLOAD_BYTES == body_limit.MAX_UPLOAD_BYTES
