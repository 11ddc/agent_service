"""请求体大小上限 —— 在 multipart 解析**之前**生效。

## 为什么必须独立一层

FastAPI 的 `UploadFile = File(...)` 依赖会在**解析 multipart 的时候**就把整个 body
交给 Starlette 的 `SpooledTemporaryFile`（超过 1MB 就落到临时磁盘）。而 Starlette
对**文件字段没有任何大小上限** —— `max_part_size`（默认 1MB）只作用于非文件字段
（见 `starlette/formparsers.py`：文件部分走 `SpooledTemporaryFile(max_size=...)`，
不参与 `max_part_size` 检查）。

于是"在上传接口里判断 `file.size > MAX_UPLOAD_BYTES`"是**事后检查**：
一个 10GB 的请求会先被完整缓冲到内存/临时盘，然后才拿到 413。
未鉴权接口上这就是一个内存/磁盘耗尽面。

## 这一层怎么做

利用 ASGI 的 `receive` 通道，在解析发生之前掐断：

1. **有 Content-Length**：直接判，超限的请求**一个字节都不会被读取**，立刻 413；
2. **没有 Content-Length**（chunked）：边收边计数，一旦超过上限就抛
   `RequestBodyTooLarge`，由 FastAPI 的异常处理器转成 413 ——
   占用被限制在上限附近，而不是无界。

## 两个上限的分工

- `MAX_BODY_BYTES`（本模块）：**整个请求体**的上限，挡的是无界缓冲；
- `MAX_UPLOAD_BYTES`（`api/upload_file.py` 里那个精确的 50MB）：**单个文件**的上限，
  仍然由上传接口自己判，好在 multipart 帧开销之外留出余量。
"""

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB：与 api/upload_file.py 的对外承诺一致

# multipart 的 boundary/分片头部会让 body 比文件本身略大；留 1MB 余量，
# 让"单文件 <= 50MB"这条业务规则继续由上传接口精确判定。
MAX_BODY_BYTES = MAX_UPLOAD_BYTES + 1024 * 1024


class RequestBodyTooLarge(Exception):
    """请求体超过上限。由 `add_body_limit` 注册的处理器转成 413。"""

    def __init__(self, limit: int) -> None:
        super().__init__(f"请求体超过上限 {limit} 字节")
        self.limit = limit


def _content_length(scope: dict) -> int | None:
    """从 ASGI scope 里取 Content-Length；取不到或非法都返回 None。"""
    for key, value in scope.get("headers") or []:
        if key == b"content-length":
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


class BodyLimitMiddleware:
    """纯 ASGI 中间件：超限时在解析前返回 413。

    刻意不用 `BaseHTTPMiddleware`：那层会把请求体包装成额外的任务/流，
    既多一次拷贝，也让"提前掐断"变得不直接。
    """

    def __init__(self, app, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        limit = self.max_body_bytes

        declared = _content_length(scope)
        if declared is not None and declared > limit:
            # 先判 Content-Length：超限请求连一个字节都不读
            await self._reject(scope, receive, send, declared, limit)
            return

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body") or b"")
                if received > limit:
                    # 抛给 FastAPI 的异常处理器 → 413。这里不自己发响应，
                    # 因为应用可能已经开始/准备开始发响应，重复发会破坏 ASGI 协议。
                    raise RequestBodyTooLarge(limit)
            return message

        await self.app(scope, limited_receive, send)

    @staticmethod
    async def _reject(scope, receive, send, declared: int, limit: int) -> None:
        from starlette.responses import JSONResponse

        response = JSONResponse(
            status_code=413,
            content={
                "detail": f"请求体过大（{declared} 字节，上限 {limit} 字节）",
            },
        )
        await response(scope, receive, send)


def add_body_limit(app) -> None:
    """给 FastAPI 应用装上请求体上限，并把超限异常映射成 413。"""
    from fastapi import Request
    from starlette.responses import JSONResponse

    async def _handle(request: Request, exc: RequestBodyTooLarge):
        return JSONResponse(
            status_code=413,
            content={"detail": f"请求体过大（上限 {exc.limit} 字节）"},
        )

    app.add_exception_handler(RequestBodyTooLarge, _handle)
    app.add_middleware(BodyLimitMiddleware, max_body_bytes=MAX_BODY_BYTES)
