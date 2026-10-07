"""可观测性最小集：request_id + 结构化日志 + 访问日志 + 指标打点。

## request_id 贯穿全链路

中间件为每个请求生成（或沿用客户端传来的）`X-Request-ID`，放进 ContextVar，
并回写响应头。日志格式里带上它 —— 于是"用户报障 → 用一行日志捞出这次请求的
全部记录"成为可能，而不是在几千行输出里大海捞针。

沿用客户端传来的值时**必须做格式校验**：`X-Request-ID` 是外部输入，
直接写进日志就是**日志注入**的载体（塞换行就能伪造日志行）。

## 日志格式

单行 key=value，人在终端能读、采集器也能解析：

    2026-10-07 22:00:00 INFO    [api.chat] rid=ab12cd34 收到 /chat user=u_1

要接 ELK/Loki 时换掉 formatter 即可，业务代码不用动。

## 访问日志

每个请求结束打一行：method / path / status / duration_ms / rid。
4xx 记 WARNING、5xx 记 ERROR —— 这样"配告警"是直接可做的，不用再做字符串匹配。

## 指标标签的基数

path 用**路由模板**（`/api/kb/documents/{doc_id}/publish`）而不是原始路径，
否则每个 doc_id 都会变成一条时间序列，把监控打爆。没匹配到路由的请求
（扫描器）统一记成 `unmatched`。
"""

import logging
import re
import time
import uuid
from contextvars import ContextVar

import config
from metrics import http_latency, http_requests
from moderation import mask_pii

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "x-request-id"

# 只允许这些字符：足够长、且不可能注入换行/引号
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._\-]{1,64}$")

_request_id: ContextVar[str | None] = ContextVar("dsh_request_id", default=None)


def current_request_id() -> str | None:
    return _request_id.get()


def sanitize_request_id(raw: str | None) -> str | None:
    """校验外部传入的 request id；不合规一律丢弃（改用服务端生成的）。"""
    if raw and _SAFE_REQUEST_ID.match(raw):
        return raw
    return None


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


class RequestIdFilter(logging.Filter):
    """把当前请求 id 注入每条日志记录（没有请求上下文时是 `-`）。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = current_request_id() or "-"
        return True


class RedactingFilter(logging.Filter):
    """把日志里的 PII（手机号/身份证/银行卡/邮箱）打码。

    为什么放在**日志层**而不是各调用点：调用点有几十处、还会不断增加，
    漏一处就泄露一次；挂在 handler 上则"只要经过日志就一定会被处理"。
    审计表里的 detail 另有一道（见 auth.audit）。

    ⚠️ 改写 `record.msg` 后必须清空 `record.args`：否则 Formatter 会拿着
    已经被插值过的字符串再套一次 %-格式化，轻则输出错乱，重则抛异常。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not config.MODERATION_MASK_LOGS:
            return True
        try:
            raw = record.getMessage()
            masked = mask_pii(raw)
            if masked != raw:
                record.msg = masked
                record.args = ()
        except Exception:  # noqa: BLE001 - 脱敏失败绝不能把日志本身弄挂
            pass
        return True


class KeyValueFormatter(logging.Formatter):
    """单行 key=value 格式，带 request_id。"""

    default_time_format = "%Y-%m-%d %H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()
        head = (
            f"{self.formatTime(record, self.default_time_format)} "
            f"{record.levelname:<7} [{record.name}] "
            f"rid={getattr(record, 'request_id', '-')}"
        )
        text = f"{head} {record.message}"
        if record.exc_info:
            text += "\n" + self.formatException(record.exc_info)
        return text


def setup_logging(level: str | None = None) -> None:
    """配置根 logger（应用入口调用一次）。

    同时把几个第三方库压到 WARNING：openai SDK 每个请求都会打一行，
    不压的话真正的业务日志会被淹掉。
    """
    handler = logging.StreamHandler()
    handler.setFormatter(KeyValueFormatter())
    handler.addFilter(RequestIdFilter())
    # 脱敏放在 handler 上：只要经过日志就一定被处理，不依赖各调用点自觉
    handler.addFilter(RedactingFilter())

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(getattr(logging, (level or config.LOG_LEVEL).upper(), logging.INFO))

    for noisy in ("httpx", "httpx2", "httpcore", "urllib3", "chromadb", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _header(scope: dict, name: bytes) -> str | None:
    for key, value in scope.get("headers") or []:
        if key.lower() == name:
            try:
                return value.decode("latin-1")
            except Exception:  # noqa: BLE001
                return None
    return None


def _route_label(scope: dict) -> str:
    """路由模板（不是原始路径）：避免高基数标签。"""
    route = scope.get("route")
    path = getattr(route, "path", None)
    if path:
        return path
    raw = scope.get("path") or "?"
    # 没匹配到路由（404/扫描器）：收敛成一个固定标签
    return raw if scope.get("endpoint") or scope.get("route") else "unmatched"


class ObservabilityMiddleware:
    """纯 ASGI 中间件：request_id + 访问日志 + 指标。

    刻意不用 `BaseHTTPMiddleware`：那一层对**流式响应**（本项目的 SSE）会额外
    包一层任务与内存流，既多一次拷贝，也可能改变取消语义。
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = sanitize_request_id(
            _header(scope, REQUEST_ID_HEADER.encode())
        ) or new_request_id()
        scope.setdefault("state", {})["request_id"] = request_id
        token = _request_id.set(request_id)

        started = time.perf_counter()
        status_code = {"value": 500}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_code["value"] = message["status"]
                message.setdefault("headers", []).append(
                    (REQUEST_ID_HEADER.encode(), request_id.encode("latin-1"))
                )
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration = time.perf_counter() - started
            method = scope.get("method", "?")
            path = _route_label(scope)
            status = status_code["value"]

            http_requests().inc(method=method, path=path, status=status)
            http_latency().observe(duration, method=method, path=path)

            level = logging.INFO
            if status >= 500:
                level = logging.ERROR
            elif status >= 400:
                level = logging.WARNING
            logger.log(
                level,
                "%s %s -> %s %.1fms",
                method,
                scope.get("path", "?"),
                status,
                duration * 1000,
            )
            _request_id.reset(token)


def add_observability(app) -> None:
    """挂载中间件。

    ⚠️ 顺序：观测层要是**最外层**（最后 add），这样连"请求体超限/JSON 解析失败"
    这类早期返回也能被记进指标与访问日志；否则它们在观测层之内，
    会变成监控里的盲区。
    """
    app.add_middleware(ObservabilityMiddleware)
