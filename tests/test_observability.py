"""可观测性测试（A5）—— 完全离线。

守的性质：

1. **request_id 贯穿**：响应头回写、日志里带上、Starlette 的 `request.state` 能读到；
2. **外部传入的 request id 必须校验**——它是外部输入，直接进日志就是**日志注入**；
3. **指标标签用路由模板**：两个不同 doc_id 的请求必须落到**同一条**时间序列，
   否则基数会随业务数据增长把监控打爆；
4. 访问日志级别与状态码匹配（4xx WARNING / 5xx ERROR），便于直接配告警。
"""

import logging

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

import observability as obs
from metrics import Registry


class _Recorder(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def logs():
    """捕获 observability 的日志（并断开真实输出，保持测试安静）。

    刻意**照生产的样子装上 RequestIdFilter** —— 只比较 Level/消息而漏掉过滤器，
    会得到"生产日志有 rid、测试里没有"的假象。
    """
    handler = _Recorder()
    handler.addFilter(obs.RequestIdFilter())
    logger = logging.getLogger("observability")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.propagate = True


@pytest.fixture
def app(monkeypatch):
    """带观测中间件的最小 app；指标用独立注册表，避免与全局状态互相污染。"""
    fresh = Registry()
    monkeypatch.setattr(obs, "http_requests", lambda: fresh.counter(
        "http_requests_total", "HTTP 请求数", ("method", "path", "status")))
    monkeypatch.setattr(obs, "http_latency", lambda: fresh.histogram(
        "http_request_duration_seconds", "HTTP 请求耗时", ("method", "path")))

    app = FastAPI()

    @app.get("/items/{item_id}")
    async def item(item_id: str, request: Request):
        return {"item_id": item_id, "rid": getattr(request.state, "request_id", None)}

    @app.get("/boom")
    async def boom():
        raise HTTPException(status_code=503, detail="挂了")

    @app.get("/crash")
    async def crash():
        raise RuntimeError("未捕获异常")

    obs.add_observability(app)
    return app, fresh


# ── request_id ──────────────────────────────────────────────
def test_request_id_is_generated_and_returned(app):
    client = TestClient(app[0], raise_server_exceptions=False)

    resp = client.get("/items/1")

    rid = resp.headers.get("x-request-id")
    assert rid and len(rid) == 16
    assert resp.json()["rid"] == rid


def test_client_supplied_request_id_is_propagated(app):
    client = TestClient(app[0])

    resp = client.get("/items/1", headers={"X-Request-ID": "trace-abc-123"})

    assert resp.headers["x-request-id"] == "trace-abc-123"


@pytest.mark.parametrize(
    "hostile",
    [
        "abc\ndef",          # 换行 → 伪造日志行
        "abc def",           # 空格
        'abc"def',           # 引号
        "x" * 100,           # 超长
        "",                  # 空
    ],
)
def test_hostile_request_id_is_replaced_not_trusted(app, hostile):
    """⚠️ request id 是外部输入：不校验就等于把日志注入的载体开放出去。"""
    client = TestClient(app[0])

    resp = client.get("/items/1", headers={"X-Request-ID": hostile})

    rid = resp.headers["x-request-id"]
    assert rid != hostile
    assert len(rid) == 16  # 换成了服务端生成的


def test_sanitize_request_id_unit():
    assert obs.sanitize_request_id("ok-123_AbC.def") == "ok-123_AbC.def"
    assert obs.sanitize_request_id("bad\nvalue") is None
    assert obs.sanitize_request_id("абв") is None  # 非 ASCII：HTTP 头里根本传不过来
    assert obs.sanitize_request_id(None) is None
    assert obs.sanitize_request_id("x" * 65) is None  # 超长


# ── 指标标签 ────────────────────────────────────────────────
def test_metric_label_is_the_route_template_not_the_raw_path(app):
    """两个不同 item_id 必须落到同一条序列（否则基数随业务增长爆掉）。"""
    client = TestClient(app[0])
    _app, reg = app

    client.get("/items/1")
    client.get("/items/2")

    text = reg.render()
    assert 'path="/items/{item_id}"' in text
    assert 'http_requests_total{method="GET",path="/items/{item_id}",status="200"} 2' in text
    assert "/items/1" not in text and "/items/2" not in text


def test_unmatched_paths_collapse_into_one_label(app):
    """扫描器打随机路径：不能每个路径一条序列。

    断言落在 **counter 采样行**上（histogram 会为同一个标签渲染十几个桶行，
    直接数字符串出现次数会数错）。
    """
    client = TestClient(app[0])
    _app, reg = app

    for suffix in ("aaa", "bbb", "ccc"):
        client.get(f"/{suffix}")

    text = reg.render()
    assert 'http_requests_total{method="GET",path="unmatched",status="404"} 3' in text
    for suffix in ("aaa", "bbb", "ccc"):
        assert f'path="/{suffix}"' not in text


def test_status_codes_are_recorded_separately(app):
    client = TestClient(app[0], raise_server_exceptions=False)
    _app, reg = app

    client.get("/items/1")
    client.get("/boom")
    client.get("/crash")

    text = reg.render()
    assert 'status="200"' in text
    assert 'status="503"' in text
    assert 'status="500"' in text


# ── 访问日志 ────────────────────────────────────────────────
def test_access_log_is_one_line_with_request_id(app, logs):
    client = TestClient(app[0], raise_server_exceptions=False)

    client.get("/items/1", headers={"X-Request-ID": "rid-42"})

    messages = [r.getMessage() for r in logs.records if "GET" in r.getMessage()]
    assert any("rid-42" in obs.KeyValueFormatter().format(r) for r in logs.records)
    assert any("/items/1 -> 200" in m for m in messages)


def test_access_log_level_reflects_status(app, logs):
    client = TestClient(app[0], raise_server_exceptions=False)

    client.get("/items/1")   # 200 → INFO
    client.get("/boom")      # 503 → ERROR
    client.get("/nope")      # 404 → WARNING

    by_status = {r.getMessage(): r.levelno for r in logs.records if "->" in r.getMessage()}
    levels = {line.split("->")[1].strip().split()[0]: lvl for line, lvl in by_status.items()}

    assert levels["200"] == logging.INFO
    assert levels["503"] == logging.ERROR
    assert levels["404"] == logging.WARNING


# ── 日志格式 ────────────────────────────────────────────────
def test_formatter_includes_request_id_and_level():
    formatter = obs.KeyValueFormatter()
    record = logging.LogRecord("api.chat", logging.INFO, "f.py", 1, "收到请求", None, None)
    record.request_id = "abc123"

    text = formatter.format(record)

    assert "rid=abc123" in text
    assert "INFO" in text
    assert "[api.chat]" in text
    assert "收到请求" in text


def test_formatter_falls_back_to_dash_without_request_context():
    formatter = obs.KeyValueFormatter()
    record = logging.LogRecord("x", logging.INFO, "f.py", 1, "无上下文", None, None)

    assert "rid=-" in formatter.format(record)


def test_request_id_filter_injects_current_value():
    filt = obs.RequestIdFilter()
    token = obs._request_id.set("zzz")  # noqa: SLF001 - 直接验证过滤器
    try:
        record = logging.LogRecord("x", logging.INFO, "f.py", 1, "m", None, None)
        assert filt.filter(record) is True
        assert record.request_id == "zzz"
    finally:
        obs._request_id.reset(token)  # noqa: SLF001


def test_context_is_cleared_after_the_request(app):
    """请求结束必须还原：否则 request_id 会串到"同一个 worker 里的下一个请求"。"""
    client = TestClient(app[0])

    client.get("/items/1")

    assert obs.current_request_id() is None


# ── 日志脱敏 ────────────────────────────────────────────────
def test_redacting_filter_masks_pii_in_log_messages():
    """脱敏放在 handler 上：只要经过日志就一定被处理，不依赖调用点自觉。"""
    import config
    import moderation as m

    original = config.MODERATION_MASK_LOGS
    config.MODERATION_MASK_LOGS = True
    try:
        record = logging.LogRecord(
            "api.chat", logging.INFO, "f.py", 1,
            "用户 %s 的订单查询", ("13812345678",), None,
        )
        filt = obs.RedactingFilter()

        assert filt.filter(record) is True

        text = obs.KeyValueFormatter().format(record)
        assert "138****5678" in text
        assert "13812345678" not in text
        assert m.mask_pii("13812345678") == "138****5678"
    finally:
        config.MODERATION_MASK_LOGS = original


def test_redacting_filter_clears_args_after_rewriting():
    """⚠️ 改写 msg 后必须清空 args，否则 Formatter 会再套一次 %-格式化。"""
    import config

    original = config.MODERATION_MASK_LOGS
    config.MODERATION_MASK_LOGS = True
    try:
        record = logging.LogRecord(
            "x", logging.INFO, "f.py", 1, "号码 %s", ("13812345678",), None
        )

        obs.RedactingFilter().filter(record)

        assert record.args == ()
        # 关键：格式化不能抛异常（args 没清空时会 "not all arguments converted"）
        assert "138****5678" in obs.KeyValueFormatter().format(record)
    finally:
        config.MODERATION_MASK_LOGS = original


def test_redacting_filter_can_be_disabled():
    import config

    original = config.MODERATION_MASK_LOGS
    config.MODERATION_MASK_LOGS = False
    try:
        record = logging.LogRecord("x", logging.INFO, "f.py", 1, "号码 13812345678", (), None)

        obs.RedactingFilter().filter(record)

        assert "13812345678" in obs.KeyValueFormatter().format(record)
    finally:
        config.MODERATION_MASK_LOGS = original


def test_redacting_filter_leaves_clean_logs_untouched():
    record = logging.LogRecord("x", logging.INFO, "f.py", 1, "订单 A1 已发货", (), None)

    obs.RedactingFilter().filter(record)

    assert obs.KeyValueFormatter().format(record).endswith("订单 A1 已发货")


def test_setup_logging_registers_both_filters():
    """真实入口必须同时装上 request_id 与脱敏两个 filter。"""
    root = logging.getLogger()
    before = list(root.handlers)
    try:
        obs.setup_logging()
        filters = root.handlers[0].filters
        assert any(isinstance(f, obs.RequestIdFilter) for f in filters)
        assert any(isinstance(f, obs.RedactingFilter) for f in filters)
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
        for handler in before:
            root.addHandler(handler)


def test_setup_logging_is_idempotent():
    """重复调用不能叠加 handler（否则每条日志打多遍）。"""
    root = logging.getLogger()
    before = list(root.handlers)
    try:
        obs.setup_logging("WARNING")
        obs.setup_logging("WARNING")
        assert len(root.handlers) == 1
        assert root.level == logging.WARNING
        assert isinstance(root.handlers[0].formatter, obs.KeyValueFormatter)
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
        for handler in before:
            root.addHandler(handler)
