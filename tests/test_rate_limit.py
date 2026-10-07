"""入口限流测试（A7）—— 完全离线：Redis 用替身，进程内桶是真实现。

守的性质：

1. **令牌桶的边界要准**：`refill()` 是纯函数，直接验证"刚好够 / 差一点"，
   限流这种东西差一个边界就会要么误杀要么放行，靠集成测试碰运气不可靠；
2. **按账号隔离**：一个账号刷爆不该影响别人（按 IP 限流会误伤整个出口）；
3. **Redis 挂了要降级而不是裸奔**：进程内桶继续按单实例限流，
   既不打挂服务，也不在故障期间完全不设防；
4. **响应要可重试**：429 必须带 `Retry-After`，客户端才知道等多久。
"""

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

import config
import rate_limit
from metrics import REGISTRY
from rate_limit import Decision, RateLimiter, _LocalBuckets, refill


@pytest.fixture(autouse=True)
def _restore_config():
    original = (config.RATE_LIMIT_ENABLED, config.RATE_LIMIT_PER_MINUTE, config.RATE_LIMIT_BURST)
    config.RATE_LIMIT_ENABLED = True
    config.RATE_LIMIT_PER_MINUTE = 60
    config.RATE_LIMIT_BURST = 5
    try:
        yield
    finally:
        config.RATE_LIMIT_ENABLED, config.RATE_LIMIT_PER_MINUTE, config.RATE_LIMIT_BURST = original


# ══════════════════════════════════════════════════════════════
# 纯函数：令牌桶
# ══════════════════════════════════════════════════════════════
def test_full_bucket_allows_and_decrements():
    tokens, ts, allowed, retry = refill(5.0, 100.0, 100.0, rate=1.0, burst=5.0)

    assert allowed is True
    assert tokens == 4.0
    assert ts == 100.0
    assert retry == 0.0


def test_empty_bucket_denies_and_reports_the_wait():
    tokens, _ts, allowed, retry = refill(0.0, 100.0, 100.0, rate=1.0, burst=5.0)

    assert allowed is False
    assert tokens == 0.0
    assert retry == pytest.approx(1.0), "差 1 个令牌、速率 1/s → 等 1 秒"


def test_bucket_refills_over_time():
    tokens, _ts, allowed, _retry = refill(0.0, 100.0, 105.0, rate=1.0, burst=5.0)

    assert allowed is True
    assert tokens == pytest.approx(4.0)


def test_refill_is_capped_at_burst():
    tokens, _ts, allowed, _retry = refill(0.0, 0.0, 10000.0, rate=1.0, burst=5.0)

    assert allowed is True
    assert tokens == pytest.approx(4.0), "空闲再久也不能攒超过桶容量"


def test_partial_refill_still_denies_when_not_enough():
    _tokens, _ts, allowed, retry = refill(0.2, 100.0, 100.5, rate=1.0, burst=5.0, cost=1.0)

    assert allowed is False
    assert retry == pytest.approx(0.3)


def test_zero_rate_does_not_divide_by_zero():
    """配置写错（每分钟 0 次）时不能抛异常把接口打成 500。"""
    _tokens, _ts, allowed, retry = refill(0.0, 0.0, 0.0, rate=0.0, burst=1.0)

    assert allowed is False
    assert retry > 0


def test_clock_going_backwards_is_treated_as_no_elapsed_time():
    """NTP 校时/单调钟回绕时不能凭空造出令牌。"""
    tokens, _ts, allowed, _retry = refill(0.0, 100.0, 50.0, rate=1.0, burst=5.0)

    assert allowed is False
    assert tokens == 0.0


# ══════════════════════════════════════════════════════════════
# 进程内桶
# ══════════════════════════════════════════════════════════════
def test_local_buckets_allow_burst_then_deny():
    buckets = _LocalBuckets()

    decisions = [buckets.consume("u1", rate=1.0, burst=3.0, now=100.0) for _ in range(4)]

    assert [d.allowed for d in decisions] == [True, True, True, False]
    assert decisions[-1].backend == "local"


def test_local_buckets_are_isolated_per_key():
    buckets = _LocalBuckets()

    for _ in range(3):
        buckets.consume("u1", rate=1.0, burst=3.0, now=100.0)

    assert buckets.consume("u2", rate=1.0, burst=3.0, now=100.0).allowed is True


def test_local_buckets_recover_after_waiting():
    buckets = _LocalBuckets()
    for _ in range(3):
        buckets.consume("u1", rate=1.0, burst=3.0, now=100.0)

    assert buckets.consume("u1", rate=1.0, burst=3.0, now=100.0).allowed is False
    assert buckets.consume("u1", rate=1.0, burst=3.0, now=103.0).allowed is True


def test_local_buckets_evict_idle_keys():
    """每个账号一个桶 -> 必须能回收，否则内存随用户数无界增长。"""
    buckets = _LocalBuckets(idle_ttl=10.0)
    for i in range(1024):
        buckets.consume(f"old{i}", rate=1.0, burst=1.0, now=0.0)

    buckets.consume("new", rate=1.0, burst=1.0, now=1000.0)

    assert len(buckets._state) < 1024


# ══════════════════════════════════════════════════════════════
# RateLimiter：Redis 优先、进程内兜底
# ══════════════════════════════════════════════════════════════
class _FakeRedis:
    def __init__(self, result=(1, "4", "0"), error=None):
        self.result = result
        self.error = error
        self.calls = []

    async def eval(self, script, numkeys, key, rate, burst, now, cost, ttl):
        self.calls.append({"key": key, "rate": rate, "burst": burst, "ttl": ttl})
        if self.error:
            raise self.error
        return list(self.result)


def test_redis_backend_is_used_when_available():
    client = _FakeRedis(result=(1, "4", "0"))
    limiter = RateLimiter(redis_client=client)

    import asyncio

    decision = asyncio.run(limiter.check("chat:u1", per_minute=60, burst=5))

    assert decision.allowed is True
    assert decision.backend == "redis"
    assert client.calls and client.calls[0]["key"] == "ratelimit:chat:u1"


def test_redis_denial_is_propagated():
    client = _FakeRedis(result=(0, "0", "2.5"))
    limiter = RateLimiter(redis_client=client)

    import asyncio

    decision = asyncio.run(limiter.check("chat:u1", per_minute=60, burst=5))

    assert decision.allowed is False
    assert decision.retry_after == pytest.approx(2.5)
    assert decision.backend == "redis"


def test_redis_failure_degrades_to_local_instead_of_open():
    """⚠️ 关键：Redis 挂掉不能变成"完全不限流"（那才是最危险的状态）。"""
    client = _FakeRedis(error=ConnectionError("redis 连不上"))
    limiter = RateLimiter(redis_client=client)

    import asyncio

    decisions = [
        asyncio.run(limiter.check("chat:u1", per_minute=60, burst=2)) for _ in range(3)
    ]

    assert [d.backend for d in decisions] == ["local", "local", "local"]
    assert [d.allowed for d in decisions] == [True, True, False]


def test_disabled_rate_limit_always_allows():
    config.RATE_LIMIT_ENABLED = False
    limiter = RateLimiter(redis_client=None)

    import asyncio

    for _ in range(10):
        assert asyncio.run(limiter.check("chat:u1", per_minute=1, burst=1)).allowed is True


# ══════════════════════════════════════════════════════════════
# 作为 FastAPI 依赖
# ══════════════════════════════════════════════════════════════
def _limited_app(as_role, user_id: str = "u_1") -> TestClient:
    """最小 app：一条带限流依赖的路由 + 指定身份的鉴权覆盖。"""
    app = FastAPI()

    @app.get("/limited", dependencies=[Depends(rate_limit.rate_limit("test"))])
    async def limited() -> dict:
        return {"ok": True}

    as_role(app, role="user", user_id=user_id)
    return TestClient(app)


@pytest.fixture
def client(monkeypatch, as_role):
    limiter = RateLimiter(redis_client=None)
    monkeypatch.setattr(rate_limit, "_limiter", limiter)
    return _limited_app(as_role), limiter


def test_requests_within_the_burst_succeed(client):
    c, _limiter = client
    config.RATE_LIMIT_PER_MINUTE = 60
    config.RATE_LIMIT_BURST = 3

    codes = [c.get("/limited").status_code for _ in range(3)]

    assert codes == [200, 200, 200]


def test_exceeding_the_burst_returns_429_with_retry_after(client):
    c, _limiter = client
    config.RATE_LIMIT_PER_MINUTE = 60
    config.RATE_LIMIT_BURST = 2

    c.get("/limited")
    c.get("/limited")
    r = c.get("/limited")

    assert r.status_code == 429
    assert "Retry-After" in r.headers
    assert int(r.headers["Retry-After"]) >= 1
    assert "过于频繁" in r.json()["detail"]


def test_429_is_counted_in_metrics(client):
    from metrics import rate_limited

    c, _limiter = client
    config.RATE_LIMIT_BURST = 1
    before = rate_limited().value(scope="test")

    c.get("/limited")
    c.get("/limited")

    assert rate_limited().value(scope="test") == before + 1


def test_buckets_are_per_user(monkeypatch, as_role):
    """按账号隔离：一个用户刷爆不影响别人（按 IP 限流会误伤整个出口）。"""
    monkeypatch.setattr(rate_limit, "_limiter", RateLimiter(redis_client=None))
    config.RATE_LIMIT_BURST = 1
    config.RATE_LIMIT_PER_MINUTE = 1

    ca = _limited_app(as_role, user_id="user_a")
    cb = _limited_app(as_role, user_id="user_b")

    assert ca.get("/limited").status_code == 200
    assert ca.get("/limited").status_code == 429
    assert cb.get("/limited").status_code == 200, "另一个账号不该被牵连"


def test_dependency_requires_authentication():
    """未认证请求先拿 401 而不是 429：不该走到业务逻辑，也不该消耗配额。"""
    app = FastAPI()

    @app.get("/limited", dependencies=[Depends(rate_limit.rate_limit("test"))])
    async def limited() -> dict:
        return {"ok": True}

    assert TestClient(app).get("/limited").status_code == 401


def test_decision_shape():
    d = Decision(allowed=True, remaining=1.0, retry_after=0.0)

    assert d.backend == "local"
    REGISTRY  # 明确：指标注册表是全局单例，这里不做干扰
