"""限流的 **Redis 路径**集成测试 —— 默认不跑，需要真 Redis。

    .\\venv\\Scripts\\python.exe -m pytest -m integration -q tests/test_rate_limit_integration.py

## 为什么需要

单测里 Redis 是替身、进程内桶是真实现，所以 **Lua 脚本本身从没被执行过**。
而"读-算-写"必须在服务端原子完成，正是这段 Lua 的意义所在 —— 不真跑一遍，
就不知道脚本语法对不对、返回值类型是不是我们假设的那样。

## 为什么每个用例都自建 client

`redis.asyncio` 的连接池绑定在**创建它的那个事件循环**上。测试里每个用例
`asyncio.run()` 都建新循环，共用一个模块级 client 时下一个用例会去复用上一个
循环留下的连接 → `RuntimeError: Event loop is closed`（实测踩过：同样的用例
换个执行顺序就时好时坏）。生产里只有一个长驻循环，所以这是测试脚手架的问题，
不是产品问题 —— 但必须在这里处理干净，否则这文件就是"时灵时不灵"的。

Redis 不可达时整文件 skip（不是 fail）：本地没有 Redis 是常态。
"""

import asyncio
import time

import pytest

from rate_limit import RateLimiter

pytestmark = pytest.mark.integration


def _redis_url() -> str:
    import config

    return config.REDIS_URL


def _run(coro_factory):
    """在**单个**事件循环里完成"建 client → 操作 → 关 client"。

    coro_factory 接收 client，返回协程；这样用例内部不必关心连接生命周期。
    """
    import redis.asyncio as aioredis

    async def _wrapper():
        client = aioredis.from_url(_redis_url())
        try:
            return await coro_factory(client)
        finally:
            await client.aclose()

    return asyncio.run(_wrapper())


@pytest.fixture(scope="module", autouse=True)
def _require_redis():
    async def _probe():
        import redis.asyncio as aioredis

        client = aioredis.from_url(_redis_url())
        try:
            await client.ping()
        finally:
            await client.aclose()

    try:
        asyncio.run(_probe())
    except Exception as e:  # pragma: no cover - 本地无 Redis 时走这里
        pytest.skip(f"Redis 不可达，跳过限流集成测试: {e}")


@pytest.fixture
def key():
    """每个用例一个独立 key，结束后清掉（Redis 是共享状态）。"""
    name = f"itest:ratelimit:{time.time_ns()}"
    yield name

    async def _cleanup(client):
        try:
            await client.delete(f"ratelimit:{name}")
        except Exception:  # noqa: BLE001
            pass

    _run(_cleanup)


def test_lua_script_allows_up_to_burst_then_denies(key):
    async def _body(client):
        limiter = RateLimiter(redis_client=client)
        return [await limiter.check(key, per_minute=60, burst=3) for _ in range(5)]

    decisions = _run(_body)

    assert [d.backend for d in decisions] == ["redis"] * 5
    assert [d.allowed for d in decisions] == [True, True, True, False, False]
    assert decisions[-1].retry_after > 0


def test_lua_script_returns_a_usable_retry_after(key):
    """返回值类型是我按 Redis 语义假设的：数字经 `tostring` 变成 bulk string。

    假设不成立（例如浮点被截断成整数）时 `retry_after` 会变成 0，
    客户端会拿到 `Retry-After: 1` 而永远重试 —— 所以这条必须真跑 Redis 验。
    """
    async def _body(client):
        limiter = RateLimiter(redis_client=client)
        for _ in range(2):
            await limiter.check(key, per_minute=60, burst=2)
        return await limiter.check(key, per_minute=60, burst=2)

    decision = _run(_body)

    assert decision.allowed is False
    assert decision.retry_after > 0.0


def test_state_is_shared_across_limiter_instances(key):
    """两个实例共享同一份额度 —— 这正是"限流外置"的意义（多副本共享配额）。"""
    async def _body(client):
        a = RateLimiter(redis_client=client)
        b = RateLimiter(redis_client=client)
        return (
            await a.check(key, per_minute=60, burst=2),
            await b.check(key, per_minute=60, burst=2),
            await b.check(key, per_minute=60, burst=2),
        )

    first, second, third = _run(_body)

    assert (first.allowed, second.allowed, third.allowed) == (True, True, False)


def test_bucket_key_has_a_ttl(key):
    """桶必须有 TTL，否则"每个账号一个 key"会在 Redis 里越积越多。"""
    async def _body(client):
        limiter = RateLimiter(redis_client=client)
        await limiter.check(key, per_minute=60, burst=1)
        return await client.ttl(f"ratelimit:{key}")

    ttl = _run(_body)

    assert ttl > 0, f"限流 key 没有过期时间（ttl={ttl}）"


def test_bucket_refills_in_real_time(key):
    """真实时钟下也要能恢复：避免"窗口只前进不补充"这类实现错误。"""
    async def _body(client):
        limiter = RateLimiter(redis_client=client)
        await limiter.check(key, per_minute=600, burst=1)   # 10/s
        denied = await limiter.check(key, per_minute=600, burst=1)
        await asyncio.sleep(0.3)                            # 够补 3 个令牌
        after = await limiter.check(key, per_minute=600, burst=1)
        return denied, after

    denied, after = _run(_body)

    assert denied.allowed is False
    assert after.allowed is True
