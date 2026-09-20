"""转人工滑动窗口(servercustomer)测试 —— 用内存假 Redis,不启动任何服务。

servercustomer 只依赖 redis_client 的 incr / expire / delete 三个方法,
把 api.chat 模块里的 redis_client 引用替换成 FakeRedis 即可。
"""
import asyncio

import pytest

from api import chat


class FakeRedis:
    """内存假 Redis:只实现 servercustomer 用到的三个方法。"""

    def __init__(self):
        self.data = {}

    async def incr(self, key):
        self.data[key] = self.data.get(key, 0) + 1
        return self.data[key]

    async def expire(self, key, ttl):
        return True

    async def delete(self, key):
        self.data.pop(key, None)
        return 1


@pytest.fixture
def fake_redis(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(chat, "redis_client", fake)
    return fake


def test_no_handoff_signal_returns_none_and_touches_nothing(fake_redis):
    result = asyncio.run(chat.servercustomer("今天天气怎么样", "s1"))

    assert result is None
    assert fake_redis.data == {}


def test_below_three_signals_returns_none(fake_redis):
    sid = "s2"

    assert asyncio.run(chat.servercustomer("我要转人工", sid)) is None
    assert asyncio.run(chat.servercustomer("转人工客服", sid)) is None
    # 窗口内只累计了 2 次,不应转人工
    assert fake_redis.data[f"user:{sid}:message_count"] == 2


def test_third_signal_triggers_handoff_and_resets_window(fake_redis):
    sid = "s3"

    asyncio.run(chat.servercustomer("我要转人工", sid))
    asyncio.run(chat.servercustomer("转人工客服", sid))
    third = asyncio.run(chat.servercustomer("我要找真人处理", sid))

    assert third is not None
    assert "转人工" in third
    # 触发后 key 被删除 → 窗口归零,下一次重新累计
    assert fake_redis.data == {}


def test_window_resets_after_handoff(fake_redis):
    sid = "s4"

    # 第一次窗口:打到 3 次转人工
    for _ in range(3):
        asyncio.run(chat.servercustomer("转人工", sid))
    # 第二次窗口:重新从 1 开始数,3 次后再触发
    asyncio.run(chat.servercustomer("转人工", sid))
    result = asyncio.run(chat.servercustomer("转人工", sid))

    assert result is None  # 第二次窗口只累计到 2
