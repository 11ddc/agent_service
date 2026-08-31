"""
会话历史存取 —— Redis（短期记忆）。

设计原则：
- 只存最近 _MAX_MESSAGES 条（8 条 = 4 轮对话），内存有界；
- TTL 过期自动清理，不随用户总量无限增长；
- Redis 异常一律静默降级（读→空列表，写→忽略），不影响主流程。

双客户端设计（重要）：
- 异步场景（api/chat.py）：用项目共享的 redis_client（redis.asyncio），
  在已有事件循环里 await aget_history() / aappend_history()；
- 同步场景（图节点）：graph.invoke 在 asyncio.to_thread 线程里执行，
  那个线程没有事件循环，用独立的**阻塞客户端** _sync_redis
  （redis.Redis），走 get_history() / append_history()。
  不要用 asyncio.run 反复驱动异步客户端——每次 asyncio.run 都会新建并关闭
  一个事件循环，而异步连接池里的连接仍绑定在已关闭的 loop 上（is_connected
  只查 reader/writer 是否非空、不查 loop 状态），下一次调用会报
  "Event loop is closed"，被静默降级成空历史，等于数据丢失。
"""

import json

import redis as redis_sync

from config import REDIS_URL
from redis_client import redis_client  # 异步客户端（async 场景用）

_KEY_PREFIX = "user:{}:history"
_DEFAULT_TTL = 1 * 3600  # 1 小时（如需 48 小时改成 48 * 3600）
_MAX_MESSAGES = 8  # 只保留最近 8 条（4 轮对话）
_MAX_ASSISTANT_LEN = 500  # 助手回答入库时截断长度，防止内存/redis 膨胀

# 同步专用客户端：图节点场景使用（from_url 惰性建连，首次发命令才连 socket）
_sync_redis = redis_sync.Redis.from_url(REDIS_URL, decode_responses=True)


# 截断长度 防止超过500字符（实际上客服场景应该不会遇到）
def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + "…"



async def aget_history(
    session_id: str, max_messages: int = _MAX_MESSAGES
) -> list[dict]:
    """异步读取最近对话历史（async 场景用）；Redis 异常 → 返回空列表。"""
    if not session_id:
        return []
    try:
        raw = await redis_client.get(_KEY_PREFIX.format(session_id))
        if not raw:
            return []
        # 转换成对象
        msgs = json.loads(raw)
        if not isinstance(msgs, list):
            return []

        # 返回后八条消息
        return msgs[-max_messages:] if max_messages else msgs
    except Exception as e:
        print(f"读取会话历史失败，忽略: {e}")
        return []


# 异步写入对话消息
async def aappend_history(
    session_id: str,
    question: str,
    answer: str,
    ttl: int = _DEFAULT_TTL,
) -> None:
    """异步写入一轮对话（user + assistant），保留最近 _MAX_MESSAGES 条并刷新 TTL。"""
    if not session_id:
        return
    try:
        key = _KEY_PREFIX.format(session_id)
        msgs = await aget_history(session_id)

        msgs.append({"role": "user", "content": (question or "").strip()})

        if answer:
            msgs.append(
                {"role": "assistant", "content": _truncate(answer, _MAX_ASSISTANT_LEN)}
            )
        msgs = msgs[-_MAX_MESSAGES:]
        await redis_client.set(key, json.dumps(msgs, ensure_ascii=False), ex=ttl)
    except Exception as e:
        print(f"写入会话历史失败，忽略: {e}")


def get_history(session_id: str, max_messages: int = _MAX_MESSAGES) -> list[dict]:
    """同步读取最近对话历史（图节点用）；Redis 异常 → 返回空列表。"""
    if not session_id:
        return []
    try:
        raw = _sync_redis.get(_KEY_PREFIX.format(session_id))
        if not raw:
            return []
        msgs = json.loads(raw)
        if not isinstance(msgs, list):
            return []
        return msgs[-max_messages:] if max_messages else msgs
    except Exception as e:
        print(f"读取会话历史失败，忽略: {e}")
        return []


def append_history(
    session_id: str, question: str, answer: str, ttl: int = _DEFAULT_TTL
) -> None:
    """同步写入一轮对话（图节点用），保留最近 _MAX_MESSAGES 条并刷新 TTL。"""
    if not session_id:
        return
    try:
        key = _KEY_PREFIX.format(session_id)
        msgs = get_history(session_id)

        msgs.append({"role": "user", "content": (question or "").strip()})

        if answer:
            msgs.append(
                {"role": "assistant", "content": _truncate(answer, _MAX_ASSISTANT_LEN)}
            )
        msgs = msgs[-_MAX_MESSAGES:]
        _sync_redis.set(key, json.dumps(msgs, ensure_ascii=False), ex=ttl)
    except Exception as e:
        print(f"写入会话历史失败，忽略: {e}")
