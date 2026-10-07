"""
Redis 客户端单例。

模块加载时创建一次、全局共享（类似 Hyperf 的 Redis 实例化即用）。
连接是惰性的：from_url 只建连接池，首次发命令才真正连 socket。
进程退出时由 OS 自动回收连接，无需显式关闭；单进程部署足够。
"""
import redis.asyncio as redis

from config import REDIS_URL

# ⚠️ **必须设超时**：redis 客户端默认没有任何 socket 超时。
# 一次"半开连接"（对端被 kill、网络设备静默丢包）会让
# `await redis_client.incr(...)`（转人工计数）或会话历史读写**永久挂住** ——
# 而它们都在请求路径上，挂住就等于这个请求永远不返回。
# 宁可快速失败并降级（不写计数、历史为空），也不要挂死。
redis_client = redis.from_url(
    REDIS_URL,
    decode_responses=True,
    socket_connect_timeout=2,
    socket_timeout=2,
)
