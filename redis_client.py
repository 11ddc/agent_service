"""
Redis 客户端单例。

模块加载时创建一次、全局共享（类似 Hyperf 的 Redis 实例化即用）。
连接是惰性的：from_url 只建连接池，首次发命令才真正连 socket。
进程退出时由 OS 自动回收连接，无需显式关闭；单进程部署足够。
"""
import redis.asyncio as redis

from config import REDIS_URL

redis_client = redis.from_url(REDIS_URL, decode_responses=True)
