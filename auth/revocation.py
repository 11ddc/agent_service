"""访问令牌的即时撤销集合（登出 / 禁用账号）。

访问令牌是无状态 JWT：签出去就有效到 `exp`。要做到"登出立即失效"，必须有一个
共享的撤销集合。这里的策略：

- **有 Redis 用 Redis**（多副本共享）：键带 TTL，过期由 Redis 自己清理；
- **没有 Redis / Redis 挂了** → 退化为**进程内**集合，并打一次 WARNING：
  撤销在本进程立即生效，跨副本要等访问令牌自然过期（默认 15 分钟）。

## 为什么检查失败时是"放行"

撤销检查是**额外的即时性**，不是身份校验本身 —— 令牌仍然是合法签名的。
为了 Redis 抖动就让全站 401，可用性代价远大于"某个已登出的令牌多活几分钟"。
这个取舍明确写在这里，便于审计时说明；要更严格就把访问令牌 TTL 调短。
"""

import logging
import time

logger = logging.getLogger(__name__)

_JTI_PREFIX = "auth:revoked:jti:"
_USER_PREFIX = "auth:revoked:user:"

# 进程内兜底：{key: 过期时间戳}
_local: dict[str, float] = {}
_warned = False


def _warn_once(exc: BaseException) -> None:
    global _warned
    if not _warned:
        _warned = True
        logger.warning(
            "Redis 不可用，访问令牌撤销退化为**进程内**："
            "跨副本的即时撤销将失效（令牌仍会在自然过期后失效）。根因 -> %s",
            exc,
        )


def _mark_local(key: str, ttl: int) -> None:
    now = time.time()
    # 顺手清理过期项，避免进程内集合无界增长
    for k in [k for k, exp in _local.items() if exp <= now]:
        _local.pop(k, None)
    _local[key] = now + max(1, ttl)


def _is_local(key: str) -> bool:
    exp = _local.get(key)
    if exp is None:
        return False
    if exp <= time.time():
        _local.pop(key, None)
        return False
    return True


async def _redis_set(key: str, ttl: int) -> bool:
    try:
        from redis_client import redis_client

        # 用 set(..., ex=) 而不是 setex：后者自 redis-py 2.6.12 起已废弃
        await redis_client.set(key, "1", ex=max(1, ttl))
        return True
    except Exception as e:  # noqa: BLE001 - 任何 Redis 问题都不能影响登录/登出主流程
        _warn_once(e)
        return False


async def _redis_exists(key: str) -> bool | None:
    """True/False = Redis 给了答案；None = 问不到，调用方按"放行"处理。"""
    try:
        from redis_client import redis_client

        return bool(await redis_client.exists(key))
    except Exception as e:  # noqa: BLE001
        _warn_once(e)
        return None


async def revoke_jti(jti: str, ttl_seconds: int) -> None:
    """把某个访问令牌加入撤销集合（登出时调用）。ttl 应设成令牌的剩余有效期。"""
    key = _JTI_PREFIX + jti
    _mark_local(key, ttl_seconds)
    await _redis_set(key, ttl_seconds)


async def is_jti_revoked(jti: str) -> bool:
    key = _JTI_PREFIX + jti
    if _is_local(key):
        return True
    return bool(await _redis_exists(key))


async def revoke_user(user_id: str, ttl_seconds: int) -> None:
    """撤销某账号的**全部**访问令牌（禁用账号 / 改密码 / 退出所有设备）。

    这样"禁用"几乎是即时生效的，而不必逐个去追已签发的令牌。
    """
    key = _USER_PREFIX + user_id
    _mark_local(key, ttl_seconds)
    await _redis_set(key, ttl_seconds)


async def is_user_revoked(user_id: str) -> bool:
    key = _USER_PREFIX + user_id
    if _is_local(key):
        return True
    return bool(await _redis_exists(key))


def reset_local() -> None:
    """清空进程内集合（测试用；生产不需要）。"""
    _local.clear()
