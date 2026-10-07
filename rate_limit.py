"""入口限流：按账号的**令牌桶**，Redis 原子执行 + 进程内兜底。

## 为什么是"按账号"而不是"按 IP"

用 IP 限流在真实环境里几乎必然被绕过（NAT/出口代理/代理池），却会误伤同一
出口后面的整个办公室。认证之后我们知道 `principal`，就该按账号限 —— 这也让
"某个账号在刷接口"这件事直接可查。IP 维度只在**未认证接口**上才有意义
（登录爆破防护走的是另一个机制：按账号锁定 + dummy_verify，见 api/auth.py）。

## 为什么是令牌桶而不是固定窗口

固定窗口的经典问题是**边界突发**：窗口末尾打满、下一窗口开头再打满，
瞬时速率是配额的两倍。令牌桶用"桶容量 = 突发额度"直接表达这件事。

## 两种存储后端跑的是**同一个算法**

- Redis（正常）：Lua 脚本一次 EVAL 完成"读-算-写"，多副本部署时共享同一份额度；
- 进程内（Redis 不可用时）：仍然按**单实例**限流，而不是完全不设防。

Redis 出错时降级到进程内并打 WARNING + 指标：既不会因为限流组件故障把整个服务
打成不可用，也不会在故障期间变成"裸奔"。这是刻意的取舍，不是遗漏。

## 与 `X-Forwarded-For` 的关系

本模块不解析任何客户端可伪造的头。要按真实客户端 IP 限流，必须先确认
反向代理可信（见 config.TRUST_PROXY_HEADERS 的说明）。
"""

import logging
import threading
import time
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request, status

import config
from metrics import rate_limited

logger = logging.getLogger(__name__)

# 桶的空闲回收：超过这个时间没被访问的 key 直接从内存里清掉，
# 否则"每个账号一个桶"会随用户数无界增长。
_IDLE_TTL = 3600.0


@dataclass(frozen=True)
class Decision:
    allowed: bool
    remaining: float
    retry_after: float  # 秒；allowed 时为 0
    backend: str = "local"


def refill(
    tokens: float, last_ts: float, now: float, rate: float, burst: float, cost: float = 1.0
) -> tuple[float, float, bool, float]:
    """令牌桶推进一格（**纯函数**，两种后端共用同一份语义）。

    返回 `(剩余令牌, 新的时间戳, 是否放行, 需要等待的秒数)`。

    抽成纯函数是为了能脱离 Redis 精确验证：限流这种东西"差一个边界"就会
    要么误杀要么放行，靠集成测试碰运气不可靠。
    """
    elapsed = max(0.0, now - last_ts)
    tokens = min(burst, tokens + elapsed * rate)
    if tokens >= cost:
        return tokens - cost, now, True, 0.0
    deficit = cost - tokens
    # rate 为 0 时要避免除零：此时唯一出路是等桶被重置（返回一个有限值即可）
    return tokens, now, False, (deficit / rate) if rate > 0 else float(_IDLE_TTL)


# Lua：一次 EVAL 完成"读-算-写"，避免 INCR+EXPIRE 那种分两步的竞态。
# 与 refill() 的语义必须保持一致（同一个公式）。
_BUCKET_LUA = """
local data = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(data[1])
local ts = tonumber(data[2])
local now = tonumber(ARGV[3])
local rate = tonumber(ARGV[1])
local burst = tonumber(ARGV[2])
local cost = tonumber(ARGV[4])
if tokens == nil then tokens = burst end
if ts == nil then ts = now end
local elapsed = now - ts
if elapsed < 0 then elapsed = 0 end
tokens = math.min(burst, tokens + elapsed * rate)
local allowed = 0
local retry = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
else
  if rate > 0 then retry = (cost - tokens) / rate else retry = 3600 end
end
redis.call('HSET', KEYS[1], 'tokens', tokens, 'ts', now)
redis.call('PEXPIRE', KEYS[1], ARGV[5])
return {allowed, tostring(tokens), tostring(retry)}
"""


class _LocalBuckets:
    """进程内令牌桶（兜底 + 测试用）。"""

    def __init__(self, idle_ttl: float = _IDLE_TTL) -> None:
        self._lock = threading.Lock()
        self._state: dict[str, tuple[float, float]] = {}
        self._idle_ttl = idle_ttl

    def consume(
        self, key: str, rate: float, burst: float, now: float | None = None, cost: float = 1.0
    ) -> Decision:
        now = time.monotonic() if now is None else now
        with self._lock:
            self._evict(now)
            tokens, last_ts = self._state.get(key, (burst, now))
            tokens, last_ts, allowed, retry = refill(tokens, last_ts, now, rate, burst, cost)
            self._state[key] = (tokens, last_ts)
        return Decision(allowed=allowed, remaining=tokens, retry_after=retry, backend="local")

    def _evict(self, now: float) -> None:
        if len(self._state) < 1024:
            return
        stale = [k for k, (_t, ts) in self._state.items() if now - ts > self._idle_ttl]
        for key in stale:
            self._state.pop(key, None)

    def reset(self) -> None:
        with self._lock:
            self._state.clear()


class RateLimiter:
    """Redis 优先、进程内兜底。"""

    def __init__(self, redis_client=None, local: _LocalBuckets | None = None) -> None:
        self._redis = redis_client
        self._local = local or _LocalBuckets()
        self._degraded_logged = False

    async def check(self, key: str, *, per_minute: int, burst: int, cost: float = 1.0) -> Decision:
        if not config.RATE_LIMIT_ENABLED:
            return Decision(allowed=True, remaining=float(burst), retry_after=0.0, backend="off")

        rate = max(0.0, per_minute) / 60.0
        burst = max(1, burst)
        ttl_ms = int(max(60.0, burst / rate * 1000 if rate > 0 else 60000))

        client = self._redis
        if client is not None:
            try:
                now = time.time()
                result = await client.eval(
                    _BUCKET_LUA, 1, f"ratelimit:{key}", rate, burst, now, cost, ttl_ms
                )
                allowed = int(result[0]) == 1
                return Decision(
                    allowed=allowed,
                    remaining=float(result[1]),
                    retry_after=float(result[2]),
                    backend="redis",
                )
            except Exception as e:  # noqa: BLE001 - 限流组件不能把主链路带崩
                if not self._degraded_logged:
                    logger.warning(
                        "Redis 限流不可用，降级为进程内限流（单实例有效，多副本会放大）: %r", e
                    )
                    self._degraded_logged = True

        return self._local.consume(key, rate, burst, cost=cost)


_limiter: RateLimiter | None = None


def get_limiter() -> RateLimiter:
    """全局单例；Redis 客户端在首次调用时惰性获取（避免 import 期建连接）。"""
    global _limiter
    if _limiter is None:
        try:
            from redis_client import redis_client as client
        except Exception:  # noqa: BLE001
            client = None
        _limiter = RateLimiter(redis_client=client)
    return _limiter


def reset_limiter() -> None:
    """测试用：丢掉单例（连同进程内的桶状态）。"""
    global _limiter
    _limiter = None


def rate_limit(scope: str, per_minute_attr: str = "RATE_LIMIT_PER_MINUTE",
               burst_attr: str = "RATE_LIMIT_BURST"):
    """生成一个 FastAPI 依赖：超额 → 429 + `Retry-After`。

    用依赖而不是中间件：限流要按**已认证账号**做，而中间件跑在鉴权之前，
    拿不到 principal（在中间件里自己解析令牌等于把鉴权逻辑复制一份）。

    ⚠️ 因此本依赖**要求已认证**：匿名请求会先拿到 401 而不是 429。
    这正是我们要的语义 —— 未认证请求根本不该走到业务逻辑；
    真要按 IP 限流未认证接口，应先能可信地拿到客户端 IP
    （见 config.TRUST_PROXY_HEADERS），那时再加一个匿名版依赖。
    """
    from auth.deps import Principal, require_user

    async def _dependency(
        request: Request, principal: Principal = Depends(require_user)
    ) -> None:
        decision = await get_limiter().check(
            f"{scope}:{principal.user_id}",
            per_minute=getattr(config, per_minute_attr),
            burst=getattr(config, burst_attr),
        )
        if decision.allowed:
            return

        rate_limited().inc(scope=scope)
        wait = max(1, int(decision.retry_after + 0.999))
        logger.warning(
            "触发限流 scope=%s user=%s backend=%s 需等待 %ss",
            scope, principal.user_id, decision.backend, wait,
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"请求过于频繁，请在 {wait} 秒后重试",
            headers={"Retry-After": str(wait)},
        )

    return _dependency
