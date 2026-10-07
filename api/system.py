"""系统自检与指标接口：`/health`、`/ready`、`/metrics`。

## 三个接口定位不同，别混用

| 接口 | 用途 | 失败的含义 | 鉴权 |
|---|---|---|---|
| `GET /health` | 存活探针（liveness） | 进程有问题 → 重启容器 | 无 |
| `GET /ready` | 就绪探针（readiness） | 依赖不可用 → 先别导流量 | 无 |
| `GET /metrics` | 指标抓取（Prometheus 文本） | —— | 默认需 admin |

**探针不能被鉴权挡住**：k8s/负载均衡的探针不会带令牌。代价是"MySQL/Redis 是否
可达"这类信息匿名可见 —— 这是可接受的取舍；真正的敏感面（业务量级、成本）在
`/metrics`，它默认要管理员。

## /health 为什么敢暴露"危险开关"

它会把 AUTH_ENABLED / ACL_ENABLED / RATE_LIMIT_ENABLED 的状态列出来。
这几个开关一旦在生产被关掉，等于"没有鉴权 / 没有权限边界 / 没有限流"。
与其指望有人记得去查配置，不如让**监控直接告警**。它们只是布尔值，
不含密钥、路径等敏感信息。

## 就绪探针**不会**去加载模型

`/ready` 只在向量库已经初始化时才报它的块数；没初始化就报 `not_initialized`。
否则一次探针就会触发 2.4GB 精排模型加载 —— 探针本该是"便宜且频繁"的。
"""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import PlainTextResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import config
from auth.deps import Role, require_user
from metrics import REGISTRY

logger = logging.getLogger(__name__)

router = APIRouter(tags=["系统"])

_STARTED_AT = time.time()

_bearer = HTTPBearer(auto_error=False)


def _uptime_seconds() -> int:
    return int(time.time() - _STARTED_AT)


@router.get("/health")
async def health() -> dict:
    """存活探针：进程活着就 200。同时暴露危险开关，便于监控告警。"""
    switches = {
        "auth_enabled": config.AUTH_ENABLED,
        "acl_enabled": config.ACL_ENABLED,
        "rate_limit_enabled": config.RATE_LIMIT_ENABLED,
        "metrics_require_auth": config.METRICS_REQUIRE_AUTH,
    }
    degraded = [name for name, enabled in switches.items() if not enabled]
    return {
        "status": "ok",
        "version": config.APP_VERSION,
        "uptime_seconds": _uptime_seconds(),
        "switches": switches,
        # 明确标出来：非空的 warnings 应当直接配成告警
        "warnings": (
            [f"以下开关在生产环境应当为 true，当前是 false: {', '.join(degraded)}"]
            if degraded
            else []
        ),
    }


@router.get("/ready")
async def ready(response: Response) -> dict:
    """就绪探针：检查依赖是否可用。

    判定口径：
    - **MySQL 不可用且认证开启** → 503（没人能登录，别导流量）；
    - Redis 不可用 → 仍就绪（会话历史/计数会降级，问答主链路可用）；
    - 向量库未初始化 → 仍就绪（首次上传时才需要，不该让探针触发模型加载）。
    """
    checks: dict[str, dict] = {}

    # MySQL：认证与父块存储都依赖它
    try:
        from db.mysql import ping

        ok, detail = ping()
        checks["mysql"] = {"ok": ok, "detail": detail[:200]}
    except Exception as e:  # noqa: BLE001 - 探针本身绝不能抛
        checks["mysql"] = {"ok": False, "detail": f"{type(e).__name__}: {e}"}

    # Redis：可选依赖
    try:
        from redis_client import redis_client

        await redis_client.ping()
        checks["redis"] = {"ok": True, "detail": "ok"}
    except Exception as e:  # noqa: BLE001
        checks["redis"] = {"ok": False, "detail": f"{type(e).__name__}: {e}"}

    # 向量库：只在已初始化时报告，**不主动加载**
    try:
        import rag.rag as rag_module

        store = rag_module._vectorstore
        if store is None:
            checks["vector_store"] = {"ok": True, "detail": "not_initialized"}
        else:
            checks["vector_store"] = {
                "ok": True,
                "detail": f"chunks={store._collection.count()}",
            }
    except Exception as e:  # noqa: BLE001
        checks["vector_store"] = {"ok": False, "detail": f"{type(e).__name__}: {e}"}

    blocker = None
    if config.AUTH_ENABLED and not checks["mysql"]["ok"]:
        blocker = "MySQL 不可用且认证已开启：无法登录，暂不接流量"

    if blocker:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "ready": blocker is None,
        "version": config.APP_VERSION,
        "uptime_seconds": _uptime_seconds(),
        "checks": checks,
        "blocker": blocker,
    }


@router.get("/metrics")
async def metrics(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> PlainTextResponse:
    """Prometheus 文本格式指标。默认要求 admin（指标会暴露业务量级与成本）。"""
    if config.METRICS_REQUIRE_AUTH:
        # 复用真实的鉴权逻辑（不复制一份令牌校验）：401 / 403 / 503 语义保持一致
        principal = await require_user(credentials)
        if not principal.has(Role.ADMIN):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="指标接口需要 admin 角色"
            )
    return PlainTextResponse(
        REGISTRY.render(), media_type="text/plain; version=0.0.4; charset=utf-8"
    )
