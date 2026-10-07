"""审计落库的容错封装。

## 原则

**审计失败不能拖垮主流程，但必须留痕。**

写不进库时打 WARNING（带原因），而不是静默吞掉 —— 审计本身失败是需要被发现的
信号："审计表写不进去了"往往意味着数据库出问题或磁盘满了。

## 不要在 detail 里写敏感值

`detail` 会长期留存并被检索。密码、令牌、完整手机号一律不进。
"""
import asyncio
import logging

from db import user_store
from db.mysql import MySQLUnavailable

logger = logging.getLogger(__name__)


def _request_id(request) -> str | None:
    """请求 id 由可观测性中间件写进 request.state（还没做时返回 None）。"""
    return getattr(getattr(request, "state", None), "request_id", None)


def _client_ip(request) -> str | None:
    if request is None:
        return None
    # 延迟 import：避免 auth.audit 与 auth.deps 互相导入
    from auth.deps import client_ip

    try:
        return client_ip(request)
    except Exception:  # noqa: BLE001 - 取 IP 失败不该影响审计
        return None


async def record(
    action: str,
    result: str,
    *,
    request=None,
    principal=None,
    actor_id: str | None = None,
    actor_name: str | None = None,
    tenant_id: str | None = None,
    target: str | None = None,
    detail: str | None = None,
) -> None:
    """写一条审计（非阻塞：同步的 DB 调用丢进线程池）。

    失败只告警，不抛 —— 审计不应该让一次正常的登录/上传失败。
    """
    try:
        await asyncio.to_thread(
            user_store.write_audit,
            action=action,
            result=result,
            actor_id=actor_id
            if actor_id is not None
            else getattr(principal, "user_id", None),
            actor_name=actor_name
            if actor_name is not None
            else getattr(principal, "username", None),
            tenant_id=tenant_id
            if tenant_id is not None
            else getattr(principal, "tenant_id", None),
            target=target,
            detail=detail,
            request_id=_request_id(request),
            client_ip=_client_ip(request),
        )
    except MySQLUnavailable as e:
        logger.warning(
            "审计写入失败（不影响主流程）: action=%s result=%s target=%s 根因=%s",
            action,
            result,
            target,
            e,
        )
    except Exception as e:  # noqa: BLE001 - 审计绝不能反噬主流程
        logger.warning("审计写入异常: action=%s 根因=%s", action, e)
