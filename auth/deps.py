"""FastAPI 依赖：把请求变成 `Principal`，并做 RBAC 判定。

## 三种失败必须分清（排查线上问题全靠这个区别）

| 状态码 | 含义 | 客户端该做什么 |
|---|---|---|
| **401** | 没带令牌 / 令牌无效或过期 / 已登出 | 去登录或刷新令牌 |
| **403** | 身份有效，但角色不够 | 别再重试，找管理员要权限 |
| **503** | 认证配置缺失，或用户存储不可用 | 不是"没授权"，而是"现在**无法验证**" —— 绝不能降级成放行 |

## 角色继承

`admin` ⊃ `kb_admin` ⊃ `operator` ⊃ `user`（左边天然拥有右边的权限）。
未知角色一律降级为 `user`：宁可少给权限，也不要让一个拼错的角色名变成管理员。
"""

import logging
from dataclasses import dataclass
from enum import Enum

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import config
from auth import revocation
from auth.security import AuthConfigError, TokenError, decode_access_token

logger = logging.getLogger(__name__)


class Role(str, Enum):
    ADMIN = "admin"  # 超管：账号、知识库、运营全权
    KB_ADMIN = "kb_admin"  # 知识库管理：上传、审核、发布
    OPERATOR = "operator"  # 坐席：看工单、看会话、看指标
    USER = "user"  # 终端用户：只能问答


_ROLE_GRANTS: dict[Role, frozenset[Role]] = {
    Role.ADMIN: frozenset({Role.ADMIN, Role.KB_ADMIN, Role.OPERATOR, Role.USER}),
    Role.KB_ADMIN: frozenset({Role.KB_ADMIN, Role.OPERATOR, Role.USER}),
    Role.OPERATOR: frozenset({Role.OPERATOR, Role.USER}),
    Role.USER: frozenset({Role.USER}),
}


def parse_role(raw: object) -> Role:
    """把字符串解析成角色；**认不出来就降级为最低权限**。"""
    try:
        return Role(str(raw or "").strip().lower())
    except ValueError:
        logger.warning("未知角色 %r，按最低权限 user 处理", raw)
        return Role.USER


@dataclass(frozen=True)
class Principal:
    """已认证的调用方身份。"""

    user_id: str
    username: str
    role: Role
    tenant_id: str
    # 业务客户号（账号在业务系统里的对应客户）。
    # 它是"认证层 → 业务数据"的桥：订单/工单类工具靠它确定"这是谁的数据"。
    customer_id: str | None = None
    jti: str | None = None
    anonymous: bool = False

    def has(self, *roles: Role) -> bool:
        granted = _ROLE_GRANTS[self.role]
        return any(r in granted for r in roles)

    @property
    def session_scope(self) -> str:
        """会话历史的命名空间前缀。

        ⚠️ 必须带身份：`session_id` 是**客户端提供**的，不隔离的话猜到别人的
        session_id 就能借上下文答出别人的事（并污染对方的计数窗口）。
        """
        return f"{self.tenant_id}:{self.user_id}"


_bearer = HTTPBearer(auto_error=False)

_UNAUTHORIZED_HEADERS = {"WWW-Authenticate": "Bearer"}


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers=_UNAUTHORIZED_HEADERS,
    )


async def require_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> Principal:
    """需要登录的接口用它。失败语义见模块文档。"""
    if not config.AUTH_ENABLED:
        # 显式的本地演示逃生门：启动时已打 WARNING，/health 里也会暴露，
        # 便于用监控告警盯住"生产忘了开认证"。
        return Principal(
            user_id="anonymous",
            username="anonymous",
            role=Role.ADMIN,
            tenant_id=config.AUTH_DEFAULT_TENANT,
            anonymous=True,
        )

    if credentials is None or not credentials.credentials:
        raise _unauthorized("缺少访问令牌（Authorization: Bearer <token>）")

    try:
        payload = decode_access_token(credentials.credentials)
    except AuthConfigError as e:
        # 服务端配置问题 → 503，不是 401：别让客户端以为是自己的令牌有问题
        logger.error("认证配置缺失，无法校验令牌: %s", e)
        raise HTTPException(status_code=503, detail="认证服务未正确配置") from e
    except TokenError as e:
        raise _unauthorized(str(e)) from e

    user_id = str(payload.get("sub") or "")
    jti = str(payload.get("jti") or "")
    if not user_id or not jti:
        raise _unauthorized("令牌缺少必要字段")

    # 即时撤销：登出 / 禁用账号之后，已签发的令牌立刻失效（否则要等到 exp）
    if await revocation.is_jti_revoked(jti) or await revocation.is_user_revoked(user_id):
        raise _unauthorized("令牌已失效（已登出或被撤销）")

    customer = payload.get("customer")
    return Principal(
        user_id=user_id,
        username=str(payload.get("name") or ""),
        role=parse_role(payload.get("role")),
        tenant_id=str(payload.get("tenant") or config.AUTH_DEFAULT_TENANT),
        customer_id=str(customer) if customer else None,
        jti=jti,
    )


def require_roles(*roles: Role | str):
    """生成一个"要求指定角色之一"的依赖。角色继承由 `Principal.has` 处理。

    用法：`principal: Principal = Depends(require_roles("kb_admin"))`
    """
    required = [r if isinstance(r, Role) else parse_role(r) for r in roles]

    async def _dependency(principal: Principal = Depends(require_user)) -> Principal:
        if not principal.has(*required):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"权限不足：需要 {' / '.join(r.value for r in required)}，"
                    f"当前角色是 {principal.role.value}"
                ),
            )
        return principal

    return _dependency


def client_ip(request: Request) -> str | None:
    """取调用方 IP。

    默认**只信** socket 对端（`request.client.host`）：X-Forwarded-For 是客户端
    可伪造的，只有确实部署在可信反向代理之后才应该采信 ——
    所以用 `TRUST_PROXY_HEADERS=1` 显式打开。
    """
    if config.TRUST_PROXY_HEADERS:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip() or None
    return request.client.host if request.client else None
