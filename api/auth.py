"""认证接口：`/api/auth/*`（独立于问答接口）。

| 方法 | 路径 | 说明 | 鉴权 |
|---|---|---|---|
| POST | `/api/auth/register` | 自助注册 | 无（可用 `AUTH_ALLOW_REGISTRATION=0` 关闭） |
| POST | `/api/auth/login` | 登录，返回访问令牌 + 刷新令牌 | 无 |
| POST | `/api/auth/refresh` | 用刷新令牌换新的一对（**轮换**） | 无（凭刷新令牌） |
| POST | `/api/auth/logout` | 登出：撤销当前访问令牌 + 指定刷新令牌 | 需要登录 |
| GET | `/api/auth/me` | 当前身份 | 需要登录 |
| POST | `/api/auth/users` | 管理员建号（可指定角色/租户/客户号） | admin |
| GET | `/api/auth/users` | 管理员看本租户账号列表 | admin |

## 几条不能省的细节

1. **不泄露"用户是否存在"**：登录失败统一返回"用户名或密码不正确"；
   用户不存在时也走一次同代价的密码校验（`dummy_verify`），避免耗时侧信道。
2. **防在线爆破**：按**账号**维度计失败次数（IP 会被 NAT/代理池绕过），
   达到阈值锁定一段时间。
3. **刷新令牌轮换 + 重放检测**：每次 refresh 都签发新令牌并**原子地**消费旧的；
   若收到一个**已撤销**的刷新令牌，说明它可能已被窃取重放 ——
   按安全惯例撤销该账号**全部**刷新令牌，强制重新登录。
4. **自助注册不能自选角色与租户**：否则就是现成的提权与越租户后门。
   要 `kb_admin`/`operator` 账号只能由管理员建（或跑 `python -m auth.bootstrap`）。
5. 密码、令牌明文一律不进日志、响应体与审计 detail。
"""

import logging
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

import config
from auth import audit, revocation
from auth.deps import Principal, Role, require_roles, require_user
from auth.security import (
    create_access_token,
    dummy_verify,
    hash_password,
    hash_token,
    new_refresh_token,
    password_problems,
    verify_password,
)
from db import user_store
from db.mysql import MySQLUnavailable

logger = logging.getLogger(__name__)

router = APIRouter()

# ⚠️ 自助注册只允许拿到这个角色（来自配置的默认值，且不接受客户端指定）
_REGISTER_ROLE = Role.USER


# ── 请求/响应模型 ────────────────────────────────────────────


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=1, max_length=256)
    display_name: str | None = Field(default=None, max_length=64)


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=256)


class RefreshRequest(BaseModel):
    refresh_token: str = Field(..., min_length=8)


class LogoutRequest(BaseModel):
    refresh_token: str | None = None


class AdminCreateUserRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=1, max_length=256)
    role: str = Field(default="user", max_length=20)
    display_name: str | None = Field(default=None, max_length=64)
    customer_id: str | None = Field(default=None, max_length=32)
    tenant_id: str | None = Field(default=None, max_length=32)


class UserOut(BaseModel):
    user_id: str
    username: str
    display_name: str | None = None
    role: str
    tenant_id: str
    customer_id: str | None = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: str
    user: UserOut


# ── 小工具 ───────────────────────────────────────────────────


def _user_out(user) -> UserOut:
    return UserOut(
        user_id=str(user["user_id"]),
        username=str(user["username"]),
        display_name=user.get("display_name"),
        role=str(user["role"]),
        tenant_id=str(user["tenant_id"]),
        customer_id=user.get("customer_id"),
    )


def _store_unavailable(action: str, exc: Exception) -> HTTPException:
    """账号存储不可用 → **503**。

    刻意不用 401：这不是"你没授权"，而是"现在无法验证身份"。
    混用会让客户端误以为该去登录，也让排查时看不出是数据库问题。
    """
    logger.error("%s 失败：账号存储不可用 -> %s", action, exc)
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="账号服务暂时不可用，请稍后重试",
    )


def _issue_tokens(user) -> TokenResponse:
    """签发一对令牌，并把刷新令牌的**哈希**入库（明文只出现在这次响应里）。"""
    access, _jti, _exp = create_access_token(user)
    refresh = new_refresh_token()
    user_store.create_refresh_token(
        token_hash=hash_token(refresh),
        user_id=str(user["user_id"]),
        expires_at=user_store.utcnow()
        + timedelta(seconds=max(60, config.AUTH_REFRESH_TTL)),
    )
    return TokenResponse(
        access_token=access,
        expires_in=config.AUTH_ACCESS_TTL,
        refresh_token=refresh,
        user=_user_out(user),
    )


# ── 端点 ─────────────────────────────────────────────────────


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(payload: RegisterRequest, request: Request) -> TokenResponse:
    """自助注册。角色与租户**都由服务端决定**，不接受客户端指定。"""
    if not config.AUTH_ALLOW_REGISTRATION:
        await audit.record(
            "auth.register", "denied", target=payload.username,
            detail="自助注册已关闭", request=request,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="本系统未开放自助注册，请联系管理员开通账号",
        )

    problems = password_problems(payload.password)
    if problems:
        raise HTTPException(
            status_code=422,  # 422：语义校验失败（用字面量，兼容新旧 FastAPI 常量名）
            detail="密码不符合要求：" + "；".join(problems),
        )

    username = payload.username.strip()
    try:
        user = user_store.create_user(
            username=username,
            password_hash=hash_password(payload.password),
            # ⚠️ 硬编码为最低角色 + 配置里的默认租户：
            #    接受客户端传 role/tenant 就是提权与越租户
            role=_REGISTER_ROLE.value,
            tenant_id=config.AUTH_DEFAULT_TENANT,
            display_name=(payload.display_name or "").strip() or None,
        )
    except user_store.DuplicateUsername:
        await audit.record(
            "auth.register", "denied", target=username,
            detail="用户名已存在", request=request,
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="用户名已存在"
        ) from None
    except MySQLUnavailable as e:
        raise _store_unavailable("注册", e) from e

    await audit.record(
        "auth.register", "ok", actor_id=user["user_id"], actor_name=username,
        tenant_id=user["tenant_id"], target=username, request=request,
    )
    logger.info("新账号注册: %s (%s)", username, _REGISTER_ROLE.value)
    return _issue_tokens(user)


@router.post("/login")
async def login(payload: LoginRequest, request: Request) -> TokenResponse:
    """登录。失败原因一律不区分（不泄露用户是否存在）。"""
    username = payload.username.strip()

    try:
        user = user_store.get_user_by_username(username, config.AUTH_DEFAULT_TENANT)
    except MySQLUnavailable as e:
        raise _store_unavailable("登录", e) from e

    if user is None:
        # 耗时对齐：否则"用户不存在"会明显更快，等于给了枚举接口
        dummy_verify(payload.password)
        await audit.record(
            "auth.login", "failed", target=username,
            detail="用户不存在", request=request,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码不正确"
        )

    if str(user["status"]) != "active":
        await audit.record(
            "auth.login", "denied", actor_id=user["user_id"], actor_name=username,
            tenant_id=user["tenant_id"], target=username,
            detail="账号已禁用", request=request,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="账号已被禁用，请联系管理员"
        )

    locked_until = user.get("locked_until")
    if locked_until is not None and locked_until > user_store.utcnow():
        await audit.record(
            "auth.login", "denied", actor_id=user["user_id"], actor_name=username,
            tenant_id=user["tenant_id"], target=username,
            detail="账号处于锁定窗口内", request=request,
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="账号因多次登录失败被临时锁定，请稍后再试",
        )

    if not verify_password(payload.password, str(user["password_hash"])):
        try:
            failed = user_store.register_login_failure(
                str(user["user_id"]),
                max_failed=config.AUTH_MAX_FAILED_LOGINS,
                lock_seconds=config.AUTH_LOCK_SECONDS,
            )
        except MySQLUnavailable as e:
            raise _store_unavailable("登录失败计数", e) from e
        await audit.record(
            "auth.login", "failed", actor_id=user["user_id"], actor_name=username,
            tenant_id=user["tenant_id"], target=username,
            detail=f"密码错误（累计 {failed} 次）", request=request,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码不正确"
        )

    try:
        user_store.register_login_success(str(user["user_id"]))
    except MySQLUnavailable as e:
        raise _store_unavailable("登录", e) from e

    await audit.record(
        "auth.login", "ok", actor_id=user["user_id"], actor_name=username,
        tenant_id=user["tenant_id"], target=username, request=request,
    )
    return _issue_tokens(user)


@router.post("/refresh")
async def refresh(payload: RefreshRequest, request: Request) -> TokenResponse:
    """用刷新令牌换新的一对。**轮换**：旧的立即作废；重放则撤销该账号全部令牌。"""
    token_hash = hash_token(payload.refresh_token)

    try:
        row = user_store.get_refresh_token(token_hash)
    except MySQLUnavailable as e:
        raise _store_unavailable("刷新令牌", e) from e

    if row is None:
        await audit.record(
            "auth.refresh", "failed", detail="刷新令牌不存在", request=request
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="刷新令牌无效"
        )

    user_id = str(row["user_id"])

    if row["revoked_at"] is not None:
        # 已撤销的令牌又被拿来用 → 可能被窃取重放。按安全惯例：全部踢下线。
        try:
            revoked = user_store.revoke_all_refresh_tokens(user_id)
        except MySQLUnavailable as e:
            raise _store_unavailable("刷新令牌", e) from e
        await audit.record(
            "auth.refresh", "denied", actor_id=user_id,
            detail=f"已撤销的刷新令牌被重放，已撤销该账号全部刷新令牌（{revoked} 个）",
            request=request,
        )
        logger.warning("刷新令牌重放：user=%s，已强制该账号重新登录", user_id)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="刷新令牌已失效，请重新登录",
        )

    if row["expires_at"] <= user_store.utcnow():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="刷新令牌已过期，请重新登录",
        )

    # 原子消费：并发用同一个刷新令牌时只允许一个成功
    try:
        if not user_store.consume_refresh_token(token_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="刷新令牌已被使用，请重新登录",
            )
        user = user_store.get_user_by_id(user_id)
    except MySQLUnavailable as e:
        raise _store_unavailable("刷新令牌", e) from e

    if user is None or str(user["status"]) != "active":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="账号不可用，请联系管理员"
        )

    await audit.record(
        "auth.refresh", "ok", actor_id=user_id, actor_name=user["username"],
        tenant_id=user["tenant_id"], request=request,
    )
    return _issue_tokens(user)


@router.post("/logout")
async def logout(
    payload: LogoutRequest,
    request: Request,
    principal: Principal = Depends(require_user),
) -> dict:
    """登出：撤销当前访问令牌（即时失效）与指定的刷新令牌。"""
    if principal.jti:
        # TTL 取访问令牌的剩余有效期：到期后这条撤销记录本身也会被清掉
        await revocation.revoke_jti(principal.jti, config.AUTH_ACCESS_TTL)

    if payload.refresh_token:
        try:
            user_store.revoke_refresh_token(hash_token(payload.refresh_token))
        except MySQLUnavailable as e:
            # 访问令牌已经撤销（本进程内即时生效），刷新令牌撤销失败只告警
            logger.warning("登出时撤销刷新令牌失败: %s", e)

    await audit.record("auth.logout", "ok", principal=principal, request=request)
    return {"success": True, "detail": "已登出"}


@router.get("/me")
async def me(
    request: Request, principal: Principal = Depends(require_user)
) -> UserOut:
    """当前身份。以**库里的记录**为准（令牌里的角色可能已经变了）。"""
    try:
        user = user_store.get_user_by_id(principal.user_id)
    except MySQLUnavailable as e:
        raise _store_unavailable("查询身份", e) from e

    if user is None or str(user["status"]) != "active":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="账号不存在或已被禁用"
        )
    return _user_out(user)


# ── 管理员接口（RBAC 的落地处）────────────────────────────────


@router.post("/users", status_code=status.HTTP_201_CREATED)
async def admin_create_user(
    payload: AdminCreateUserRequest,
    request: Request,
    principal: Principal = Depends(require_roles("admin")),
) -> UserOut:
    """管理员建号。这是唯一能创建 `kb_admin` / `operator` 的入口。

    自助注册被硬编码为最低角色，所以没有这个接口就**无法产生**任何有权限的账号
    （首个管理员用 `python -m auth.bootstrap` 引导）。
    """
    problems = password_problems(payload.password)
    if problems:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="密码不符合要求：" + "；".join(problems),
        )

    role = str(payload.role or "user").strip().lower()
    if role not in {r.value for r in Role}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"未知角色 {payload.role!r}（可选：{', '.join(r.value for r in Role)}）",
        )

    # 租户默认落在管理员自己的租户：跨租户建号需要显式指定，避免手滑越界
    tenant_id = (payload.tenant_id or principal.tenant_id).strip() or principal.tenant_id
    username = payload.username.strip()

    try:
        user = user_store.create_user(
            username=username,
            password_hash=hash_password(payload.password),
            role=role,
            tenant_id=tenant_id,
            display_name=(payload.display_name or "").strip() or None,
            customer_id=(payload.customer_id or "").strip() or None,
        )
    except user_store.DuplicateUsername:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="用户名已存在"
        ) from None
    except MySQLUnavailable as e:
        raise _store_unavailable("建号", e) from e

    await audit.record(
        "auth.admin_create_user", "ok", principal=principal, target=username,
        detail=f"role={role} tenant={tenant_id}", request=request,
    )
    logger.info("管理员 %s 建号 %s（role=%s）", principal.username, username, role)
    return _user_out(user)


@router.get("/users")
async def admin_list_users(
    request: Request,
    principal: Principal = Depends(require_roles("admin")),
    limit: int = 100,
) -> list[UserOut]:
    """列出**本租户**的账号（跨租户列表要看别的租户，得显式传 tenant）。"""
    try:
        users = user_store.list_users(tenant_id=principal.tenant_id, limit=limit)
    except MySQLUnavailable as e:
        raise _store_unavailable("账号列表", e) from e
    return [_user_out(u) for u in users]
