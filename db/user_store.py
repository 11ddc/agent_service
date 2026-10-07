"""账号 / 刷新令牌 / 审计日志的存取层（MySQL）。

## 三条硬约定

1. **失败即失败**：认证不能降级。MySQL 不可用就抛 `MySQLUnavailable`，由 API 层转成
   503 —— 绝不能"连不上就放行"（那是最典型的"静默失效"）。
2. **只存哈希**：密码存 bcrypt 哈希，刷新令牌存 sha256 哈希。库被读走也不能直接登录。
3. **时间统一 UTC naive**（`datetime.now(timezone.utc)` 去掉 tzinfo）：多副本部署时，
   用本地时区比较"是否过期"会因时区不同而不一致。

> 注意 `db/mysql.py` 的 `mysql_cursor()` 会把块内**任何**异常收敛成 `MySQLUnavailable`，
> 所以"用户名重复"这类业务错误不能靠捕获驱动层异常来区分 —— 见 `create_user` 的做法。
"""

import secrets
from datetime import datetime, timedelta, timezone

from db.mysql import MySQLUnavailable, mysql_cursor


class UserStoreError(RuntimeError):
    """账号层的业务错误（区别于 MySQLUnavailable 的连接/查询失败）。"""


class DuplicateUsername(UserStoreError):
    """同一租户内用户名已存在。"""


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def new_user_id() -> str:
    # 与其它主键一致地用"前缀 + 随机串"，而不是自增：不暴露用户总量、也便于分库
    return "u_" + secrets.token_urlsafe(20)[:26]


_USER_COLUMNS = (
    "user_id, username, display_name, password_hash, role, customer_id, tenant_id, "
    "status, created_at, updated_at, last_login_at, failed_logins, locked_until"
)
_USER_KEYS = [c.strip() for c in _USER_COLUMNS.split(",")]


def _user_row(row) -> dict | None:
    return dict(zip(_USER_KEYS, row)) if row else None


# ── 账号 ─────────────────────────────────────────────────────


def create_user(
    *,
    username: str,
    password_hash: str,
    role: str,
    tenant_id: str,
    display_name: str | None = None,
    customer_id: str | None = None,
) -> dict:
    """建账号。用户名在同一租户内唯一。

    先 SELECT 再 INSERT：`mysql_cursor()` 会把唯一键冲突也收敛成 `MySQLUnavailable`，
    所以只能这样区分"重名"与"连不上"；并发注册同一名字时靠下面的兜底再判一次。
    """
    if get_user_by_username(username, tenant_id) is not None:
        raise DuplicateUsername(f"用户名已存在: {username}")

    user_id = new_user_id()
    now = utcnow()
    try:
        with mysql_cursor() as cur:
            cur.execute(
                "INSERT INTO users (user_id, username, display_name, password_hash, role, "
                "customer_id, tenant_id, status, created_at, updated_at, failed_logins) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, 'active', %s, %s, 0)",
                (
                    user_id,
                    username,
                    display_name,
                    password_hash,
                    role,
                    customer_id,
                    tenant_id,
                    now,
                    now,
                ),
            )
    except MySQLUnavailable:
        # 竞态：另一个并发请求刚建了同名账号。此时唯一键会报错并被收敛成
        # MySQLUnavailable —— 再查一次就能把"重名"和"真的连不上"分开。
        if get_user_by_username(username, tenant_id) is not None:
            raise DuplicateUsername(f"用户名已存在: {username}") from None
        raise

    created = get_user_by_id(user_id)
    if created is None:  # pragma: no cover - 写入成功却读不到，说明库有问题
        raise UserStoreError("账号写入成功但读不回来，请检查数据库连接")
    return created


def get_user_by_username(username: str, tenant_id: str) -> dict | None:
    with mysql_cursor() as cur:
        cur.execute(
            f"SELECT {_USER_COLUMNS} FROM users WHERE tenant_id = %s AND username = %s",
            (tenant_id, username),
        )
        return _user_row(cur.fetchone())


def get_user_by_id(user_id: str) -> dict | None:
    with mysql_cursor() as cur:
        cur.execute(
            f"SELECT {_USER_COLUMNS} FROM users WHERE user_id = %s", (user_id,)
        )
        return _user_row(cur.fetchone())


def list_users(tenant_id: str | None = None, limit: int = 100) -> list[dict]:
    limit = max(1, min(int(limit), 500))
    with mysql_cursor() as cur:
        if tenant_id:
            cur.execute(
                f"SELECT {_USER_COLUMNS} FROM users WHERE tenant_id = %s "
                f"ORDER BY created_at DESC LIMIT {limit}",
                (tenant_id,),
            )
        else:
            cur.execute(
                f"SELECT {_USER_COLUMNS} FROM users ORDER BY created_at DESC LIMIT {limit}"
            )
        return [dict(zip(_USER_KEYS, row)) for row in cur.fetchall()]


def register_login_success(user_id: str) -> None:
    now = utcnow()
    with mysql_cursor() as cur:
        cur.execute(
            "UPDATE users SET last_login_at = %s, failed_logins = 0, locked_until = NULL, "
            "updated_at = %s WHERE user_id = %s",
            (now, now, user_id),
        )


def register_login_failure(user_id: str, *, max_failed: int, lock_seconds: int) -> int:
    """记一次登录失败；达到阈值就锁定账号。返回累计失败次数。

    用**账号维度**的锁定而不是 IP 维度：IP 会被 NAT 与代理池绕过，
    而在线爆破的目标始终是某个具体账号。
    """
    now = utcnow()
    with mysql_cursor() as cur:
        cur.execute("SELECT failed_logins FROM users WHERE user_id = %s", (user_id,))
        row = cur.fetchone()
        failed = (int(row[0]) if row and row[0] is not None else 0) + 1
        locked_until = (
            now + timedelta(seconds=max(0, lock_seconds)) if failed >= max_failed else None
        )
        cur.execute(
            "UPDATE users SET failed_logins = %s, locked_until = %s, updated_at = %s "
            "WHERE user_id = %s",
            (failed, locked_until, now, user_id),
        )
    return failed


# ── 刷新令牌 ─────────────────────────────────────────────────


def create_refresh_token(
    *,
    token_hash: str,
    user_id: str,
    expires_at: datetime,
    user_agent: str | None = None,
    now: datetime | None = None,
) -> None:
    with mysql_cursor() as cur:
        cur.execute(
            "INSERT INTO refresh_tokens (token_hash, user_id, issued_at, expires_at, "
            "user_agent) VALUES (%s, %s, %s, %s, %s)",
            (token_hash, user_id, now or utcnow(), expires_at, (user_agent or None)),
        )


def get_refresh_token(token_hash: str) -> dict | None:
    with mysql_cursor() as cur:
        cur.execute(
            "SELECT token_hash, user_id, issued_at, expires_at, revoked_at "
            "FROM refresh_tokens WHERE token_hash = %s",
            (token_hash,),
        )
        row = cur.fetchone()
    if not row:
        return None
    keys = ["token_hash", "user_id", "issued_at", "expires_at", "revoked_at"]
    return dict(zip(keys, row))


def revoke_refresh_token(token_hash: str) -> None:
    """撤销单个令牌。已经撤销过就保持原时间（幂等，便于排查"最早何时失效"）。"""
    with mysql_cursor() as cur:
        cur.execute(
            "UPDATE refresh_tokens SET revoked_at = %s "
            "WHERE token_hash = %s AND revoked_at IS NULL",
            (utcnow(), token_hash),
        )


def consume_refresh_token(token_hash: str) -> bool:
    """**原子地**消费一个刷新令牌（轮换用）。返回 True 表示这次调用抢到了它。

    为什么不能"先查再改"：两个并发请求同时拿同一个刷新令牌来换新令牌时，
    先查后改会让**两个都通过**。这里用一条带条件的 UPDATE 让数据库来判唯一赢家
    （`rowcount == 1` 才说明这次真的消费成功）。
    """
    with mysql_cursor() as cur:
        cur.execute(
            "UPDATE refresh_tokens SET revoked_at = %s "
            "WHERE token_hash = %s AND revoked_at IS NULL AND expires_at > %s",
            (utcnow(), token_hash, utcnow()),
        )
        return int(cur.rowcount or 0) == 1


def revoke_all_refresh_tokens(user_id: str) -> int:
    """撤销某账号全部未撤销的刷新令牌（改密码 / 禁用账号 / "退出所有设备"）。"""
    with mysql_cursor() as cur:
        cur.execute(
            "UPDATE refresh_tokens SET revoked_at = %s "
            "WHERE user_id = %s AND revoked_at IS NULL",
            (utcnow(), user_id),
        )
        return int(cur.rowcount or 0)


def purge_expired_refresh_tokens(keep_days: int = 1) -> int:
    """清理过期令牌（保留一天便于排查）。应由运维定时任务调用。"""
    cutoff = utcnow() - timedelta(days=max(0, keep_days))
    with mysql_cursor() as cur:
        cur.execute("DELETE FROM refresh_tokens WHERE expires_at < %s", (cutoff,))
        return int(cur.rowcount or 0)


# ── 审计日志 ─────────────────────────────────────────────────


def write_audit(
    *,
    action: str,
    result: str,
    actor_id: str | None = None,
    actor_name: str | None = None,
    tenant_id: str | None = None,
    target: str | None = None,
    detail: str | None = None,
    request_id: str | None = None,
    client_ip: str | None = None,
    now: datetime | None = None,
) -> None:
    """写一条审计。**调用方必须容忍它失败**（见 auth/audit.py 的封装）。

    企业要能回答"谁在什么时候做了什么、结果如何"，这张表就是那个答案。
    detail 里严禁写密钥、完整手机号等敏感值。
    """
    with mysql_cursor() as cur:
        cur.execute(
            "INSERT INTO audit_log (created_at, actor_id, actor_name, tenant_id, action, "
            "target, result, detail, request_id, client_ip) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                now or utcnow(),
                actor_id,
                actor_name,
                tenant_id,
                action,
                (target or None),
                result,
                (detail[:500] if detail else None),
                request_id,
                client_ip,
            ),
        )
