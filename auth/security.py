"""密码哈希与令牌签发。

## 密码：bcrypt

- 每个密码独立 salt（bcrypt 自带），验证是常数时间比较；
- ⚠️ bcrypt 只取**前 72 字节**，超长部分被静默丢弃 —— 两个前 72 字节相同的长密码
  会被判定成同一个密码。这里先做 sha256 + base64（固定 44 字节）再交给 bcrypt，
  规避这个坑（Django 也是这么做的）。

## 访问令牌：JWT HS256

- 负载只放必要字段：`sub / name / role / tenant / jti / iat / exp / type`；
- 解码时**显式限定 algorithms=["HS256"]**：不限定的话，`alg=none` 与算法混淆
  是 JWT 最经典的漏洞；
- 短时效（默认 15 分钟），"登出立即失效"靠 `auth.revocation` 的撤销集合。

## 刷新令牌：不透明随机串

刻意**不用 JWT** —— 它必须可撤销。库里只存 sha256 哈希，明文只在响应里出现一次。
这里用 sha256 而不是 bcrypt：令牌本身是 48 字节的高熵随机串，不存在字典攻击面，
bcrypt 只会让每次刷新多花几百毫秒。
"""

import base64
import hashlib
import secrets
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

import config


class TokenError(RuntimeError):
    """访问令牌无效：格式错 / 签名不对 / 过期 / 类型不对。"""


class AuthConfigError(RuntimeError):
    """认证配置缺失（例如没设 AUTH_JWT_SECRET）。"""


# ══════════════════════════════════════════════════════════
# 密码
# ══════════════════════════════════════════════════════════


def _prepare(password: str) -> bytes:
    """把任意长度的密码压成固定 44 字节再喂给 bcrypt（见模块文档的说明）。"""
    return base64.b64encode(hashlib.sha256(password.encode("utf-8")).digest())


def hash_password(password: str, rounds: int | None = None) -> str:
    cost = int(rounds if rounds is not None else config.AUTH_BCRYPT_ROUNDS)
    # 低于 4 无意义；高于 15 会让一次登录慢到不可用（15 约 3~5 秒）
    cost = max(4, min(cost, 15))
    return bcrypt.hashpw(_prepare(password), bcrypt.gensalt(rounds=cost)).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    if not password_hash:
        return False
    try:
        return bcrypt.checkpw(_prepare(password), password_hash.encode("ascii"))
    except (ValueError, TypeError):
        # 哈希格式损坏（例如有人手工改过库）：当作"验证失败"，不要变成 500
        return False


_dummy_hash: str | None = None


def dummy_verify(password: str) -> None:
    """用户不存在时也付出一次同代价的校验。

    不做这一步的话，"用户名不存在"会比"密码错误"返回得**明显更快** ——
    攻击者据此就能枚举出哪些用户名真实存在。
    """
    global _dummy_hash
    if _dummy_hash is None:
        _dummy_hash = hash_password("dummy-password-for-timing-equalisation")
    verify_password(password, _dummy_hash)


def password_problems(password: str, *, min_len: int | None = None) -> list[str]:
    """返回不满足的策略项；空列表表示通过。

    刻意**不**强制"必须同时含大小写字母数字符号"：那类规则会把用户逼成
    `Password1!` 这种可预测口令。真正有效的是长度与明显弱口令。
    """
    pwd = password or ""
    need = int(min_len if min_len is not None else config.AUTH_MIN_PASSWORD_LEN)
    problems: list[str] = []
    if len(pwd) < need:
        problems.append(f"长度至少 {need} 个字符")
    if len(pwd.encode("utf-8")) > 1024:
        problems.append("密码过长（不超过 1024 字节）")
    if pwd and len(set(pwd)) <= 2:
        problems.append("不要用重复字符组成的弱口令")
    if pwd and pwd.strip() != pwd:
        problems.append("首尾不要留空白字符")
    return problems


# ══════════════════════════════════════════════════════════
# 访问令牌（JWT）
# ══════════════════════════════════════════════════════════


# HS256 的密钥就是**签名强度本身**。密钥太短可以被离线暴力破解 ——
# 破解之后任何人都能伪造任意身份的令牌。PyJWT 自己也会对短密钥发警告。
MIN_SECRET_BYTES = 32


def _secret() -> str:
    secret = config.AUTH_JWT_SECRET
    if not secret:
        raise AuthConfigError(
            "未配置 AUTH_JWT_SECRET。生成一个："
            'python -c "import secrets;print(secrets.token_urlsafe(48))"'
        )
    size = len(secret.encode("utf-8"))
    if size < MIN_SECRET_BYTES:
        raise AuthConfigError(
            f"AUTH_JWT_SECRET 太短（{size} 字节，至少 {MIN_SECRET_BYTES} 字节）。"
            "HS256 的密钥长度直接决定签名强度，短密钥可被离线暴力破解 → "
            "破解后可以伪造任意身份的令牌。"
            '生成一个：python -c "import secrets;print(secrets.token_urlsafe(48))"'
        )
    return secret


def create_access_token(user: Mapping[str, object]) -> tuple[str, str, datetime]:
    """返回 (令牌, jti, 过期时间)。jti 用于登出时即时撤销。"""
    now = datetime.now(timezone.utc)
    exp = now + timedelta(seconds=max(60, config.AUTH_ACCESS_TTL))
    jti = secrets.token_urlsafe(16)
    payload = {
        "sub": str(user["user_id"]),
        "name": str(user.get("username") or ""),
        "role": str(user.get("role") or ""),
        "tenant": str(user.get("tenant_id") or config.AUTH_DEFAULT_TENANT),
        "jti": jti,
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
        "type": "access",
    }
    # 业务客户号：订单/工单类工具靠它确定"这是谁的数据"。
    # 放进令牌可以省掉每个请求查一次库；代价是改绑定需要重新登录
    # （可接受，且"禁用账号"另有 user 级撤销兜底）。
    if user.get("customer_id"):
        payload["customer"] = str(user["customer_id"])
    return jwt.encode(payload, _secret(), algorithm="HS256"), jti, exp


def decode_access_token(token: str) -> dict:
    """校验并解出负载；任何问题都抛 TokenError。"""
    try:
        payload = jwt.decode(
            token,
            _secret(),
            algorithms=["HS256"],  # 白名单：防 alg=none 与算法混淆
            options={"require": ["exp", "sub", "jti"]},
        )
    except jwt.ExpiredSignatureError as e:
        raise TokenError("令牌已过期") from e
    except jwt.InvalidTokenError as e:
        raise TokenError(f"令牌无效（{type(e).__name__}）") from e

    if payload.get("type") != "access":
        # 防止把刷新令牌（或将来别的令牌）当访问令牌用
        raise TokenError("令牌类型不对")
    return payload


# ══════════════════════════════════════════════════════════
# 刷新令牌（不透明串）
# ══════════════════════════════════════════════════════════


def new_refresh_token() -> str:
    return secrets.token_urlsafe(48)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
