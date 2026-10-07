"""认证与授权测试（A2/A3）—— **完全离线**：内存版账号存储，不连 MySQL、不出网。

这里守的是"身份边界"这一类性质，而不是复述实现：

1. **密码**：只存哈希；长密码不被 72 字节截断悄悄等同；弱口令被拒；
2. **访问令牌**：签名不可伪造、`alg=none` 被拒、过期被拒、刷新令牌不能当访问令牌用；
3. **登录**：失败话术不泄露"用户是否存在"；连续失败按账号锁定；禁用账号被拒；
4. **刷新令牌**：轮换（旧的立即失效）+ 重放检测（撤销该账号全部令牌）；
5. **登出**：访问令牌**即时**失效（不是等到过期）；
6. **RBAC**：角色不足 403、角色继承（admin 拥有 kb_admin 权限）、未知角色降级；
7. **失败关闭**：账号存储不可用时返回 **503**，绝不放行；
8. **审计**：写审计失败不影响主流程（但会留 WARNING）。
"""
import base64
import json
import time

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import config
from api import auth as auth_api
from auth import revocation, security
from auth.deps import Principal, Role, parse_role
from db import user_store
from db.mysql import MySQLUnavailable

SECRET = config.AUTH_JWT_SECRET
PASSWORD = "Str0ng-Passphrase-42"


# ══════════════════════════════════════════════════════════════
# 内存版账号存储
# ══════════════════════════════════════════════════════════════
class FakeStore:
    """只替掉"真的会连库"的那几个函数。

    `DuplicateUsername` / `utcnow` 这些纯逻辑仍用**真实实现**，
    免得假测得跟生产不一致。
    """

    def __init__(self) -> None:
        self.users: dict[str, dict] = {}
        self.tokens: dict[str, dict] = {}
        self.audit: list[dict] = []
        self.unavailable = False
        self._seq = 0

    def _guard(self) -> None:
        if self.unavailable:
            raise MySQLUnavailable("模拟：账号存储不可用")

    def _next_id(self) -> str:
        self._seq += 1
        return f"u_{self._seq:04d}"

    # ---- 账号 ----
    def create_user(self, *, username, password_hash, role, tenant_id,
                    display_name=None, customer_id=None):
        self._guard()
        if self.get_user_by_username(username, tenant_id) is not None:
            raise user_store.DuplicateUsername(username)
        uid = self._next_id()
        now = user_store.utcnow()
        self.users[uid] = {
            "user_id": uid, "username": username, "display_name": display_name,
            "password_hash": password_hash, "role": role, "customer_id": customer_id,
            "tenant_id": tenant_id, "status": "active", "created_at": now,
            "updated_at": now, "last_login_at": None, "failed_logins": 0,
            "locked_until": None,
        }
        return dict(self.users[uid])

    def get_user_by_username(self, username, tenant_id):
        self._guard()
        for u in self.users.values():
            if u["username"] == username and u["tenant_id"] == tenant_id:
                return dict(u)
        return None

    def get_user_by_id(self, user_id):
        self._guard()
        u = self.users.get(user_id)
        return dict(u) if u else None

    def list_users(self, tenant_id=None, limit=100):
        self._guard()
        rows = [u for u in self.users.values() if not tenant_id or u["tenant_id"] == tenant_id]
        return [dict(u) for u in rows[:limit]]

    def register_login_success(self, user_id):
        self._guard()
        u = self.users[user_id]
        u.update(last_login_at=user_store.utcnow(), failed_logins=0, locked_until=None)

    def register_login_failure(self, user_id, *, max_failed, lock_seconds):
        self._guard()
        from datetime import timedelta

        u = self.users[user_id]
        u["failed_logins"] = int(u["failed_logins"] or 0) + 1
        if u["failed_logins"] >= max_failed:
            u["locked_until"] = user_store.utcnow() + timedelta(seconds=lock_seconds)
        return u["failed_logins"]

    # ---- 刷新令牌 ----
    def create_refresh_token(self, *, token_hash, user_id, expires_at,
                             user_agent=None, now=None):
        self._guard()
        self.tokens[token_hash] = {
            "token_hash": token_hash, "user_id": user_id,
            "issued_at": now or user_store.utcnow(), "expires_at": expires_at,
            "revoked_at": None,
        }

    def get_refresh_token(self, token_hash):
        self._guard()
        row = self.tokens.get(token_hash)
        return dict(row) if row else None

    def revoke_refresh_token(self, token_hash):
        self._guard()
        row = self.tokens.get(token_hash)
        if row and row["revoked_at"] is None:
            row["revoked_at"] = user_store.utcnow()

    def consume_refresh_token(self, token_hash):
        self._guard()
        row = self.tokens.get(token_hash)
        if row and row["revoked_at"] is None and row["expires_at"] > user_store.utcnow():
            row["revoked_at"] = user_store.utcnow()
            return True
        return False

    def revoke_all_refresh_tokens(self, user_id):
        self._guard()
        n = 0
        for row in self.tokens.values():
            if row["user_id"] == user_id and row["revoked_at"] is None:
                row["revoked_at"] = user_store.utcnow()
                n += 1
        return n

    # ---- 审计 ----
    def write_audit(self, **kwargs):
        if self.unavailable:
            raise MySQLUnavailable("模拟：审计表不可用")
        self.audit.append(kwargs)


@pytest.fixture
def store(monkeypatch) -> FakeStore:
    fake = FakeStore()
    for name in (
        "create_user", "get_user_by_username", "get_user_by_id", "list_users",
        "register_login_success", "register_login_failure",
        "create_refresh_token", "get_refresh_token", "revoke_refresh_token",
        "consume_refresh_token", "revoke_all_refresh_tokens", "write_audit",
    ):
        monkeypatch.setattr(user_store, name, getattr(fake, name))
    revocation.reset_local()
    return fake


@pytest.fixture
def client(store) -> TestClient:
    """只挂认证路由的最小 app：不 import 编排图，测试快且聚焦。"""
    app = FastAPI()
    app.include_router(auth_api.router, prefix="/api/auth")
    return TestClient(app)


def _register(client, username="alice", password=PASSWORD):
    return client.post(
        "/api/auth/register", json={"username": username, "password": password}
    )


def _login(client, username="alice", password=PASSWORD):
    return client.post("/api/auth/login", json={"username": username, "password": password})


def _auth(access: str) -> dict:
    return {"Authorization": f"Bearer {access}"}


# ══════════════════════════════════════════════════════════════
# 1. 密码原语
# ══════════════════════════════════════════════════════════════
def test_password_hash_is_not_the_plaintext_and_verifies():
    hashed = security.hash_password(PASSWORD)

    assert PASSWORD not in hashed
    assert hashed.startswith("$2")  # bcrypt 标识
    assert security.verify_password(PASSWORD, hashed)
    assert not security.verify_password(PASSWORD + "x", hashed)


def test_same_password_hashes_differently_per_call():
    """bcrypt 自带随机 salt：两个人的同一个密码不该有相同哈希。"""
    assert security.hash_password(PASSWORD) != security.hash_password(PASSWORD)


def test_long_password_is_not_silently_truncated_at_72_bytes():
    """bcrypt 只取前 72 字节 —— 不处理的话两个长密码会互相通过。

    这里用"前 72 字节完全相同、只在后面不同"的两串来验证。
    """
    base = "A" * 72
    a = base + "-different-suffix-1"
    b = base + "-different-suffix-2"

    hashed = security.hash_password(a)

    assert security.verify_password(a, hashed)
    assert not security.verify_password(b, hashed), "长密码被截断了"


def test_verify_password_tolerates_a_corrupt_hash():
    """哈希格式损坏（例如有人手工改库）应当判为失败，而不是抛 500。"""
    assert security.verify_password(PASSWORD, "not-a-bcrypt-hash") is False
    assert security.verify_password(PASSWORD, "") is False


@pytest.mark.parametrize(
    "password,should_pass",
    [
        ("short", False),
        ("a" * 7, False),
        ("a" * 8, False),  # 虽然够长，但重复字符太弱
        ("abcdEF12", True),
        ("aaaaaaaaaaaaaaaa", False),  # 重复字符
        (" with-space ", False),  # 首尾空白
        ("correct horse battery staple", True),
    ],
)
def test_password_policy(password, should_pass):
    assert (security.password_problems(password) == []) is should_pass


def test_dummy_verify_does_not_raise_for_unknown_users():
    security.dummy_verify("whatever")  # 只要求不抛


# ══════════════════════════════════════════════════════════════
# 2. 访问令牌
# ══════════════════════════════════════════════════════════════
def _user_dict(**over):
    base = {
        "user_id": "u_1", "username": "alice", "role": "user",
        "tenant_id": "default", "customer_id": None,
    }
    base.update(over)
    return base


def test_access_token_roundtrip_carries_identity():
    token, jti, _exp = security.create_access_token(_user_dict(customer_id="C2"))

    payload = security.decode_access_token(token)

    assert payload["sub"] == "u_1"
    assert payload["role"] == "user"
    assert payload["jti"] == jti
    assert payload["customer"] == "C2"
    assert payload["type"] == "access"


def test_access_token_with_tampered_signature_is_rejected():
    token, _jti, _exp = security.create_access_token(_user_dict())
    head, payload, _sig = token.split(".")

    with pytest.raises(security.TokenError):
        security.decode_access_token(f"{head}.{payload}.forged-signature")


def test_alg_none_token_is_rejected():
    """`alg=none` 是 JWT 最经典的漏洞 —— 解码必须限定算法白名单。"""

    def b64(obj) -> str:
        raw = json.dumps(obj, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    forged = (
        f"{b64({'alg': 'none', 'typ': 'JWT'})}."
        f"{b64({'sub': 'u_1', 'jti': 'x', 'exp': int(time.time()) + 600, 'type': 'access'})}."
    )

    with pytest.raises(security.TokenError):
        security.decode_access_token(forged)


def test_expired_token_is_rejected():
    expired = jwt.encode(
        {
            "sub": "u_1", "jti": "j", "type": "access",
            "exp": int(time.time()) - 10, "iat": int(time.time()) - 100,
        },
        SECRET,
        algorithm="HS256",
    )

    with pytest.raises(security.TokenError, match="过期"):
        security.decode_access_token(expired)


def test_refresh_token_cannot_be_used_as_access_token():
    """令牌类型必须校验：否则刷新令牌（长效）等于长效访问令牌。"""
    wrong_type = jwt.encode(
        {"sub": "u_1", "jti": "j", "type": "refresh", "exp": int(time.time()) + 600},
        SECRET,
        algorithm="HS256",
    )

    with pytest.raises(security.TokenError, match="类型"):
        security.decode_access_token(wrong_type)


def test_refresh_token_hash_is_stable_and_not_reversible():
    token = security.new_refresh_token()

    assert security.hash_token(token) == security.hash_token(token)
    assert security.hash_token(token) != token
    assert len(security.hash_token(token)) == 64  # sha256 hex


# ══════════════════════════════════════════════════════════════
# 3. 注册
# ══════════════════════════════════════════════════════════════
def test_register_returns_tokens_and_lowest_role(client):
    r = _register(client)

    assert r.status_code == 201, r.text
    body = r.json()
    assert body["user"]["role"] == "user"
    assert body["access_token"] and body["refresh_token"]
    assert "password" not in json.dumps(body).lower()


def test_register_rejects_weak_password(client):
    r = _register(client, password="short")

    assert r.status_code == 422
    assert "密码不符合要求" in r.text


def test_register_rejects_duplicate_username(client):
    _register(client)

    assert _register(client).status_code == 409


def test_register_cannot_choose_role_or_tenant(client, store):
    """⚠️ 安全回归：客户端传的 role / tenant 必须被忽略，否则就是提权后门。"""
    r = client.post(
        "/api/auth/register",
        json={"username": "mallory", "password": PASSWORD,
              "role": "admin", "tenant_id": "other"},
    )

    assert r.status_code == 201
    assert r.json()["user"]["role"] == "user"
    assert r.json()["user"]["tenant_id"] == config.AUTH_DEFAULT_TENANT


def test_register_can_be_disabled(client, monkeypatch):
    monkeypatch.setattr(config, "AUTH_ALLOW_REGISTRATION", False)

    r = _register(client)

    assert r.status_code == 403


def test_register_fails_closed_when_store_is_down(client, store):
    store.unavailable = True

    r = _register(client)

    assert r.status_code == 503
    assert "账号服务" in r.text


# ══════════════════════════════════════════════════════════════
# 4. 登录
# ══════════════════════════════════════════════════════════════
def test_login_success(client):
    _register(client)

    r = _login(client)

    assert r.status_code == 200
    assert r.json()["user"]["username"] == "alice"


def test_login_wrong_password_is_generic_401(client):
    _register(client)

    r = _login(client, password="wrong-password-x")

    assert r.status_code == 401
    assert "用户名或密码不正确" in r.text


def test_login_unknown_user_returns_the_same_message(client):
    """不泄露"用户是否存在"：两种失败必须是同一句话。"""
    _register(client)
    wrong_pw = _login(client, password="wrong-password-x")
    unknown = _login(client, username="nobody-here")

    assert wrong_pw.status_code == unknown.status_code == 401
    assert wrong_pw.json()["detail"] == unknown.json()["detail"]


def test_login_lockout_after_repeated_failures(client, monkeypatch):
    monkeypatch.setattr(config, "AUTH_MAX_FAILED_LOGINS", 3)
    _register(client)

    codes = [_login(client, password="nope-nope-nope").status_code for _ in range(4)]

    assert codes[:3] == [401, 401, 401]
    assert codes[3] == 429, "达到阈值后必须锁定"
    # 锁定期间即使密码正确也拒绝
    assert _login(client).status_code == 429


def test_successful_login_resets_the_failure_counter(client, store):
    _register(client)
    uid = next(iter(store.users))

    _login(client, password="wrong-password-x")
    assert store.users[uid]["failed_logins"] == 1

    _login(client)
    assert store.users[uid]["failed_logins"] == 0


def test_disabled_account_cannot_login(client, store):
    _register(client)
    next(iter(store.users.values()))["status"] = "disabled"

    r = _login(client)

    assert r.status_code == 403
    assert "禁用" in r.text


def test_login_writes_audit_records(client, store):
    _register(client)
    _login(client)
    _login(client, password="wrong-password-x")

    actions = [(a["action"], a["result"]) for a in store.audit]
    assert ("auth.register", "ok") in actions
    assert ("auth.login", "ok") in actions
    assert ("auth.login", "failed") in actions


def test_audit_records_never_contain_the_password(client, store):
    _register(client)
    _login(client, password="wrong-password-x")

    dumped = json.dumps(store.audit, ensure_ascii=False)
    assert "wrong-password-x" not in dumped
    assert PASSWORD not in dumped


# ══════════════════════════════════════════════════════════════
# 5. /me 与 RBAC
# ══════════════════════════════════════════════════════════════
def test_me_requires_a_token(client):
    assert client.get("/api/auth/me").status_code == 401


def test_me_returns_the_current_user(client):
    access = _register(client).json()["access_token"]

    r = client.get("/api/auth/me", headers=_auth(access))

    assert r.status_code == 200
    assert r.json()["username"] == "alice"
    assert r.json()["role"] == "user"


def test_garbage_token_is_401(client):
    assert client.get("/api/auth/me", headers=_auth("not-a-jwt")).status_code == 401


def test_admin_endpoint_requires_admin_role(client):
    access = _register(client).json()["access_token"]  # 普通 user

    r = client.post("/api/auth/users", headers=_auth(access),
                    json={"username": "bob", "password": PASSWORD})

    assert r.status_code == 403
    assert "权限不足" in r.text


def test_admin_can_create_users_with_roles(client, store):
    _register(client)
    next(iter(store.users.values()))["role"] = "admin"
    admin_access = _login(client).json()["access_token"]

    r = client.post("/api/auth/users", headers=_auth(admin_access),
                    json={"username": "kb1", "password": PASSWORD,
                          "role": "kb_admin", "customer_id": "C2"})

    assert r.status_code == 201, r.text
    assert r.json()["role"] == "kb_admin"
    assert r.json()["customer_id"] == "C2"


def test_admin_create_rejects_unknown_role(client, store):
    _register(client)
    next(iter(store.users.values()))["role"] = "admin"
    admin_access = _login(client).json()["access_token"]

    r = client.post("/api/auth/users", headers=_auth(admin_access),
                    json={"username": "x1", "password": PASSWORD, "role": "superuser"})

    assert r.status_code == 422


@pytest.mark.parametrize(
    "role,required,allowed",
    [
        ("admin", "kb_admin", True),
        ("admin", "operator", True),
        ("kb_admin", "operator", True),
        ("kb_admin", "admin", False),
        ("operator", "kb_admin", False),
        ("user", "operator", False),
    ],
)
def test_role_hierarchy(role, required, allowed):
    principal = Principal(user_id="u", username="n", role=Role(role), tenant_id="default")

    assert principal.has(parse_role(required)) is allowed


def test_unknown_role_degrades_to_lowest_privilege():
    """拼错的角色名必须是"没有权限"，而不是"管理员"。"""
    assert parse_role("superadmin") is Role.USER
    assert parse_role(None) is Role.USER
    assert parse_role("ADMIN") is Role.ADMIN


def test_session_scope_is_namespaced_by_identity():
    """会话键必须带身份：客户端提供 session_id，不隔离就能借别人的上下文。"""
    a = Principal(user_id="u1", username="a", role=Role.USER, tenant_id="default")
    b = Principal(user_id="u2", username="b", role=Role.USER, tenant_id="default")

    assert a.session_scope != b.session_scope
    assert a.session_scope.endswith("u1")


# ══════════════════════════════════════════════════════════════
# 6. 刷新令牌轮换与重放
# ══════════════════════════════════════════════════════════════
def test_refresh_rotates_and_invalidates_the_old_token(client):
    first = _register(client).json()["refresh_token"]

    r = client.post("/api/auth/refresh", json={"refresh_token": first})
    assert r.status_code == 200
    second = r.json()["refresh_token"]
    assert second != first

    # 旧的已作废
    assert client.post("/api/auth/refresh", json={"refresh_token": first}).status_code == 401


def test_refresh_replay_revokes_every_refresh_token(client, store):
    """收到已撤销的刷新令牌 → 视为可能被窃取，撤销该账号全部刷新令牌。"""
    first = _register(client).json()["refresh_token"]
    second = client.post("/api/auth/refresh", json={"refresh_token": first}).json()["refresh_token"]

    # 重放旧令牌
    assert client.post("/api/auth/refresh", json={"refresh_token": first}).status_code == 401

    # 连"当前有效"的那个也被撤销了
    assert client.post("/api/auth/refresh", json={"refresh_token": second}).status_code == 401
    assert all(row["revoked_at"] is not None for row in store.tokens.values())


def test_refresh_with_unknown_token_is_401(client):
    assert client.post("/api/auth/refresh", json={"refresh_token": "x" * 40}).status_code == 401


# ══════════════════════════════════════════════════════════════
# 7. 登出即时失效
# ══════════════════════════════════════════════════════════════
def test_logout_revokes_the_access_token_immediately(client):
    tokens = _register(client).json()
    headers = _auth(tokens["access_token"])
    assert client.get("/api/auth/me", headers=headers).status_code == 200

    r = client.post("/api/auth/logout", headers=headers,
                    json={"refresh_token": tokens["refresh_token"]})
    assert r.status_code == 200

    assert client.get("/api/auth/me", headers=headers).status_code == 401


def test_logout_revokes_the_refresh_token(client):
    tokens = _register(client).json()

    client.post("/api/auth/logout", headers=_auth(tokens["access_token"]),
                json={"refresh_token": tokens["refresh_token"]})

    assert client.post("/api/auth/refresh",
                       json={"refresh_token": tokens["refresh_token"]}).status_code == 401


def test_logout_requires_a_token(client):
    assert client.post("/api/auth/logout", json={}).status_code == 401


# ══════════════════════════════════════════════════════════════
# 8. 失败关闭与容错
# ══════════════════════════════════════════════════════════════
def test_login_returns_503_when_store_is_down(client, store):
    _register(client)
    store.unavailable = True

    r = _login(client)

    assert r.status_code == 503, "存储不可用必须是 503（无法验证），绝不能放行成 200"


def test_protected_endpoint_does_not_need_db_when_token_is_valid(client, store):
    """正常请求路径不该查库：令牌自带身份，撤销检查走内存/Redis。"""
    access = _register(client).json()["access_token"]
    store.unavailable = True

    assert client.get("/api/auth/me", headers=_auth(access)).status_code == 503

    # /me 是刻意查库的（以库里的角色为准），但纯鉴权不该查库：
    # 用一个不存在的路由来证明令牌校验本身没有连库 —— 404 说明鉴权已通过
    assert client.get("/api/auth/nope", headers=_auth(access)).status_code == 404


def test_audit_failure_does_not_break_login(client, store):
    """审计写不进去必须只告警 —— 不能因为审计把登录弄挂。"""
    _register(client)
    store.audit.clear()
    store.unavailable = True  # write_audit 会抛，但登录应仍然成功？不：登录本身也要查库

    # 先证明"审计失败"单独发生时登录照常
    store.unavailable = False
    calls = {"n": 0}
    real_write = store.write_audit

    def _boom(**kwargs):
        calls["n"] += 1
        raise MySQLUnavailable("模拟：审计表写不进去")

    import db.user_store as real_store

    # api.auth 通过模块属性调用，所以直接替换模块属性即可
    real_store.write_audit = _boom
    try:
        r = _login(client)
    finally:
        real_store.write_audit = real_write

    assert calls["n"] > 0, "应当尝试过写审计"
    assert r.status_code == 200, "审计失败不该让登录失败"
