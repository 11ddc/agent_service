"""认证的**真实集成测试** —— 真 MySQL、真 JWT、真 bcrypt。默认不跑。

    .\\venv\\Scripts\\python.exe -m pytest -m integration -q tests/test_auth_integration.py

## 为什么需要

认证里有几件事只有连真库才验证得动：

- 唯一键冲突 → `DuplicateUsername` 的**竞态兜底**（`mysql_cursor` 会把唯一键冲突
  也收敛成 `MySQLUnavailable`，所以"重名"与"连不上"的区分靠再查一次）；
- 刷新令牌的**原子消费**（`UPDATE ... WHERE revoked_at IS NULL` 的 rowcount）；
- 时间字段用 UTC naive 存进 `datetime` 列后比较是否仍然正确。

单测里存储是内存替身，永远看不到这些。

不访问任何大模型：只打认证接口与一次"越权上传"。
"""

import uuid

import pytest
from fastapi.testclient import TestClient

import main
from db.mysql import MySQLUnavailable, mysql_cursor

pytestmark = pytest.mark.integration

PASSWORD = "Str0ng-Passphrase-42"


@pytest.fixture(scope="module")
def client():
    return TestClient(main.app)


@pytest.fixture
def account(client):
    """注册一个随机账号，用完把它的数据清掉（含审计与刷新令牌）。"""
    username = "itest_" + uuid.uuid4().hex[:10]

    try:
        r = client.post(
            "/api/auth/register", json={"username": username, "password": PASSWORD}
        )
    except MySQLUnavailable as e:  # pragma: no cover
        pytest.skip(f"MySQL 不可用，跳过认证集成测试: {e}")

    if r.status_code == 503:
        pytest.skip("账号存储不可用，跳过认证集成测试")

    assert r.status_code == 201, r.text
    tokens = r.json()

    yield username, tokens

    with mysql_cursor() as cur:
        cur.execute("SELECT user_id FROM users WHERE username = %s", (username,))
        row = cur.fetchone()
        if row:
            cur.execute("DELETE FROM refresh_tokens WHERE user_id = %s", (row[0],))
            cur.execute("DELETE FROM audit_log WHERE actor_id = %s", (row[0],))
            cur.execute("DELETE FROM audit_log WHERE actor_name = %s", (username,))
            cur.execute("DELETE FROM users WHERE user_id = %s", (row[0],))


def _auth(access: str) -> dict:
    return {"Authorization": f"Bearer {access}"}


def test_register_login_and_me(client, account):
    username, tokens = account

    assert tokens["user"]["role"] == "user"

    r = client.post("/api/auth/login", json={"username": username, "password": PASSWORD})
    assert r.status_code == 200, r.text
    access = r.json()["access_token"]

    r = client.get("/api/auth/me", headers=_auth(access))
    assert r.status_code == 200
    assert r.json()["username"] == username


def test_duplicate_username_is_409_not_503(client, account):
    """重名必须映射成 409：如果实现得不好，唯一键冲突会被当成"数据库连不上"（503）。"""
    username, _tokens = account

    r = client.post(
        "/api/auth/register", json={"username": username, "password": PASSWORD}
    )

    assert r.status_code == 409, r.text


def test_protected_endpoints_require_a_token(client):
    for path in ("/api/chat", "/api/chat/stream", "/api/upload"):
        r = client.post(path, json={}) if path != "/api/upload" else client.post(path)
        assert r.status_code == 401, f"{path} -> {r.status_code}"


def test_normal_user_cannot_upload_to_the_knowledge_base(client, account):
    """越权边界：普通用户上传知识库必须 403（知识库内容会进所有人的提示词）。"""
    _username, tokens = account

    r = client.post(
        "/api/upload",
        headers=_auth(tokens["access_token"]),
        files={"file": ("x.pdf", b"%PDF-1.4 minimal", "application/pdf")},
    )

    assert r.status_code == 403, r.text
    assert "权限不足" in r.text


def test_refresh_rotation_and_replay_detection_against_real_db(client, account):
    _username, tokens = account

    r = client.post("/api/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert r.status_code == 200, r.text
    rotated = r.json()["refresh_token"]

    # 原子消费：旧的已经被消费掉
    assert client.post(
        "/api/auth/refresh", json={"refresh_token": tokens["refresh_token"]}
    ).status_code == 401

    # 重放已撤销的令牌 → 撤销该账号全部刷新令牌（连刚换到的那个也失效）
    assert client.post(
        "/api/auth/refresh", json={"refresh_token": rotated}
    ).status_code == 401


def test_logout_invalidates_the_access_token_immediately(client, account):
    _username, tokens = account
    headers = _auth(tokens["access_token"])

    assert client.get("/api/auth/me", headers=headers).status_code == 200

    r = client.post(
        "/api/auth/logout", headers=headers,
        json={"refresh_token": tokens["refresh_token"]},
    )
    assert r.status_code == 200

    assert client.get("/api/auth/me", headers=headers).status_code == 401


def test_lockout_after_repeated_failures_against_real_db(client, account):
    username, _tokens = account

    codes = []
    for _ in range(6):
        codes.append(
            client.post(
                "/api/auth/login",
                json={"username": username, "password": "definitely-wrong"},
            ).status_code
        )

    assert 401 in codes, codes
    assert 429 in codes, codes
    # 锁定后即使密码正确也拒绝
    assert client.post(
        "/api/auth/login", json={"username": username, "password": PASSWORD}
    ).status_code == 429


def test_audit_rows_are_written_without_sensitive_values(client, account):
    """审计必须留痕，且**不能**把密码写进去（detail 会长期留存并被检索）。"""
    username, _tokens = account
    wrong = "sh0uld-never-appear-in-audit"
    client.post("/api/auth/login", json={"username": username, "password": PASSWORD})
    client.post("/api/auth/login", json={"username": username, "password": wrong})

    with mysql_cursor() as cur:
        cur.execute(
            "SELECT action, result, detail FROM audit_log WHERE actor_name = %s",
            (username,),
        )
        rows = cur.fetchall()

    assert rows, "应当写了审计记录"
    dumped = str(rows)
    assert PASSWORD not in dumped
    assert wrong not in dumped
