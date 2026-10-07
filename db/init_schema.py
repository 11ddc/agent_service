"""应用 db/schema.sql 建表（幂等）。

新环境起库只有这一条命令：
    python -m db.init_schema

设计要点：
- **幂等**：DDL 全是 CREATE TABLE IF NOT EXISTS，重复执行安全；
- **不连库不报错**：MySQL 不可用时抛 MySQLUnavailable 并把原因打全（这是初始化
  脚本，用户就是要看到失败原因，不能像检索链路那样静默降级）；
- **顺带自检**：把建完的表、列数、字符集打出来，避免"以为建好了其实没有"。
"""

import sys
from pathlib import Path

from db.mysql import MySQLUnavailable, mysql_cursor

SCHEMA_FILE = Path(__file__).resolve().parent / "schema.sql"


def _statements(sql: str) -> list[str]:
    """把 schema.sql 切成一条条可执行语句。

    先丢掉 `--` 注释行，再按 `;` 切分。schema.sql 由本仓库维护，
    COMMENT 字符串里不含分号，所以这个朴素切法是可靠的。
    """
    lines = []
    for line in sql.splitlines():
        if line.strip().startswith("--"):
            continue
        lines.append(line)
    joined = "\n".join(lines)
    return [s.strip() for s in joined.split(";") if s.strip()]


def apply_schema(sql_file: Path = SCHEMA_FILE) -> list[str]:
    """执行 schema.sql，返回执行过的语句条数说明。"""
    sql = sql_file.read_text(encoding="utf-8")
    stmts = _statements(sql)
    with mysql_cursor() as cur:
        for stmt in stmts:
            cur.execute(stmt)
    return stmts


def verify() -> bool:
    """打印实际落库的表结构，供人工核对（列缺失是会静默降级的那类问题）。

    ⚠️ 返回 bool 而不是 None：以前这个函数把 ok 丢掉、main() 无条件返回 0，
    于是**缺列的库也能让建表脚本"成功退出"** —— CI 与部署脚本都看不出漂移。
    """
    expected = {
        "parents": [
            "parent_id",
            "doc_id",
            "source",
            "source_hash",
            # 这三列是 parent_store 真的会写的（引用可核验、页码定位）——
            # 上一版漏了它们，等于"真实存在的漂移方向恰好检查不到"
            "breadcrumb",
            "page_start",
            "page_end",
            "order_idx",
            "char_len",
            "text",
        ],
        "documents": [
            "doc_id",
            "filename",
            "source",
            "uploaded_at",
            "parsed_status",
            "error_msg",
            "chunk_count",
            "parent_count",
            "chunk_schema_ver",
        ],
        # 认证相关（A2）。漏列的后果同样是静默的：认证链路会连不上账号表，
        # 而表现只是"登录接口 503"，看不出是 schema 漂移。
        "users": [
            "user_id",
            "username",
            "display_name",
            "password_hash",
            "role",
            "customer_id",
            "tenant_id",
            "status",
            "created_at",
            "updated_at",
            "last_login_at",
            "failed_logins",
            "locked_until",
        ],
        "refresh_tokens": [
            "token_hash",
            "user_id",
            "issued_at",
            "expires_at",
            "revoked_at",
            "user_agent",
        ],
        "audit_log": [
            "id",
            "created_at",
            "actor_id",
            "actor_name",
            "tenant_id",
            "action",
            "target",
            "result",
            "detail",
            "request_id",
            "client_ip",
        ],
    }
    with mysql_cursor() as cur:
        cur.execute(
            "SELECT TABLE_NAME, COLUMN_NAME FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA = DATABASE() ORDER BY TABLE_NAME, ORDINAL_POSITION"
        )
        actual: dict[str, list[str]] = {}
        for table, column in cur.fetchall():
            actual.setdefault(table, []).append(column)

    ok = True
    for table, cols in expected.items():
        have = actual.get(table)
        if have is None:
            print(f"[FAIL] 表 {table} 不存在")
            ok = False
            continue
        missing = [c for c in cols if c not in have]
        extra = [c for c in have if c not in cols]
        flag = "OK  " if not missing else "FAIL"
        if missing:
            ok = False
        print(f"[{flag}] {table}: {len(have)} 列", end="")
        if missing:
            print(f" | 缺少代码需要的列: {missing}", end="")
        if extra:
            print(f" | 额外列（可能是预留）: {extra}", end="")
        print()
    print("schema 自检:", "通过" if ok else "未通过")
    return ok


def main() -> int:
    try:
        stmts = apply_schema()
        print(f"已执行 {len(stmts)} 条 DDL（全部 CREATE TABLE IF NOT EXISTS，幂等）")
        # 自检不通过必须体现为**非 0 退出码**，否则 CI/部署脚本看不出 schema 漂移
        return 0 if verify() else 1
    except MySQLUnavailable as e:
        print(f"MySQL 不可用，无法建表: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
