"""引导第一个管理员：`python -m auth.bootstrap`。

## 为什么必须有

自助注册被**硬编码为最低角色**（防止提权），而"管理员建号"接口又要求管理员身份 ——
没有引导脚本就永远造不出第一个有权限的账号。

## 用法

    python -m auth.bootstrap --username admin --password <密码>
    # 更推荐：密码走环境变量，避免进 shell 历史与进程列表
    BOOTSTRAP_ADMIN_PASSWORD=<密码> python -m auth.bootstrap --username admin

    # 账号已存在时默认**不动它**（避免误改线上账号）；要提权才加 --promote

    python -m auth.bootstrap --username admin --promote

幂等、可重复执行。MySQL 不可用时明确失败并给出原因（这是引导脚本，
用户就是要看到失败原因，不能像检索链路那样静默降级）。
"""

import argparse
import os
import sys

import config
from auth.security import hash_password, password_problems
from db import user_store
from db.mysql import MySQLUnavailable

_PASSWORD_ENV = "BOOTSTRAP_ADMIN_PASSWORD"


def _read_password(args) -> str:
    if args.password:
        return args.password
    from_env = (os.getenv(_PASSWORD_ENV) or "").strip()
    if from_env:
        return from_env
    # 从终端读，且不回显
    import getpass

    return getpass.getpass("请输入管理员密码: ")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="引导/提权一个管理员账号")
    parser.add_argument("--username", required=True, help="管理员登录名")
    parser.add_argument("--password", default=None, help=f"密码（或用 {_PASSWORD_ENV}）")
    parser.add_argument("--display-name", default="管理员")
    parser.add_argument("--tenant-id", default=config.AUTH_DEFAULT_TENANT)
    parser.add_argument(
        "--promote",
        action="store_true",
        help="账号已存在时也把它提升为 admin（默认不动已存在的账号）",
    )
    args = parser.parse_args(argv)

    try:
        existing = user_store.get_user_by_username(args.username, args.tenant_id)
    except MySQLUnavailable as e:
        print(f"MySQL 不可用，无法引导管理员: {e}", file=sys.stderr)
        return 1

    if existing is not None:
        if not args.promote:
            print(
                f"账号 {args.username!r} 已存在（role={existing['role']}）。"
                f"如需提升为 admin，请加 --promote",
                file=sys.stderr,
            )
            return 1
        changed = _promote(existing)
        print(f"账号 {args.username!r} 已提升为 admin（原 role={existing['role']}）")
        return 0 if changed else 1

    password = _read_password(args)
    problems = password_problems(password)
    if problems:
        print("密码不符合要求：" + "；".join(problems), file=sys.stderr)
        return 1

    try:
        user = user_store.create_user(
            username=args.username,
            password_hash=hash_password(password),
            role="admin",
            tenant_id=args.tenant_id,
            display_name=args.display_name,
        )
    except MySQLUnavailable as e:
        print(f"建号失败: {e}", file=sys.stderr)
        return 1

    print(f"已创建管理员 {user['username']}（user_id={user['user_id']}）")
    print("请立即用 /api/auth/login 登录并确认，然后按需创建其它账号。")
    return 0


def _promote(user: dict) -> bool:
    """把已有账号提升为 admin。走一条显式 SQL，避免为此在存储层开一个通用更新口。"""
    from db.mysql import mysql_cursor

    try:
        with mysql_cursor() as cur:
            cur.execute(
                "UPDATE users SET role = 'admin', updated_at = %s WHERE user_id = %s",
                (user_store.utcnow(), user["user_id"]),
            )
    except MySQLUnavailable as e:
        print(f"提权失败: {e}", file=sys.stderr)
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
