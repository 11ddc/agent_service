#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════
# 首次部署：把 .env 准备好（幂等，可以反复执行）
#
#     python3 deploy/init-env.py
#
# 做三件事，**都不会覆盖你已经填好的值**：
#   1) .env 不存在就从 .env.example 复制一份
#   2) AUTH_JWT_SECRET 还是占位符/为空 → 生成一把 64 字符随机密钥写进去
#      （这是唯一一个"不填就启动失败"的变量，见 auth/security.py:113）
#   3) 补齐 docker-compose.yml 强制要求的 MYSQL_ROOT_PASSWORD / MYSQL_PASSWORD
#      / MYSQL_DATABASE / MYSQL_USER / FRONTEND_DIR
#
# 为什么这些不写进 .env.example 当默认值：compose 用的是 `${VAR:?}` 语法，
# 只要变量"有值"（哪怕是个占位符）它就放行，于是服务带着谁都知道的口令启动。
# 真正的值必须在目标机器上现生成 —— 所以模板里它们是注释状态，由本脚本填。
#
# 本脚本**不打印任何密钥值**：终端回滚缓冲、CI 日志里都不会留下痕迹。
#
# 只用标准库，无第三方依赖。服务器上没有 python3 的话，照着 .env.example 里
# 【Docker Compose 部署】那一段手动填也一样（就 5 行）。
# ════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import os
import re
import secrets
import sys
from pathlib import Path

# Windows 控制台默认可能是 GBK，打印 ✓/中文时抛 UnicodeEncodeError 会把脚本整死。
# 换编码会变乱码，所以只放宽错误处理：显示成 ? 也比中途崩掉强。
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass

# 模板里那个"看起来像密钥"的占位符。它 12 个汉字 = 36 字节，能通过应用
# "≥32 字节"的校验，所以不能指望启动时报错来发现它 —— 必须主动识别并替换。
PLACEHOLDER = "请替换成随机生成的长密钥"

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / ".env.example"
ENV_FILE = ROOT / ".env"

# 只要求平台密钥已填的清单（不填不会启动失败，但主链路会降级）
PLATFORM_KEYS = (
    "DEEPSEEK_API_KEY",
    "GENERATE_API_KEY",
    "QIAN_WEN_QUERYSTION_API_KEY",
    "QIANWEN_API_KEY",
    "ZHI_PU_API_KEY",
)


def read_env() -> str:
    """按字节读 + UTF-8 解码：不经 universal-newlines，CRLF 原样保留。"""
    return ENV_FILE.read_bytes().decode("utf-8")


def write_env(text: str) -> None:
    """UTF-8 无 BOM 写回；先写临时文件再替换，避免写一半把配置弄坏。"""
    tmp = ENV_FILE.parent / (ENV_FILE.name + ".tmp")
    tmp.write_bytes(text.encode("utf-8"))
    tmp.replace(ENV_FILE)


def get_val(text: str, key: str) -> str | None:
    """取某个键的当前值；键不存在返回 None（注意与"存在但为空"区分）。

    `[^\\r\\n]*` 而不是 `.*`：否则会把行尾的 \\r 一起吃进值里，
    拿去做长度判断或与占位符比较就会莫名其妙地不相等。
    """
    m = re.search(rf"(?m)^{re.escape(key)}=([^\r\n]*)", text)
    if m is None:
        return None
    return m.group(1).strip().strip('"').strip("'")


def upsert(text: str, key: str, value: str) -> str:
    """替换已有的键；没有就追加。用 count=1，不会误伤同名的后续行。"""
    pattern = re.compile(rf"(?m)^{re.escape(key)}=[^\r\n]*")
    if pattern.search(text):
        return pattern.sub(f"{key}={value}", text, count=1)
    if text and not text.endswith("\n"):
        text += "\n"
    return f"{text}{key}={value}\n"


def main() -> int:
    if not TEMPLATE.exists():
        print(f"✗ 找不到 {TEMPLATE} —— 请在仓库根目录执行本脚本", file=sys.stderr)
        return 1

    if not ENV_FILE.exists():
        ENV_FILE.write_bytes(TEMPLATE.read_bytes())
        print(f"✓ 已从 {TEMPLATE.name} 生成 {ENV_FILE.name}")
    else:
        print(f"· {ENV_FILE.name} 已存在，只补齐缺失项（不动已有值）")

    text = read_env()

    # ── AUTH_JWT_SECRET：唯一一个"不填就起不来"的变量 ──────────────────
    current = get_val(text, "AUTH_JWT_SECRET")
    if not current or current == PLACEHOLDER:
        text = upsert(text, "AUTH_JWT_SECRET", secrets.token_urlsafe(48))
        print("✓ AUTH_JWT_SECRET 已生成新的 64 字符随机密钥并写入")
    else:
        print(f"· AUTH_JWT_SECRET 已有值（{len(current)} 字符），保持不动")

    # ── compose 强制要求的其余变量 ────────────────────────────────────
    # 每条都是"缺了才补"，所以重复执行不会产生重复行、也不会改掉已生效的口令。
    wanted: list[tuple[str, str]] = [
        ("MYSQL_ROOT_PASSWORD", secrets.token_urlsafe(24)),
        ("MYSQL_DATABASE", "rag"),
        ("MYSQL_USER", "app"),
        ("MYSQL_PASSWORD", secrets.token_urlsafe(24)),
        # 前端是**独立仓库**，默认约定 clone 成后端仓库的隔壁目录。
        # 位置不同就先用 FRONTEND_DIR=... 环境变量覆盖，或事后直接改 .env。
        ("FRONTEND_DIR", os.environ.get("FRONTEND_DIR", "../agent_vue")),
    ]

    appended: list[str] = []
    for key, value in wanted:
        if get_val(text, key) is not None:
            print(f"· {key} 已存在，保持不动")
        else:
            appended.append(f"{key}={value}")
            print(f"✓ 已写入 {key}")

    if appended:
        if text and not text.endswith("\n"):
            text += "\n"
        block = "\n# ── 以下由 deploy/init-env.py 生成 ──\n" + "\n".join(appended) + "\n"
        text += block

    write_env(text)

    # ── 体检：还有哪些占位符没填 ──────────────────────────────────────
    print("\n── 仍需你手动填写的项（脚本无法代生成，要去平台申请）──")
    pending = [
        key
        for key in PLATFORM_KEYS
        if not (v := get_val(text, key)) or "xxxx" in v.lower()
    ]
    if not pending:
        print("  ✓ 平台密钥看起来都已填写")
    else:
        for key in pending:
            print(f"  ✗ {key}  （还是占位符或为空）")
        print(f"\n  → 共 {len(pending)} 项待填。不填不会导致启动失败（主链路会降级），")
        print(f"     但答案质量会明显下降。编辑：nano {ENV_FILE}")

    print("\n── 下一步 ──")
    print("  1) compose 配置自检：     docker compose config --quiet && echo 配置 OK")
    print("  2) 建挂载目录并改属主（容器以 uid 10001 运行，漏了首次上传会 EACCES）：")
    print("       mkdir -p knowledge_base chroma_db")
    print("       sudo chown -R 10001:10001 knowledge_base chroma_db")
    print("  3) 构建并启动：           docker compose up -d --build")
    print("  4) 三条初始化：见 docker-compose.yml 头部注释")
    print(f"\n  想看一眼密钥（别外传）：  grep AUTH_JWT_SECRET {ENV_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
