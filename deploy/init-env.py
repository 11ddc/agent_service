#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════
# 首次部署：把 .env 准备好（幂等，可以反复执行）
#
#     python3 deploy/init-env.py
#
# 做三件事，**都不会覆盖你已经填好的值**：
#   1) .env 不存在就从 .env.example 复制一份
#   2) AUTH_JWT_SECRET 还是占位符/为空 → 生成一把 64 字符随机密钥写进去
#      （登录令牌的签名密钥。缺失或太短时**不是**启动失败，而是签/验令牌时抛
#        AuthConfigError —— 服务能起来，但登录与所有鉴权接口全废）
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
import stat
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

# 与 auth/security.py 的 MIN_SECRET_BYTES 保持一致。短于这个值的密钥不会被
# "缺失检查"拦住（它有值），但服务一旦签/验令牌就抛 AuthConfigError ——
# 属于"看起来配好了、一用就炸"，所以这里按同一标准主动重生成。
MIN_SECRET_BYTES = 32

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / ".env.example"
ENV_FILE = ROOT / ".env"

# 平台密钥体检表：(环境变量, 缺失是否致命, 用途)
#
# "致命"不是猜的：这几个模块都在**模块级**构造 OpenAI 客户端
#     agent/langchina.py:25            → DEEPSEEK_API_KEY
#     intent/problemdecomposition.py:31 → ZHI_PU_API_KEY
#     tools_agent/tool_llm.py:31       → ZHI_PU_API_KEY
# 而 openai SDK 在 api_key 为 None 时**构造即抛** OpenAIError，所以缺了不是
# "功能降级"，是 import 失败 → 容器启动即崩。
PLATFORM_KEYS = (
    ("DEEPSEEK_API_KEY", True, "改写 / 意图仲裁 / 兜底 Agent / RAG 答案生成"),
    ("ZHI_PU_API_KEY", True, "问题拆分 + 视觉 OCR + 工具调用（同一把 key）"),
    ("QIANWEN_API_KEY", False, "云端向量（EMBEDDING_PROVIDER=dashscope 才用）；默认走本地"),
)

# 当前代码**不读**的历史变量：填了不生效，别为此去申请 key。
#   BASE_URL                    —— 全仓库没有任何 os.getenv("BASE_URL")
#   GENERATE_API_KEY            —— 答案生成改用 DeepSeek 后废弃（rag/generatellm.py
#                                  用的是 DEEPSEEK_API_KEY，模型默认 deepseek-chat）
#   QIAN_WEN_QUERYSTION_API_KEY —— 问题拆分改用智谱后废弃（intent/problemdecomposition.py
#                                  现在读 ZHI_PU_API_KEY）
DEAD_KEYS = ("GENERATE_API_KEY", "BASE_URL", "QIAN_WEN_QUERYSTION_API_KEY")


def read_env() -> str:
    """按字节读 + UTF-8 解码：不经 universal-newlines，CRLF 原样保留。"""
    return ENV_FILE.read_bytes().decode("utf-8")


def write_env(text: str) -> None:
    """UTF-8 无 BOM 写回；先写临时文件再替换，避免写一半把配置弄坏。

    ⚠️ 权限必须显式处理：`Path.replace()` 之后目标文件继承的是**临时文件**的权限，
    而 `write_bytes` 按 umask 创建（通常 644）。`.env` 里是真实 API key，
    在多人共用的服务器上 644 意味着任何本地用户都能读走它。所以：
        已存在 → 原样沿用它的权限（`chmod 600 .env` 之后不会被悄悄改宽）
        新建   → 直接给 600
    """
    tmp = ENV_FILE.parent / (ENV_FILE.name + ".tmp")
    mode = stat.S_IMODE(ENV_FILE.stat().st_mode) if ENV_FILE.exists() else 0o600
    tmp.write_bytes(text.encode("utf-8"))
    os.chmod(tmp, mode)
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
        # 模板本身可以公开（值全是占位符），但复制出来的 .env 会装真实 key
        os.chmod(ENV_FILE, 0o600)
        print(f"✓ 已从 {TEMPLATE.name} 生成 {ENV_FILE.name}（权限 600）")
    else:
        print(f"· {ENV_FILE.name} 已存在，只补齐缺失项（不动已有值）")

    text = read_env()

    # ── AUTH_JWT_SECRET：登录/鉴权的签名密钥 ──────────────────
    current = get_val(text, "AUTH_JWT_SECRET")
    too_short = bool(current) and len(current.encode("utf-8")) < MIN_SECRET_BYTES
    if not current or current == PLACEHOLDER or too_short:
        text = upsert(text, "AUTH_JWT_SECRET", secrets.token_urlsafe(48))
        if too_short:
            print(
                f"✓ AUTH_JWT_SECRET 原来只有 {len(current.encode('utf-8'))} 字节"
                f"（不足 {MIN_SECRET_BYTES}），已重新生成为 64 字符随机密钥"
            )
        else:
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

    # ── 体检：还差哪些平台密钥 ────────────────────────────────────────
    print("\n── 平台密钥体检 ──")
    fatal_missing: list[str] = []
    optional_missing: list[str] = []
    for key, fatal, why in PLATFORM_KEYS:
        value = get_val(text, key)
        if value and "xxxx" not in value.lower():
            print(f"  ✓ {key}")
            continue
        tag = "必填" if fatal else "可选"
        print(f"  ✗ {key}  [{tag}] {why}")
        (fatal_missing if fatal else optional_missing).append(key)

    if fatal_missing:
        print(
            f"\n  ⚠️ 有 {len(fatal_missing)} 个**必填** key 是空的。承载它们的模块在"
            "\n     import 期就构造 OpenAI 客户端，而 api_key 为 None 会直接抛异常 ——"
            "\n     缺了不是降级，是 **api 容器启动即崩**。申请后务必补齐："
        )
        for key in fatal_missing:
            print(f"       {key}")
    elif not optional_missing:
        print("\n  ✓ 平台密钥都齐了")

    if optional_missing:
        print(f"\n  · 未填的可选项（不影响启动）：{', '.join(optional_missing)}")

    dead = [key for key in DEAD_KEYS if get_val(text, key) is not None]
    if dead:
        print(
            f"\n  提示：{', '.join(dead)} 当前代码**不读**（见文件顶部 DEAD_KEYS 注释），"
            "填了不生效，可以直接从 .env 删掉。"
        )

    if fatal_missing or optional_missing:
        print(f"\n  编辑：nano {ENV_FILE}")

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
