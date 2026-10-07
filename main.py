import logging
import os
import sys

from dotenv import load_dotenv

# ⚠️ load_dotenv 必须在下面的 import 之前跑：agent.graph / agent.langchina 等模块
# 在 **import 时** 就会构造 LLM 客户端并从环境变量取 api_key。
load_dotenv(encoding="utf-8-sig")  # utf-8-sig:兼容带 BOM 的 .env

# ══════════════════════════════════════════════════════════════
# stdout / stderr 编码兜底
#
# 项目里还有大量 print（正在按评审报告的 A5 逐步收编成 logging），而 Windows 上
# stdout 一旦被重定向（写日志文件 / 容器 / CI），编码就是 cp936 —— 打印 emoji、
# `¥`、`⚠` 这类字符会直接抛 UnicodeEncodeError。
#
# 这不是"理论风险"：`¥`（U+00A5）正是订单金额的货币符号、`⚠` 是运输超期提示，
# 也就是说**业务数据本身**就可能让一行日志把请求打成 500
# （实测：`agent/graph.py` 打印问题文本、`tools_agent/tool_llm.py` 打印工具返回时都会触发）。
#
# 这里把两个流设成 UTF-8 + errors="replace"：最坏情况是日志里出现一个替代字符，
# 而不是请求失败。根治办法仍是把 print 换成 logging（A5）。
# ══════════════════════════════════════════════════════════════
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - 流被替换过时可能没有 reconfigure
        pass

# ══════════════════════════════════════════════════════════════
# 启动前自检：这 4 个 key 是**import 期硬依赖**
#
# 为什么要有这一段：agent/graph.py 在模块级就 `Generator = RAGGenerator()`、
# agent/langchina.py 在模块级就构造 ChatOpenAI、intent/problemdecomposition.py 在
# 模块级就构造 OpenAI —— 缺任何一个 key，`import main` 会直接抛
#     openai.OpenAIError: Missing credentials. Please pass an `api_key` ...
# 这个报错完全指不到"是哪个变量缺了"，别人 clone 下来只会看到一坨堆栈就放弃。
# 这里提前拦一道，把缺的变量名和用途一次列清楚。
# ══════════════════════════════════════════════════════════════
REQUIRED_KEYS = {
    "DEEPSEEK_API_KEY": "主模型：query 改写 / 意图仲裁 / 兜底 Agent（deepseek-chat）",
    "GENERATE_API_KEY": "答案生成：RAG 出答案（DashScope qwen3-32b）",
    "QIAN_WEN_QUERYSTION_API_KEY": "多问题拆分（DashScope qwen3.5-flash）",
    "ZHI_PU_API_KEY": "工具调用模型（智谱 glm-4.5-air）",
}

# 认证相关的必需项。只在开启认证时才要求 —— 但**默认就是开启的**。
#
# 为什么必须启动即失败：JWT 密钥是"谁能签发令牌"的唯一凭据。
# 若允许缺省，代码里就得有个内置默认密钥，那等于**任何人都能伪造任意身份的令牌**，
# 而服务看起来一切正常 —— 这是最危险的一类配置。
AUTH_REQUIRED_KEYS = {
    "AUTH_JWT_SECRET": (
        "认证：JWT 签名密钥（生成："
        'python -c "import secrets;print(secrets.token_urlsafe(48))"）'
    ),
}


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    return default if not raw else raw not in {"0", "false", "off", "no"}


def _required_keys() -> dict[str, str]:
    keys = dict(REQUIRED_KEYS)
    if _env_bool("AUTH_ENABLED", True):
        keys.update(AUTH_REQUIRED_KEYS)
    return keys


def preflight() -> None:
    """缺 key 就打印一份可照做的清单然后退出，别让用户去啃 OpenAI 的堆栈。"""
    required = _required_keys()
    missing = [k for k in required if not (os.getenv(k) or "").strip()]
    if not missing:
        if not _env_bool("AUTH_ENABLED", True):
            # 关掉认证是**显式**的危险动作：必须显眼，别让它悄悄进生产
            print(
                "\n[WARNING] AUTH_ENABLED=false：接口**不做任何身份校验**，"
                "任何能访问端口的人都能问答、上传、读知识库。"
                "这只应出现在本地演示环境；/health 也会暴露该状态。\n",
                file=sys.stderr,
            )
        return

    lines = [
        "",
        "=" * 68,
        "  启动失败：.env 里缺少必需的配置（这些值在 import 期就会被读取）",
        "=" * 68,
        "",
    ]
    # ⚠️ 这里只用 ASCII 符号：Windows 控制台默认 cp936，打不出 ✗ 这类字符
    for key in missing:
        lines.append(f"  [缺失] {key}")
        lines.append(f"         {required[key]}")
    lines += [
        "",
        "  怎么修：",
        "    1) 如果还没有 .env，先从模板复制一份：",
        "         Windows:  copy .env.example .env",
        "         Linux/mac: cp .env.example .env",
        "    2) 打开 .env，把上面列出的变量填成自己的真实 key",
        "       （每个变量该去哪申请，.env.example 里都写了链接）",
        "    3) 重新启动： python main.py",
        "",
        "  只想跑测试？不需要任何 key，直接 pytest 即可（tests/conftest.py 会注入占位值）。",
        "=" * 68,
        "",
    ]
    print("\n".join(lines), file=sys.stderr)
    raise SystemExit(1)


preflight()

import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402

import config  # noqa: E402
from body_limit import add_body_limit  # noqa: E402
from cors import setup_cors  # noqa: E402
from observability import add_observability, setup_logging  # noqa: E402
from router.router import register_routers  # noqa: E402

# 日志：统一走 observability.setup_logging —— 它装的是**带 request_id** 的单行
# key=value 格式，并顺手把第三方库压到 WARNING。
# 不配置 handler 时 root 没有 handler，INFO 会被直接丢掉（只有 print 可见），
# 所以必须在应用入口配一次。
setup_logging()

# from agent.graph import graph
# from ag_ui_langgraph import add_langgraph_fastapi_endpoint

# Redis 客户端统一在 redis_client.py 里管理（模块级单例，全局共享）

# 创建 FastAPI 应用实例
app = FastAPI(title="我的智能问答系统 API", version=config.APP_VERSION)

# 1. 请求体上限：必须在 multipart 解析**之前**生效，否则超限请求会先被完整
#    缓冲到内存/临时盘才拿到 413（详见 body_limit.py）
add_body_limit(app)

# 2. 配置 CORS。
#    ⚠️ 顺序有讲究：Starlette 里**后加的在最外层**。CORS 必须比请求体上限更靠外，
#    否则"按 Content-Length 直接拒掉"的那条 413 不会带上 CORS 头，
#    浏览器会把前端看到的错误变成 CORS 报错而不是 413。
setup_cors(app)

# 3. 可观测性放**最外层**：这样连"请求体超限""JSON 解析失败"这类早期返回
#    也会被记进访问日志与指标；否则它们在观测层之内，会变成监控盲区。
add_observability(app)

# 4. 将注册封装好的路由，以函数的形式挂载到 FastAPI 实例上
register_routers(app)

#
# add_langgraph_fastapi_endpoint(app, graph, "/agent")


# 启动脚本
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)  # 端口要与前端配置的 api-url 一致
