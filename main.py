import logging
import os
import sys

from dotenv import load_dotenv

# ⚠️ load_dotenv 必须在下面的 import 之前跑：agent.graph / agent.langchina 等模块
# 在 **import 时** 就会构造 LLM 客户端并从环境变量取 api_key。
load_dotenv(encoding="utf-8-sig")  # utf-8-sig:兼容带 BOM 的 .env

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


def preflight() -> None:
    """缺 key 就打印一份可照做的清单然后退出，别让用户去啃 OpenAI 的堆栈。"""
    missing = [k for k in REQUIRED_KEYS if not (os.getenv(k) or "").strip()]
    if not missing:
        return

    lines = [
        "",
        "=" * 68,
        "  启动失败：.env 里缺少必需的 API Key（这些 key 在 import 期就会被读取）",
        "=" * 68,
        "",
    ]
    # ⚠️ 这里只用 ASCII 符号：Windows 控制台默认 cp936，打不出 ✗ 这类字符
    for key in missing:
        lines.append(f"  [缺失] {key}")
        lines.append(f"         {REQUIRED_KEYS[key]}")
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

from cors import setup_cors  # noqa: E402
from router.router import register_routers  # noqa: E402

# 日志：改写 / 意图识别 / 路由 里的"降级、兜底、被拒"关键路径都用 logging 记录。
# 不配置 basicConfig 时 root 没有 handler，INFO 会被直接丢掉（只有 print 可见），
# 所以统一在应用入口配一次。
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
# 第三方库的 INFO 太吵，压到 WARNING：
# openai SDK 3.x 基于 httpx2（注意不是 httpx），每个请求都会打一行
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpx2").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# from agent.graph import graph
# from ag_ui_langgraph import add_langgraph_fastapi_endpoint

# Redis 客户端统一在 redis_client.py 里管理（模块级单例，全局共享）

# 创建 FastAPI 应用实例
app = FastAPI(title="我的智能问答系统 API", version="1.0.0")

# 1. 配置 CORS
setup_cors(app)

# 2. 将注册封装好的路由，以函数的形式挂载到 FastAPI 实例上
register_routers(app)

#
# add_langgraph_fastapi_endpoint(app, graph, "/agent")


# 启动脚本
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)  # 端口要与前端配置的 api-url 一致
