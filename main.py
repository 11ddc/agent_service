"""服务入口。

启动：python main.py（端口与前端配置的 api-url 保持一致：127.0.0.1:8000）
"""

import sys

from dotenv import load_dotenv

# ⚠️ load_dotenv 必须在下面的 import 之前跑：agent.graph / agent.langchina 等模块
# 在 **import 时** 就会构造 LLM 客户端并从环境变量取 api_key。
load_dotenv(encoding="utf-8-sig")  # utf-8-sig:兼容带 BOM 的 .env

# stdout / stderr 编码兜底：Windows 上流被重定向（写日志文件 / 容器 / CI）时编码是
# cp936，打印 `¥`（订单金额）或 `⚠`（运输超期）会抛 UnicodeEncodeError 把请求打成 500。
# 设成 UTF-8 + errors="replace" 后，最坏只是日志里出现一个替代字符。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - 流被替换过时可能没有 reconfigure
        pass

import uvicorn
from fastapi import FastAPI

import config
from body_limit import add_body_limit
from cors import setup_cors
from observability import add_observability, setup_logging
from router.router import register_routers

# 日志：统一走 observability.setup_logging —— 带 request_id 的单行 key=value 格式，
# 并把第三方库压到 WARNING。不配 handler 的话 INFO 会被直接丢掉。
setup_logging()

# 创建 FastAPI 应用实例
app = FastAPI(title="我的智能问答系统 API", version=config.APP_VERSION)

# 1. 请求体上限：必须在 multipart 解析**之前**生效，否则超限请求会先被完整缓冲
add_body_limit(app)

# 2. CORS。⚠️ 顺序有讲究：Starlette 里**后加的在最外层**，CORS 必须比请求体上限更靠外，
#    否则"按 Content-Length 直接拒掉"的那条 413 不带 CORS 头，前端看到的是 CORS 报错。
setup_cors(app)

# 3. 可观测性放**最外层**：连"请求体超限""JSON 解析失败"这类早期返回也会被记进
#    访问日志与指标，否则它们会变成监控盲区。
add_observability(app)

# 4. 挂载路由
register_routers(app)

# 启动脚本
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)  # 端口要与前端配置的 api-url 一致
