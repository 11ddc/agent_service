import logging

import uvicorn
from fastapi import FastAPI

from cors import setup_cors
from router.router import register_routers

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

# 2. 将注册封装好的路由，以函数的形式挂载到 FastAPI 应用实例上
register_routers(app)

#
# add_langgraph_fastapi_endpoint(app, graph, "/agent")


# 启动脚本
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)  # 端口要与前端配置的 api-url 一致
