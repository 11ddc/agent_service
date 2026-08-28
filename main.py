from fastapi import FastAPI
import uvicorn
from cors import setup_cors
from router.router import register_routers

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
    uvicorn.run(app, host="127.0.0.1", port=8000) # 端口要与前端配置的 api-url 一致
