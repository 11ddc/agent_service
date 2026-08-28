from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

def setup_cors(app: FastAPI) -> None:
    """
    配置 CORS 中间件
    允许前端（Vue 开发服务器）访问后端 API
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",      # Vue 默认开发端口
            "http://127.0.0.1:5173",
            # 后续可以添加更多前端域名，如生产环境地址
            # "https://your-production-domain.com"
        ],
        allow_credentials=True,
        allow_methods=["*"],              # 允许所有 HTTP 方法（GET, POST, PUT, DELETE等）
        allow_headers=["*"],              # 允许所有请求头
    )