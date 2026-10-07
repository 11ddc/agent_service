from fastapi import FastAPI

from api import auth as auth_api
from api import kb as kb_api
from api import system as system_api
from api.api import chat, upload_file


def register_routers(app: FastAPI) -> None:
    """
    统一注册路由，注册的路由挂载到 main 里去
    每个路由模块都带有一个 prefix（路径前缀）
    """
    # 探针与指标挂在**根路径**（/health、/ready、/metrics）：
    # k8s 的探针与 Prometheus 抓取都按约定走根路径，加前缀反而要额外配置
    app.include_router(system_api.router)
    # 认证独立成一组：登录/注册/刷新/登出不需要"已登录身份"，
    # 与管理类接口分开挂载，排查与配网关策略都更清楚
    app.include_router(auth_api.router, prefix="/api/auth", tags=["认证"])
    app.include_router(kb_api.router, prefix="/api/kb", tags=["知识库管理"])
    app.include_router(chat.router, prefix="/api", tags=["聊天"])
    app.include_router(upload_file.router, prefix="/api", tags=["文件上传"])
    # app.include_router(assistant.router, prefix="/api/v1/assistant", tags=["AI助手"])

    # 后续添加新路由模块时，只需要在这里加一行即可
    # app.include_router(knowledge.router, prefix="/api/v1", tags=["知识库"])
