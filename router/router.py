from fastapi import FastAPI
from api.api import chat, upload_file


def register_routers(app: FastAPI) -> None:
    """
    统一注册路由，注册的路由挂载到 main 里去
    每个路由模块都带有一个 prefix（路径前缀）
    """
    app.include_router(chat.router, prefix="/api", tags=["聊天"])
    app.include_router(upload_file.router, prefix="/api", tags=["文件上传"])
    # app.include_router(assistant.router, prefix="/api/v1/assistant", tags=["AI助手"])

    # 后续添加新路由模块时，只需要在这里加一行即可
    # app.include_router(knowledge.router, prefix="/api/v1", tags=["知识库"])
