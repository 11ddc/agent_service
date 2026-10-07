"""LangGraph checkpointer 的选择，以及"外置"这件事的现实。

## 它影响什么

兜底 Agent（`agent/langchina.py`）用 checkpointer 保存**会话内多轮记忆**，
按 `thread_id` 索引 —— 那个 id 用的是带身份命名空间的 `session_key`
（早期用客户端可控的 session_id，两个客户端撞同一个值就会共享记忆，已修）。

## 现状与取舍（不要把它当成"已完成外置"）

默认 `InMemorySaver`：进程内、**重启即丢**、多副本各存一份。同一个用户被负载
均衡打到另一个副本时，Agent 会"忘记"刚才的对话 —— 用户只会觉得"客服怎么又忘了"，
这类故障很难被发现，所以这里选择**每次启动打一条 WARNING 说清楚**，
而不是默默用内存实现假装没事。

真外置需要额外依赖（本仓库**没有**列进 requirements，装了才能用）：

    pip install langgraph-checkpoint-sqlite      # 单机/单副本：落磁盘
    pip install langgraph-checkpoint-postgres    # 多副本：共享存储

企业部署请显式设 `AGENT_CHECKPOINT_BACKEND=sqlite`（或 postgres）。
配了但依赖没装时**直接抛错**（fail fast），而不是悄悄退回 memory ——
后者会让人以为已经外置了。
"""

import logging
import sqlite3

import config

logger = logging.getLogger(__name__)

BACKEND_MEMORY = "memory"
BACKEND_SQLITE = "sqlite"
BACKEND_POSTGRES = "postgres"
BACKENDS = (BACKEND_MEMORY, BACKEND_SQLITE, BACKEND_POSTGRES)

_INSTALL_HINT = {
    BACKEND_SQLITE: "pip install langgraph-checkpoint-sqlite",
    BACKEND_POSTGRES: "pip install langgraph-checkpoint-postgres",
}


class CheckpointBackendUnavailable(RuntimeError):
    """配了某个后端但依赖不可用（或后端名不认识）。"""


def resolve_backend() -> str:
    return (config.AGENT_CHECKPOINT_BACKEND or BACKEND_MEMORY).strip().lower()


def create_checkpointer(backend: str | None = None):
    """按配置创建 checkpointer。

    | backend | 依赖 | 适用 |
    |---|---|---|
    | memory（默认） | 无 | 本地开发/单副本且能接受重启丢记忆 |
    | sqlite | langgraph-checkpoint-sqlite | 单机或容器单副本 |
    | postgres | langgraph-checkpoint-postgres | 多副本共享 |
    """
    backend = (backend or resolve_backend()).strip().lower()

    if backend == BACKEND_MEMORY:
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()

    if backend == BACKEND_SQLITE:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ModuleNotFoundError as e:
            raise CheckpointBackendUnavailable(
                f"AGENT_CHECKPOINT_BACKEND=sqlite 需要额外依赖：{_INSTALL_HINT[backend]}。"
                "装好后重启即可；不想装就显式设回 memory。"
            ) from e
        path = config.AGENT_CHECKPOINT_SQLITE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False：langgraph 会在工作线程里读写（api/chat 把图丢进线程池）
        conn = sqlite3.connect(str(path), check_same_thread=False)
        logger.info("会话记忆使用 SQLite checkpointer: %s", path)
        return SqliteSaver(conn)

    if backend == BACKEND_POSTGRES:
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
        except ModuleNotFoundError as e:
            raise CheckpointBackendUnavailable(
                f"AGENT_CHECKPOINT_BACKEND=postgres 需要额外依赖：{_INSTALL_HINT[backend]}。"
                "多副本部署请装它并配好连接串，否则各副本的会话记忆是割裂的。"
            ) from e
        return PostgresSaver.from_conn_string(config.AGENT_CHECKPOINT_POSTGRES_URL)

    raise CheckpointBackendUnavailable(
        f"未知的 AGENT_CHECKPOINT_BACKEND={backend!r}（可选：{', '.join(BACKENDS)}）"
    )


def warn_if_not_externalized(backend: str | None = None) -> str | None:
    """memory 后端返回告警文案并打 WARNING，供启动自检调用。"""
    backend = (backend or resolve_backend()).strip().lower()
    if backend != BACKEND_MEMORY:
        return None
    message = (
        "会话记忆使用进程内实现（InMemorySaver）：重启即丢，多副本各存一份。"
        "企业部署请设置 AGENT_CHECKPOINT_BACKEND=sqlite（单副本）或 postgres（多副本），"
        "并安装对应的 langgraph-checkpoint-* 包。"
    )
    logger.warning(message)
    return message
