"""checkpointer 后端选择测试（A7 的"checkpoint 外置"部分）—— 完全离线。

守的性质：

1. **默认 memory 能用**（零依赖，不阻塞本地开发）；
2. **配了某个后端但依赖没装时明确报错**，而不是悄悄退回 memory ——
   后者会让人以为"已经外置了"，而实际行为是重启丢记忆、多副本各存一份；
3. **启动时必须就此告警**：这类故障的症状是"客服怎么又忘了"，极难归因；
4. **未知后端名报错并列出可选项**（拼错 `sqlit` 不该静默变成 memory）。
"""

import pytest

import config
from agent import checkpoint as cp


@pytest.fixture(autouse=True)
def _restore_backend():
    original = config.AGENT_CHECKPOINT_BACKEND
    try:
        yield
    finally:
        config.AGENT_CHECKPOINT_BACKEND = original


# ── memory ───────────────────────────────────────────────────
def test_default_backend_is_memory():
    config.AGENT_CHECKPOINT_BACKEND = "memory"

    assert cp.resolve_backend() == "memory"
    assert type(cp.create_checkpointer()).__name__ == "InMemorySaver"


def test_memory_backend_warns_about_not_being_externalized():
    """默认值不是"最佳实践"，只是零依赖 —— 必须说清楚，否则会被当成已完成外置。"""
    config.AGENT_CHECKPOINT_BACKEND = "memory"

    message = cp.warn_if_not_externalized()

    assert message is not None
    assert "重启即丢" in message
    assert "AGENT_CHECKPOINT_BACKEND" in message


def test_externalized_backend_does_not_warn():
    assert cp.warn_if_not_externalized("sqlite") is None
    assert cp.warn_if_not_externalized("postgres") is None


# ── 缺依赖时有明确的错 ────────────────────────────────────────
@pytest.mark.parametrize(
    "backend,package",
    [("sqlite", "langgraph-checkpoint-sqlite"), ("postgres", "langgraph-checkpoint-postgres")],
)
def test_missing_dependency_raises_with_install_hint(backend, package):
    """fail fast：绝不能悄悄退回 memory。"""
    try:
        import langgraph.checkpoint.sqlite  # noqa: F401

        pytest.skip("本环境已装 sqlite checkpointer，跳过缺依赖分支")
    except ModuleNotFoundError:
        pass

    with pytest.raises(cp.CheckpointBackendUnavailable) as exc:
        cp.create_checkpointer(backend)

    assert package in str(exc.value)
    assert "pip install" in str(exc.value)


def test_unknown_backend_lists_the_options():
    with pytest.raises(cp.CheckpointBackendUnavailable) as exc:
        cp.create_checkpointer("sqlit")  # 拼错

    message = str(exc.value)
    assert "sqlit" in message
    for name in cp.BACKENDS:
        assert name in message


def test_backend_name_is_case_and_space_insensitive():
    """`SQLITE ` 这种写法不该被判成"未知后端"（运维手写配置的常见形态）。"""
    with pytest.raises(cp.CheckpointBackendUnavailable) as exc:
        cp.create_checkpointer("  SQLite  ")

    # 判定为 sqlite（只是依赖没装），而不是"未知后端"
    assert "未知的" not in str(exc.value)
    assert "langgraph-checkpoint-sqlite" in str(exc.value)


def test_resolve_backend_reads_config():
    config.AGENT_CHECKPOINT_BACKEND = "Postgres"

    assert cp.resolve_backend() == "postgres"


def test_sqlite_path_is_absolute_within_the_project():
    """默认落盘位置必须在项目内：容器里靠卷挂载保留，不能写到莫名其妙的地方。"""
    assert config.AGENT_CHECKPOINT_SQLITE_PATH.is_absolute()
    assert str(config.AGENT_CHECKPOINT_SQLITE_PATH).startswith(str(config.ROOT))
