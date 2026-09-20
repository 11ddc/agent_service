"""MCP 客户端桥（`mcp_client.py`）测试 —— 零服务：不起子进程、不连任何服务。

这一层以前完全没有测试，而它有一个**静默失效**的隐患：服务清单来自
环境变量，配置写错的话工具会整体消失、但不会有任何报错。所以这里重点守：

1. **配置解析**：单个对象 / 数组 / 非法 JSON / 缺字段，都不能让工具链消失；
2. **命名空间路由**：`mcp__<server>__<tool>` 必须路由到正确服务，
   工具名里含下划线也不能切错；指向未配置服务时要报错，不能静默改投别家；
3. **逐服务降级**：某个服务起不来，只跳过它，其余服务的工具照常可用 ——
   订单服务没起来不该让通用工具一起消失。
"""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import mcp_client


@pytest.fixture(autouse=True)
def _clear_mcp_env(monkeypatch):
    """绝不依赖"环境变量恰好没设"。

    注意：其它测试模块 import config 时会 load_dotenv()，把 .env 里的
    MCP_SERVERS 写进 os.environ —— 那样"无配置时用默认服务"这条用例就会
    随 import 顺序时好时坏。所以每个用例都显式清干净。
    """
    monkeypatch.delenv("MCP_SERVERS", raising=False)


# ── 1. 配置解析 ─────────────────────────────────────────────


def test_without_env_falls_back_to_builtin_server():
    specs = mcp_client.load_server_specs()

    assert [s.name for s in specs] == ["mcp_server"]
    assert specs[0].args[-1].endswith("mcp_server.py")


def test_accepts_single_json_object(monkeypatch):
    """.env 里老的写法是单个对象，必须继续支持。"""
    monkeypatch.setenv(
        "MCP_SERVERS",
        '{"serverName":"order","transport":"stdio",'
        '"command":"python","args":["mcp_order_server.py"]}',
    )

    specs = mcp_client.load_server_specs()

    assert [s.name for s in specs] == ["order"]


def test_accepts_json_array_of_servers(monkeypatch):
    monkeypatch.setenv(
        "MCP_SERVERS",
        '[{"serverName":"mcp_server","command":"python","args":["mcp_server.py"]},'
        '{"serverName":"order","command":"python","args":["mcp_order_server.py"]}]',
    )

    specs = mcp_client.load_server_specs()

    assert [s.name for s in specs] == ["mcp_server", "order"]


def test_invalid_json_falls_back_instead_of_killing_the_tool_chain(monkeypatch):
    """配置写错只该退化成默认，不该让整条 MCP 链路消失。"""
    monkeypatch.setenv("MCP_SERVERS", "{这不是 JSON")

    specs = mcp_client.load_server_specs()

    assert [s.name for s in specs] == ["mcp_server"]


def test_entry_without_server_name_is_skipped(monkeypatch):
    monkeypatch.setenv(
        "MCP_SERVERS",
        '[{"command":"python","args":["a.py"]},'
        '{"serverName":"order","command":"python","args":["mcp_order_server.py"]}]',
    )

    specs = mcp_client.load_server_specs()

    assert [s.name for s in specs] == ["order"]


def test_all_entries_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("MCP_SERVERS", '[{"nope":1},{"also":"bad"}]')

    specs = mcp_client.load_server_specs()

    assert [s.name for s in specs] == ["mcp_server"]


# ── 2. 命令与路径解析 ───────────────────────────────────────


def test_python_command_is_pinned_to_current_interpreter(monkeypatch):
    """直接用 "python" 依赖 PATH，在 Windows 上常指向另一个环境或不存在。"""
    monkeypatch.setenv(
        "MCP_SERVERS",
        '{"serverName":"order","command":"python","args":["mcp_order_server.py"]}',
    )

    spec = mcp_client.load_server_specs()[0]

    assert spec.command == sys.executable


def test_relative_script_path_is_resolved_against_project_root(monkeypatch):
    monkeypatch.setenv(
        "MCP_SERVERS",
        '{"serverName":"order","command":"python","args":["mcp_order_server.py"]}',
    )

    spec = mcp_client.load_server_specs()[0]

    arg = Path(spec.args[0])
    assert arg.is_absolute()
    assert arg == mcp_client.ROOT / "mcp_order_server.py"
    assert arg.exists(), "解析出来的脚本路径必须真实存在"


def test_non_script_args_are_left_untouched(monkeypatch):
    """`-m` 这类参数不能被当路径改坏。"""
    monkeypatch.setenv(
        "MCP_SERVERS",
        '{"serverName":"x","command":"python","args":["-m","some_module"]}',
    )

    spec = mcp_client.load_server_specs()[0]

    assert spec.args == ("-m", "some_module")


# ── 3. 命名空间路由 ─────────────────────────────────────────

SPECS = [
    mcp_client.ServerSpec("mcp_server", "python", ("/tmp/mcp_server.py",)),
    mcp_client.ServerSpec("order", "python", ("/tmp/mcp_order_server.py",)),
]


def test_namespaced_name_routes_to_its_server():
    spec, raw = mcp_client.split_namespaced("mcp__order__query_order", SPECS)

    assert spec is not None and spec.name == "order"
    assert raw == "query_order"


def test_tool_name_with_underscores_is_not_mis_split():
    """工具名本身含下划线时，按 "__" 硬切会切错服务。"""
    spec, raw = mcp_client.split_namespaced("mcp__order__query_recent_orders", SPECS)

    assert spec is not None and spec.name == "order"
    assert raw == "query_recent_orders"


def test_unknown_server_namespace_is_not_silently_rerouted():
    spec, raw = mcp_client.split_namespaced("mcp__weather__get_weather", SPECS)

    assert spec is None
    assert raw == "mcp__weather__get_weather"


def test_call_with_unconfigured_server_reports_clearly(monkeypatch):
    """指向没配置的服务要报错并列出已配置项，不能静默改投别的服务。"""
    monkeypatch.setenv(
        "MCP_SERVERS",
        '{"serverName":"order","command":"python","args":["mcp_order_server.py"]}',
    )

    with pytest.raises(RuntimeError, match="未在 MCP_SERVERS 中配置"):
        mcp_client.call_mcp_tool("mcp__weather__get_weather", {})


# ── 4. 逐服务降级（这一层最重要的一条）──────────────────────


def _fake_client_factory(by_script: dict):
    """按启动脚本名给出行为：工具名列表，或一个要抛的异常。"""

    class _Ctx:
        def __init__(self, params):
            self._script = Path(params.args[-1]).name

        async def __aenter__(self):
            behavior = by_script[self._script]
            if isinstance(behavior, BaseException):
                raise behavior
            self._names = behavior
            return self

        async def __aexit__(self, *exc):
            return False

        async def list_tools(self):
            return SimpleNamespace(
                tools=[
                    SimpleNamespace(name=n, description=f"描述 {n}", input_schema={})
                    for n in self._names
                ]
            )

    return lambda params: _Ctx(params)


def test_one_broken_server_does_not_hide_the_others(monkeypatch):
    """订单服务没起来，通用工具必须照常可用。

    这是多服务改造的核心收益：以前一个服务起不来，MCP 工具会整体消失。
    """
    specs = [
        mcp_client.ServerSpec("order", "python", ("/tmp/mcp_order_server.py",)),
        mcp_client.ServerSpec("mcp_server", "python", ("/tmp/mcp_server.py",)),
    ]
    monkeypatch.setattr(
        mcp_client,
        "Client",
        _fake_client_factory(
            {
                "mcp_order_server.py": RuntimeError("服务未启动"),
                "mcp_server.py": ["add"],
            }
        ),
    )

    out = asyncio.run(mcp_client._list_tools_openai(specs))

    assert [t["function"]["name"] for t in out] == ["mcp__mcp_server__add"]


def test_all_servers_broken_yields_empty_list_without_raising(monkeypatch):
    """全挂时返回空列表（工具为空），而不是把异常抛到图节点里炸掉请求。"""
    specs = [mcp_client.ServerSpec("order", "python", ("/tmp/mcp_order_server.py",))]
    monkeypatch.setattr(
        mcp_client,
        "Client",
        _fake_client_factory({"mcp_order_server.py": RuntimeError("挂了")}),
    )

    assert asyncio.run(mcp_client._list_tools_openai(specs)) == []


def test_tools_from_multiple_servers_are_merged_with_namespace(monkeypatch):
    specs = [
        mcp_client.ServerSpec("mcp_server", "python", ("/tmp/mcp_server.py",)),
        mcp_client.ServerSpec("order", "python", ("/tmp/mcp_order_server.py",)),
    ]
    monkeypatch.setattr(
        mcp_client,
        "Client",
        _fake_client_factory(
            {
                "mcp_server.py": ["add"],
                "mcp_order_server.py": ["query_order", "query_refund"],
            }
        ),
    )

    out = asyncio.run(mcp_client._list_tools_openai(specs))

    assert [t["function"]["name"] for t in out] == [
        "mcp__mcp_server__add",
        "mcp__order__query_order",
        "mcp__order__query_refund",
    ]
    assert all(t["type"] == "function" for t in out)
