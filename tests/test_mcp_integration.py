"""订单 MCP 的**端到端**集成测试 —— 真的起子进程。

标记为 `integration`，默认不跑（`pytest.ini` 里 `addopts = -m "not integration"`），
所以它不影响默认套件"零外部服务、零真实出网"的承诺。要跑它：

    .\\venv\\Scripts\\python.exe -m pytest -m integration -q

## 为什么必须单独有这一层

这条链路上出现过**只有真起子进程才暴露**的问题：mcp SDK 只让子进程继承一份
**白名单**环境变量（PATH / SystemRoot / …），业务配置不在其中。不显式转发的话，
服务端会用自己的默认值继续跑 —— 典型表现是"你把 `ORDER_DEMO_SINGLE_TENANT`
设成 0，但服务端仍然认为自己在演示模式"，**而且完全不报错**。

单测里 `Client` 是替身，永远看不到这一点；只有真起进程才能验证。
本文件只连本机的 stdio 子进程，不访问任何外部服务。
"""

import json

import pytest

import mcp_client

pytestmark = pytest.mark.integration


@pytest.fixture
def order_server(monkeypatch):
    monkeypatch.setenv(
        "MCP_SERVERS",
        json.dumps(
            [
                {
                    "serverName": "order",
                    "transport": "stdio",
                    "command": "python",
                    "args": ["mcp_order_server.py"],
                }
            ]
        ),
    )


def _call(caller: str, order_no: str = "ORD-20250820-001") -> str:
    return mcp_client.call_mcp_tool(
        "mcp__order__query_order", {"order_no": order_no}, caller_id=caller
    )


def test_demo_off_fails_closed_for_every_principal(order_server, monkeypatch):
    """关掉演示模式后，任何 principal（包括"长得像客户号"的）都拿不到数据。"""
    monkeypatch.setenv("ORDER_DEMO_SINGLE_TENANT", "0")
    monkeypatch.delenv("ORDER_PRINCIPAL_MAP", raising=False)

    for caller in ("C1", "C2", "C3", "some-session-uuid"):
        out = _call(caller)
        assert "已签收" not in out, f"caller={caller} 不该拿到订单数据"
        assert "身份" in out or "转人工" in out


def test_principal_map_works_end_to_end(order_server, monkeypatch):
    """服务端显式映射是唯一合法的身份来源，映射到的客户能正常查询。"""
    monkeypatch.setenv("ORDER_DEMO_SINGLE_TENANT", "0")
    monkeypatch.setenv("ORDER_PRINCIPAL_MAP", json.dumps({"u_real": "C2"}))

    assert "ORD-20250820-001" in _call("u_real")
    assert "已签收" not in _call("u_other"), "未映射的 principal 必须失败关闭"


def test_demo_toggle_actually_reaches_the_child_process(order_server, monkeypatch):
    """⚠️ 静默故障的回归：开关必须**真的**生效，而不是被子进程忽略。

    这条用例的价值全在"真起子进程"上：如果 `_server_params` 忘了转发这个变量，
    子进程会一直用自己的默认值（演示模式开），于是"关掉演示模式"形同虚设，
    而所有单测仍然全绿。
    """
    monkeypatch.setenv("ORDER_DEMO_SINGLE_TENANT", "1")
    assert "ORD-20250820-001" in _call("whatever"), "演示模式下应当可用"

    monkeypatch.setenv("ORDER_DEMO_SINGLE_TENANT", "0")
    assert "已签收" not in _call("whatever"), "关掉之后必须失败关闭"


def test_identity_env_var_is_the_only_identity_source(order_server, monkeypatch):
    """工具参数里塞身份字段无效：身份只认子进程环境变量。"""
    monkeypatch.setenv("ORDER_DEMO_SINGLE_TENANT", "0")
    monkeypatch.setenv("ORDER_PRINCIPAL_MAP", json.dumps({"u_real": "C2"}))

    forged = mcp_client.call_mcp_tool(
        "mcp__order__query_order",
        {
            "order_no": "ORD-20250820-001",
            # 模型能看到的只有工具参数 —— 把"我是谁"塞进来应当完全无效
            mcp_client.CALLER_ENV_VAR: "u_real",
            "caller_id": "u_real",
            "customer_id": "C2",
        },
        caller_id="u_other",
    )

    assert "已签收" not in forged, "伪造的参数不该改变身份"
