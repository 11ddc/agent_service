"""订单 MCP 服务端（`mcp_order_server.py`）测试 —— 零服务：不起子进程、不连网络。

服务端本身只是薄适配层，所以这里不重复测业务逻辑（那在 test_mcp_order_data.py），
只守**接线**：服务名、工具清单、以及喂给模型的那份"工具说明书"。

接线错了的表现很隐蔽：服务能起来、工具能列出，但模型因为描述不对而永远选错工具，
或者客户端的命名空间对不上导致工具能列出却调不动 —— 两者都不报错。
"""
import asyncio
import json
import re

import pytest

import mcp_order_data
import mcp_order_server


def _tools() -> list:
    """用公开 API 取工具清单（不碰私有属性）。"""
    return asyncio.run(mcp_order_server.mcp.list_tools())


@pytest.fixture(autouse=True)
def _default_caller(monkeypatch):
    """默认注入一个 principal（演示模式下会映射到演示客户）。

    服务端要求身份：解析不出来就失败关闭（这是刻意的）。大多数用例关心的是
    "接线"，所以统一给一个合法 principal；个别用例再覆盖或删除它。
    """
    monkeypatch.setenv(mcp_order_data.CALLER_ENV_VAR, "u_test_principal")


def test_server_name_matches_the_namespace_used_by_the_client():
    """serverName 必须与客户端配置一致。

    不一致的话工具能列出来、调用却路由到别的服务或直接报"未配置"——
    而工具列表看起来完全正常。
    """
    assert mcp_order_server.SERVER_NAME == "order"


def test_exposes_exactly_the_expected_tools():
    names = [t.name for t in _tools()]

    assert names == [
        "query_order",
        "query_logistics",
        "query_refund",
        "query_recent_orders",
    ]


@pytest.mark.parametrize(
    "tool_name,required",
    [
        ("query_order", ["order_no"]),
        ("query_logistics", ["order_no"]),
        ("query_refund", ["order_no"]),
        # query_recent_orders 现在**不接受任何"查谁"的参数**：调用方身份由服务端
        # 注入，模型只能选 limit（有默认值，所以整个工具没有必填项）。
        ("query_recent_orders", []),
    ],
)
def test_tool_schema_requires_the_right_arguments(tool_name, required):
    """参数 schema 来自类型注解 —— 注解写错，模型就会传错参数名。"""
    tool = next(t for t in _tools() if t.name == tool_name)

    assert tool.input_schema.get("required", []) == required


def test_order_tools_never_expose_a_customer_identifier():
    """回归：订单工具的 schema 里**不能**出现 phone / 客户号 / 身份字段。

    以前 query_recent_orders 的入参是 `phone: str` —— 那等于给模型一个
    "按手机号枚举他人订单"的入口。身份必须由服务端注入（见 mcp_client 的
    CALLER_ENV_VAR），永远不出现在工具参数里。
    """
    forbidden = {"phone", "customer_id", "caller", "caller_id", "user_id", "session_id"}

    for tool in _tools():
        props = set((tool.input_schema or {}).get("properties") or {})
        leaked = props & forbidden
        assert not leaked, f"{tool.name} 暴露了身份类参数: {leaked}"


def test_tools_fail_closed_when_identity_is_missing(monkeypatch):
    """没有身份时必须**拒绝返回数据**。

    这条守卫挡的是"客户端忘了注入身份"：数据层的 customer_id=None 是
    "不过滤"的宽松路径（只该给单测直调用），绝不能从服务端走下去。
    """
    monkeypatch.delenv(mcp_order_data.CALLER_ENV_VAR, raising=False)

    out = mcp_order_server.query_order("ORD-20250820-001")

    assert "身份" in out or "转人工" in out
    assert "已签收" not in out, "没有身份不得拿到任何订单数据"


def test_tools_fail_closed_when_demo_mode_is_off(monkeypatch):
    """关掉单租户演示模式后，会话 id 这种"认不出的身份"必须失败关闭。"""
    monkeypatch.setenv(mcp_order_data.DEMO_ENV_VAR, "0")
    monkeypatch.setenv(mcp_order_data.CALLER_ENV_VAR, "session-abc")

    out = mcp_order_server.query_order("ORD-20250820-001")

    assert "身份" in out or "转人工" in out
    assert "已签收" not in out


def test_demo_mode_keeps_the_tool_chain_usable(monkeypatch):
    """演示模式（默认）下，真实的 session id 也能拿到演示客户的数据 —— 单租户。

    这是"安全"与"可用"的折中：没有认证层时不假装能区分用户，而是让所有调用方
    看到同一份数据，并把开关与前提写进文档（.env.example / README）。
    """
    monkeypatch.delenv(mcp_order_data.DEMO_ENV_VAR, raising=False)
    monkeypatch.setenv(mcp_order_data.CALLER_ENV_VAR, "7f3a-session-uuid")

    out = mcp_order_server.query_order("ORD-20250820-001")

    assert "ORD-20250820-001" in out


def test_tools_only_return_the_callers_own_orders(monkeypatch):
    """两个 principal 分别绑到两个客户：越权只能拿到"查不到"。"""
    mine = mcp_order_data.customer_id_for_phone("13800001111")
    other = mcp_order_data.customer_id_for_phone("13900002222")
    monkeypatch.setenv(mcp_order_data.DEMO_ENV_VAR, "0")
    monkeypatch.setenv(
        mcp_order_data.PRINCIPAL_MAP_ENV_VAR,
        json.dumps({"u_mine": mine, "u_other": other}),
    )

    monkeypatch.setenv(mcp_order_data.CALLER_ENV_VAR, "u_mine")
    assert "ORD-20250820-001" in mcp_order_server.query_order("ORD-20250820-001")

    # 换一个 principal 去查同一张单：必须拿不到任何字段，
    # 且话术与"订单根本不存在"完全同形（否则就成了存在性探测接口）
    monkeypatch.setenv(mcp_order_data.CALLER_ENV_VAR, "u_other")
    stolen = mcp_order_server.query_order("ORD-20250820-001")
    missing = mcp_order_server.query_order("ORD-19990101-999")

    assert "已签收" not in stolen
    assert "¥" not in stolen
    assert stolen.replace("ORD-20250820-001", "X") == missing.replace(
        "ORD-19990101-999", "X"
    )


def test_caller_named_like_a_customer_id_cannot_select_that_customer(monkeypatch):
    """⚠️ 安全回归：把调用方**取名叫客户号**，不能因此读到那个客户的订单。

    这正是上一版留下的真实越权：身份映射里有"如果这个值看起来像客户号，就直接
    采信"的推断，而 caller_id 的一端是**客户端可控**的 session_id，于是
    `POST /api/chat {"session_id": "C3"}` 就能读到 C3 的订单。

    现在映射只认服务端来源：演示模式、显式映射表（或认证层）。
    """
    monkeypatch.setenv(mcp_order_data.DEMO_ENV_VAR, "0")
    monkeypatch.delenv(mcp_order_data.PRINCIPAL_MAP_ENV_VAR, raising=False)
    monkeypatch.setenv(mcp_order_data.CALLER_ENV_VAR, "C3")  # 直接报一个客户号

    out = mcp_order_server.query_order("ORD-20250820-001")

    assert "身份" in out or "转人工" in out
    assert "已签收" not in out, "取名叫客户号不该拿到该客户的数据"


def test_demo_mode_still_serves_the_demo_customer(monkeypatch):
    """演示模式（默认）下工具链可用，但所有 principal 看到的是**同一个**客户。"""
    monkeypatch.delenv(mcp_order_data.DEMO_ENV_VAR, raising=False)

    monkeypatch.setenv(mcp_order_data.CALLER_ENV_VAR, "u_any")
    first = mcp_order_server.query_recent_orders()
    monkeypatch.setenv(mcp_order_data.CALLER_ENV_VAR, "u_somebody_else")
    second = mcp_order_server.query_recent_orders()

    assert first == second, "演示模式下换个 principal 不该看到不同的数据"
    assert "ORD-" in first


def test_recent_orders_cannot_be_redirected_by_a_positional_argument(monkeypatch):
    """旧接口的 `phone` 参数已经删掉：就算硬塞一个手机号进来，也改不了"查谁"。

    这里传的是**别人**的手机号。因为服务端只认环境变量里的身份，
    两次调用的订单集合必须完全一致 —— 参数会影响不到查询对象。
    """
    mine = mcp_order_data.customer_id_for_phone("13800001111")
    monkeypatch.setenv(mcp_order_data.CALLER_ENV_VAR, mine)

    def _orders(text: str) -> set:
        return set(re.findall(r"ORD-\d{8}-\d{3}", text))

    without = mcp_order_server.query_recent_orders()
    with_phone = mcp_order_server.query_recent_orders("13900002222")

    assert _orders(without), "至少应该查到自己名下的订单"
    assert _orders(with_phone) == _orders(without)


@pytest.mark.parametrize("tool", _tools(), ids=lambda t: t.name)
def test_every_tool_description_says_when_to_use_it(tool):
    """docstring 就是模型的选型依据：必须写清"何时用"与"何时别用"。"""
    desc = tool.description or ""

    assert "使用场景" in desc
    assert "不适用场景" in desc, f"{tool.name} 没说清什么时候不该用它"


def test_order_tools_distinguish_order_logistics_and_refund():
    """三个工具职责相邻，描述里必须互相指向，否则模型会拿订单工具去答退款问题。"""
    by_name = {t.name: (t.description or "") for t in _tools()}

    assert "query_logistics" in by_name["query_order"]
    assert "query_refund" in by_name["query_order"]
    assert "query_order" in by_name["query_refund"]


def test_calling_a_tool_delegates_to_the_data_layer():
    """注册后的函数仍是普通函数，可直接调用（服务端只做转发）。"""
    out = mcp_order_server.query_order("ord-20250820-001")

    assert "ORD-20250820-001" in out
    assert "已签收" in out


def test_refund_tool_reports_missing_refund_record():
    out = mcp_order_server.query_refund("ORD-20250820-001")

    assert "没有" in out and "退款" in out
