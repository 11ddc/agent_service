"""订单 MCP 服务端（`mcp_order_server.py`）测试 —— 零服务：不起子进程、不连网络。

服务端本身只是薄适配层，所以这里不重复测业务逻辑（那在 test_mcp_order_data.py），
只守**接线**：服务名、工具清单、以及喂给模型的那份"工具说明书"。

接线错了的表现很隐蔽：服务能起来、工具能列出，但模型因为描述不对而永远选错工具，
或者客户端的命名空间对不上导致工具能列出却调不动 —— 两者都不报错。
"""
import asyncio

import pytest

import mcp_order_server


def _tools() -> list:
    """用公开 API 取工具清单（不碰私有属性）。"""
    return asyncio.run(mcp_order_server.mcp.list_tools())


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
        ("query_recent_orders", ["phone"]),
    ],
)
def test_tool_schema_requires_the_right_arguments(tool_name, required):
    """参数 schema 来自类型注解 —— 注解写错，模型就会传错参数名。"""
    tool = next(t for t in _tools() if t.name == tool_name)

    assert tool.input_schema["required"] == required


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
