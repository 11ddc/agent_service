"""tool_call_node 的回归测试 —— 零服务：不调智谱 / 不连 MCP / 不碰 Redis。

补这个文件的原因：`tool_call_node` 之前**一个测试都没有**，于是下面两件事
在真实用户路径上活了很久没人发现。

1. **占位工具把假数据喂给模型。** `tools_agent/tool_llm.py` 的 searchOrder 曾返回
   "搜索成功，您的订单为。。。。。。。。。。。。"，模型据此编出一段语气自信的
   订单状态回答，而接口返回 200、日志一切正常 —— 用户完全看不出是编的。
2. **未知工具会打死循环。** 原实现只 print 一句就忽略：state 里没追加任何消息，
   最后一条仍是那条带 tool_calls 的 AIMessage → tool_continue 返回 "tool_call"
   → llm_call → tool_call → …（该分支当时也没有测试）。
"""
import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage

import agent.graph as g


def _state(tool_name: str) -> dict:
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": tool_name, "args": {"query": "x"}, "id": "call_1"}
                ],
            )
        ]
    }


def _only_message(state: dict) -> ToolMessage:
    """每个 tool_call 必须恰好回一条 ToolMessage —— 不多不少。"""
    out = g.tool_call_node(state)["messages"]
    assert len(out) == 1, f"应恰好回 1 条 ToolMessage，实际 {len(out)} 条"
    msg = out[0]
    assert isinstance(msg, ToolMessage)
    assert msg.tool_call_id == "call_1"
    return msg


# ── 1. 占位工具：必须诚实，绝不能返回假数据 ──


def test_placeholder_order_tool_never_returns_fake_data():
    text = _only_message(_state("searchOrder")).content

    assert "暂不支持" in text
    # 下面这两个断言才是这个测试存在的理由：一旦有人把假数据改回来，立刻变红
    assert "搜索成功" not in text
    assert "。。。。" not in text


def test_placeholder_add_tool_is_refused_too():
    assert "暂不支持" in _only_message(_state("add")).content


def test_placeholder_tools_are_not_exposed_to_the_model():
    """占位工具不能出现在喂给模型的 tools 列表里。"""
    from tools_agent import tool_llm

    blob = json.dumps(tool_llm.tools, ensure_ascii=False)

    assert "searchOrder" not in blob
    assert "add" not in blob


def test_placeholder_tool_body_raises_instead_of_lying():
    """第二层防护：即使被误注册回 tools 列表，函数体也要当场炸，而不是静默返回假数据。"""
    from tools_agent import tool_llm

    with pytest.raises(Exception) as ei:
        tool_llm.searchOrder.invoke({"query": "订单号", "session_id": "s1"})

    assert "未接入真实订单系统" in str(ei.value)


# ── 2. 未知工具：必须回消息，否则 tool_continue 死循环 ──


def test_unknown_tool_still_gets_a_tool_message():
    assert "不可用" in _only_message(_state("no_such_tool_xyz")).content


def test_unknown_tool_does_not_leave_dangling_tool_calls():
    """死循环的根因断言：不能留下"有 tool_calls 但没人应答"的消息。

    只 print 不追加消息时，state["messages"][-1] 仍是那条 AIMessage，
    tool_continue 于是一直返回 "tool_call" → llm_call → tool_call → …
    """
    out = g.tool_call_node(_state("no_such_tool_xyz"))["messages"]

    assert {m.tool_call_id for m in out} == {"call_1"}


# ── 3. MCP 工具：本次改动不能把真实工具一起弄坏 ──


def test_mcp_tool_still_executes(monkeypatch):
    calls = []

    def _fake_call(name, args):
        calls.append((name, args))
        return "外部工具结果"

    monkeypatch.setattr(g, "call_mcp_tool", _fake_call)

    assert _only_message(_state("mcp__weather__get")).content == "外部工具结果"
    assert calls == [("mcp__weather__get", {"query": "x"})]


def test_mcp_tool_failure_degrades_to_message(monkeypatch):
    def _boom(name, args):
        raise RuntimeError("外部服务超时")

    monkeypatch.setattr(g, "call_mcp_tool", _boom)

    assert "MCP 工具执行失败" in _only_message(_state("mcp__weather__get")).content


# ── 4. 没有 tool_calls 时不该凭空造消息 ──


def test_no_tool_calls_returns_empty():
    state = {"messages": [AIMessage(content="我直接回答")]}

    assert g.tool_call_node(state)["messages"] == []
