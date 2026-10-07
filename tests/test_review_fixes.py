"""本轮评审修复的回归测试（上传路径逃逸 / 请求体上限 见各自文件）—— 零服务、零真实 API。

覆盖四件事：
1. **工具循环步数上限**：模型反复请求工具时必须有终点；
2. **出网调用超时**：4 个 LLM 客户端 + MCP 桥；
3. **身份注入点的变量名**：客户端注入与服务端读取必须同名（不一致 = 静默失去保护）；
4. **请求入口不得再用 `print`**：cp936 下会抛 `UnicodeEncodeError`。
"""
import asyncio
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

import agent.graph as g
import mcp_client
import mcp_order_data
import tools_agent.tool_llm as tool_llm
from agent import langchina
from intent import problemdecomposition
from rag.generatellm import RAGGenerator

ROOT = Path(__file__).resolve().parent.parent


# ══════════════════════════════════════════════════════════════
# 1. 工具循环必须有步数上限
# ══════════════════════════════════════════════════════════════
def _tool_call_state(rounds: int) -> dict:
    return {
        "tool_rounds": rounds,
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"name": "add", "args": {}, "id": "call_1"}],
            )
        ],
    }


@pytest.mark.parametrize("rounds", [0, 1, tool_llm.TOOL_MAX - 1])
def test_tool_loop_keeps_going_below_the_cap(rounds):
    assert g.tool_continue(_tool_call_state(rounds)) == "tool_call"


@pytest.mark.parametrize("rounds", [tool_llm.TOOL_MAX, tool_llm.TOOL_MAX + 5])
def test_tool_loop_stops_at_the_cap(rounds):
    """超过 TOOL_MAX 必须收尾。

    回归背景：`TOOL_MAX` 以前**定义了却没有任何地方引用**，于是工具循环
    `llm_call ⇄ tool_call` 的唯一终点是 langgraph 的 DEFAULT_RECURSION_LIMIT
    （默认 10007）—— 模型只要反复请求工具，就能打出上千轮 LLM 调用。
    """
    assert g.tool_continue(_tool_call_state(rounds)) == "data_node"


def test_tool_loop_without_tool_calls_still_finishes():
    state = {"messages": [AIMessage(content="直接回答")]}

    assert g.tool_continue(state) == "data_node"


def test_capped_loop_gives_an_honest_answer_instead_of_an_empty_string():
    """到上限时模型往往只回 tool_calls、`content` 为空 —— 不能把空串交给上层。

    空 answer 会让 api/chat.py 再调一次兜底 Agent（白跑一轮），
    最终还可能把空字符串发给前端。
    """
    answer = g.data_node(_tool_call_state(tool_llm.TOOL_MAX))["answer"]

    assert answer, "不能返回空串"
    assert str(tool_llm.TOOL_MAX) in answer, "要说明是轮次用尽，而不是假装成功"


def test_llm_call_node_actually_counts_rounds(monkeypatch):
    """轮次必须真的被累加，否则上面的上限永远触发不了。"""

    class _Msg:
        content = "已经查到了"
        tool_calls = None

    class _Res:
        choices = [type("_C", (), {"message": _Msg()})()]

    monkeypatch.setattr(g, "call_zhipu_chat", lambda messages: _Res())

    out = g.llm_call_node({"messages": [], "tool_rounds": 2})

    assert out["tool_rounds"] == 3


def test_graph_declares_a_finite_recursion_limit():
    """整图步数上限必须是有限值，且远小于 langgraph 的默认 10007。"""
    assert 0 < g.GRAPH_RECURSION_LIMIT < 1000


def test_tool_loop_terminates_at_the_cap_in_the_real_compiled_graph(monkeypatch):
    """端到端证据：把**编译后的 tool_agent 图**真的跑一遍。

    光断言 `tool_continue` 的返回值不足以证明 B3 修好了：
    - 计数能不能活过 langgraph 的 super-step（状态合并）？
    - 它会不会一路跑到 `recursion_limit` 才炸？

    这里让模型**永远**请求工具（"add" 是占位工具，每次都会回一条
    "暂不支持" 的 ToolMessage —— 正是死循环的燃料），然后要求：
    LLM 恰好被调用 `TOOL_MAX` 次，并且最终给出的是一句诚实的话术。
    """
    calls = {"n": 0}

    def _always_wants_a_tool(messages):
        calls["n"] += 1
        msg = SimpleNamespace(
            content="",
            tool_calls=[
                SimpleNamespace(
                    id="call_1",
                    function=SimpleNamespace(name="add", arguments='{"a": 1, "b": 2}'),
                )
            ],
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    monkeypatch.setattr(g, "call_zhipu_chat", _always_wants_a_tool)

    out = g.tool_agent.invoke(
        {"question": "3 加 5 等于几", "session_id": "s1", "messages": []},
        config={"recursion_limit": g.GRAPH_RECURSION_LIMIT},
    )

    assert calls["n"] == tool_llm.TOOL_MAX, "应该在工具轮次上限处停下，而不是跑满 recursion_limit"
    assert str(tool_llm.TOOL_MAX) in (out.get("answer") or "")


# ══════════════════════════════════════════════════════════════
# 2. 出网调用必须有超时
# ══════════════════════════════════════════════════════════════
def test_fallback_agent_client_has_a_timeout():
    assert langchina.model.request_timeout is not None
    assert langchina.model.max_retries is not None


def test_generate_client_has_a_timeout_and_bounded_retries():
    client = RAGGenerator().generate_client

    assert client.timeout is not None
    assert client.max_retries == 1


@pytest.mark.parametrize("which", ["zhipu", "split"])
def test_tool_and_splitter_clients_have_timeouts(which):
    client = tool_llm.zhipu_client if which == "zhipu" else problemdecomposition.client

    assert client.timeout is not None
    assert client.max_retries == 1


def test_mcp_bridge_times_out_on_a_hung_server():
    """MCP 服务是子进程：挂住时不会自己退出，必须由桥来超时。"""

    async def _hang():
        await asyncio.sleep(30)

    with pytest.raises(RuntimeError, match="超时"):
        mcp_client._run_bridge(_hang(), "测试挂起", timeout=0.05)


def test_mcp_timeout_constant_is_positive():
    assert mcp_client.MCP_TIMEOUT > 0


# ══════════════════════════════════════════════════════════════
# 3. 身份注入点两边必须同名
# ══════════════════════════════════════════════════════════════
def test_caller_env_var_is_identical_on_both_sides():
    """客户端注入的变量名与服务端读取的必须一致。

    不一致的后果是**静默**的：服务端永远读到 None，而数据层把 None 当作
    "未绑定身份"的宽松模式 —— 保护消失，但没有任何报错。
    """
    assert mcp_client.CALLER_ENV_VAR == mcp_order_data.CALLER_ENV_VAR


def test_server_params_inject_the_caller_into_the_environment():
    """身份走子进程**环境变量**：这是模型碰不到的地方（它只能写工具参数）。"""
    spec = mcp_client.ServerSpec("order", "python", ("/tmp/mcp_order_server.py",))

    plain = mcp_client._server_params(spec)
    with_caller = mcp_client._server_params(spec, caller_id="C2")

    assert plain.env is None, "没身份时不该改动子进程环境"
    assert with_caller.env == {mcp_client.CALLER_ENV_VAR: "C2"}


def test_caller_identity_is_not_part_of_tool_arguments(monkeypatch):
    """桥必须把身份单独传给 `_server_params`，而不是塞进模型可写的 arguments。"""
    captured = {}

    def _fake_params(spec, caller_id=None):
        captured["caller_id"] = caller_id
        return mcp_client.StdioServerParameters(command="x", args=[])

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def call_tool(self, name, arguments):
            captured["arguments"] = arguments
            return SimpleNamespace(
                is_error=False, content=[SimpleNamespace(text="ok")]
            )

    monkeypatch.setattr(mcp_client, "_server_params", _fake_params)
    monkeypatch.setattr(mcp_client, "Client", lambda *a, **kw: _FakeClient())

    spec = mcp_client.ServerSpec("order", "python", ("/tmp/mcp_order_server.py",))
    out = asyncio.run(
        mcp_client._call_tool_text(
            spec, "query_order", {"order_no": "X"}, caller_id="C2"
        )
    )

    assert out == "ok"
    assert captured["caller_id"] == "C2"
    assert captured["arguments"] == {"order_no": "X"}
    assert "caller_id" not in captured["arguments"]


# ══════════════════════════════════════════════════════════════
# 4. 请求入口不得再用 print
# ══════════════════════════════════════════════════════════════
def test_request_handlers_do_not_print():
    """cp936 下 print 非 ASCII 会抛 UnicodeEncodeError。

    stdout 被重定向（日志文件 / 容器 / CI）时，Windows 上 `sys.stdout.encoding`
    就是 cp936，而入口那行曾经印的是 `"✅ ..."` —— 每个 /api/chat 请求都会
    在进入业务逻辑之前 500。

    用 AST 找**真正的 print 调用**（而不是源码里出现这个词 —— 注释里提到它
    是正常的，上面这段说明本身就提到过）。
    """
    tree = ast.parse((ROOT / "api" / "chat.py").read_text(encoding="utf-8"))

    offenders = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]

    assert not offenders, f"请求路径上不该再有 print，命中第 {offenders} 行"


def test_chat_handler_docstring_is_not_shadowed():
    """原来 print 写在 docstring **之前**，那个字符串根本不是 docstring。"""
    from api import chat

    assert chat.chat.__doc__, "/chat 的 docstring 被前面的语句顶掉了"
    assert "编排图" in chat.chat.__doc__


def test_chat_module_keeps_a_logger():
    from api import chat

    assert chat.logger.name == "api.chat"


# ══════════════════════════════════════════════════════════════
# 5. "零真实 API 调用"必须是被强制的，而不是靠自觉
# ══════════════════════════════════════════════════════════════
def test_outbound_network_guard_actually_blocks():
    """守闸本身也要被测 —— 否则它可能只是一个静默的空壳。

    用 IP 字面量，避免真实 DNS 查询；`create_connection` 与裸 `connect`
    两条路都要挡住（httpx 走前者，anyio/裸 socket 走后者）。
    """
    import socket

    with pytest.raises(RuntimeError, match="禁止真实出网"):
        socket.create_connection(("93.184.216.34", 443), timeout=1)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(RuntimeError, match="禁止真实出网"):
            sock.connect(("93.184.216.34", 443))
    finally:
        sock.close()


def test_outbound_network_guard_allows_loopback():
    """回环必须放行：Windows 上 asyncio 的内部自管道要用它。

    这里不真的建监听，只验证"回环地址不会触发守卫" —— 用一个必然失败的端口，
    断言失败原因是**连接被拒**而不是守卫拦截。
    """
    import socket

    with pytest.raises(OSError) as exc:
        socket.create_connection(("127.0.0.1", 1), timeout=0.5)

    assert "禁止真实出网" not in str(exc.value)
