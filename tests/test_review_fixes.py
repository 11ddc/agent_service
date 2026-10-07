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

    def _fake_params(spec, caller_id=None, resolved_customer=None, **kwargs):
        captured["caller_id"] = caller_id
        captured["resolved_customer"] = resolved_customer
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
            spec, "query_order", {"order_no": "X"}, caller_id="C2", resolved_customer="C3"
        )
    )

    assert out == "ok"
    assert captured["caller_id"] == "C2"
    assert captured["resolved_customer"] == "C3"
    assert captured["arguments"] == {"order_no": "X"}
    assert "caller_id" not in captured["arguments"]
    assert "resolved_customer" not in captured["arguments"]


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


# ══════════════════════════════════════════════════════════════
# 6. 对抗性复核（第二轮）发现的问题
# ══════════════════════════════════════════════════════════════
def test_agent_node_passes_its_own_recursion_limit(monkeypatch):
    """兜底 Agent 必须**自己**带步数上限。

    不传的话用它自己的默认值（langchain 内部是 9999），而这个 Agent 同样带工具、
    模型可以反复调用。实测：编译图**自己的** config 会生效，所以不能指望上层的
    `recursion_limit` 兜住它。
    """
    seen = {}

    class _FakeAgent:
        def invoke(self, payload, config=None):
            seen["config"] = config
            return {"messages": [SimpleNamespace(content="兜底答案")]}

    monkeypatch.setattr(g, "agent", _FakeAgent())

    out = g.agent_node({"question": "q", "session_id": "s1"})

    assert out["answer"] == "兜底答案"
    assert seen["config"]["recursion_limit"] == g.AGENT_RECURSION_LIMIT
    assert seen["config"]["configurable"]["thread_id"] == "s1"


def test_multi_loop_caps_the_number_of_sub_questions(monkeypatch):
    """子问题数量来自 LLM 输出，必须设上限，否则一次请求跑 N 遍完整 RAG。"""
    monkeypatch.setattr(g, "MAX_SUB_QUESTIONS", 2)
    asked = []

    class _FakeQuestionGraph:
        def invoke(self, payload, config=None):
            asked.append(payload["question"])
            return {"answer": f"A:{payload['question']}", "meta": {}}

    monkeypatch.setattr(g, "question_graph", _FakeQuestionGraph())

    out = g.multi_loop_node(
        {
            "sub_questions": ["q1", "q2", "q3", "q4"],
            "session_id": "s",
            "principal_id": "u1",
        }
    )

    assert asked == ["q1", "q2"], "只该处理前 MAX_SUB_QUESTIONS 个"
    assert "只处理了前 2 个" in out["answer"], "必须诚实告知还有没处理的"
    assert "q3" not in out["answer"]


def test_multi_loop_forwards_the_principal_to_the_subgraph(monkeypatch):
    """身份必须透传，否则子图里的工具调用拿不到 principal（会失败关闭）。"""
    seen = {}

    class _FakeQuestionGraph:
        def invoke(self, payload, config=None):
            seen.update(payload)
            return {"answer": "A", "meta": {}}

    monkeypatch.setattr(g, "question_graph", _FakeQuestionGraph())

    g.multi_loop_node(
        {"sub_questions": ["q1"], "session_id": "s", "principal_id": "u42"}
    )

    assert seen["principal_id"] == "u42"
    assert seen["session_id"] == "s"


def test_redis_client_has_socket_timeouts():
    """Redis 默认没有 socket 超时 —— 半开连接会让请求路径永久挂住。"""
    import redis_client

    kwargs = redis_client.redis_client.connection_pool.connection_kwargs

    assert kwargs.get("socket_timeout"), "缺少 socket_timeout"
    assert kwargs.get("socket_connect_timeout"), "缺少 socket_connect_timeout"


def test_mysql_connection_has_read_and_write_timeouts():
    """pymysql 的 read_timeout 默认是 None（永不超时）。"""
    from db.mysql import _connect_kwargs

    kwargs = _connect_kwargs("mysql+pymysql://u:p@127.0.0.1:3306/rag?charset=utf8mb4")

    assert kwargs["connect_timeout"] > 0
    assert kwargs["read_timeout"] > 0
    assert kwargs["write_timeout"] > 0
    assert kwargs["charset"] == "utf8mb4"


def test_cp936_cannot_encode_the_symbols_our_business_data_uses():
    """先证明这个坑是真的，再验证兜底写法有效。

    实测：`¥`（U+00A5，订单金额用的就是它）与 `⚠`（U+26A0，运输超期提示用的）
    **都编不进 cp936**（注意 U+FFE5 `￥` 可以，两者不是同一个字符）。
    stdout 被重定向时编码就是 cp936，所以一行 print 业务数据就能把请求打成 500。
    """
    import io

    strict = io.TextIOWrapper(io.BytesIO(), encoding="cp936", errors="strict")
    with pytest.raises(UnicodeEncodeError):
        strict.write("订单金额 ¥2699.00")
    with pytest.raises(UnicodeEncodeError):
        strict.write("在途已超过 5 天 ⚠")

    # main.py 的兜底写法：UTF-8 + replace → 最坏是日志里一个替代字符，而不是请求失败
    tolerant = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", errors="replace")
    tolerant.write("订单金额 ¥2699.00 ⚠")
    tolerant.flush()


def test_main_installs_the_stdout_encoding_guard():
    """兜底必须真的挂在入口上，而不是只写在注释里。"""
    src = (ROOT / "main.py").read_text(encoding="utf-8")

    assert 'reconfigure(encoding="utf-8", errors="replace")' in src


# ══════════════════════════════════════════════════════════════
# 7. MCP 子进程的**配置转发**（静默故障回归）
# ══════════════════════════════════════════════════════════════
def test_server_params_forward_business_config_to_the_child(monkeypatch):
    """业务侧配置必须**显式转发**给 MCP 子进程。

    SDK 只让子进程继承一份白名单环境变量（PATH / SystemRoot / …），业务配置不在其中。
    不转发的话服务端会用自己的默认值继续跑 —— 例如"把 `ORDER_DEMO_SINGLE_TENANT`
    设成 0，服务端却仍然认为在演示模式"，**而且不报错**。

    这不是假想：实测就是这样，端到端跑一次才发现"关掉演示模式"完全没生效。
    """
    monkeypatch.setenv("ORDER_DEMO_SINGLE_TENANT", "0")
    monkeypatch.setenv("ORDER_PRINCIPAL_MAP", '{"u1": "C2"}')
    spec = mcp_client.ServerSpec("order", "python", ("/tmp/mcp_order_server.py",))

    params = mcp_client._server_params(spec, caller_id="u1")

    assert params.env["ORDER_DEMO_SINGLE_TENANT"] == "0"
    assert params.env["ORDER_PRINCIPAL_MAP"] == '{"u1": "C2"}'
    assert params.env[mcp_client.CALLER_ENV_VAR] == "u1"


def test_server_params_forward_a_whitelist_not_the_whole_environment(monkeypatch):
    """转发必须走**白名单**：密钥之类的绝不能进子进程。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-should-not-be-forwarded")
    monkeypatch.setenv("GENERATE_API_KEY", "sk-also-not")
    spec = mcp_client.ServerSpec("order", "python", ("/tmp/mcp_order_server.py",))

    env = mcp_client._server_params(spec, caller_id="u1").env or {}

    assert "DEEPSEEK_API_KEY" not in env
    assert "GENERATE_API_KEY" not in env
    assert set(env) <= set(mcp_client._FORWARDED_ENV) | {mcp_client.CALLER_ENV_VAR}


def test_server_params_leave_env_untouched_when_nothing_to_pass(monkeypatch):
    """没有任何要传的东西时不要凭空造一个 env（保持"不改动子进程环境"的语义）。"""
    for name in mcp_client._FORWARDED_ENV:
        monkeypatch.delenv(name, raising=False)
    spec = mcp_client.ServerSpec("order", "python", ("/tmp/mcp_order_server.py",))

    assert mcp_client._server_params(spec).env is None
