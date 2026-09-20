# mcp_client.py — MCP 客户端桥(供 LangGraph 同步图节点在线程中调用)
#
# 背景:graph 的节点在 asyncio.to_thread 工作线程里以同步方式执行,
#       而 mcp SDK 的 Client 是异步的。这里保留"每次调用开一个独立连接
#       (async with Client(...),进入自动 spawn+握手、退出自动清理)"的写法,
#       再用 asyncio.run 包一层,对外暴露**同步函数**,图节点直接调用即可。
#
# 注意:asyncio.run 要求调用方线程里没有正在运行的事件循环——
#       当前用法(同步图跑在线程里)满足;如果将来改全异步 ainvoke,
#       需要把这两个函数换回 async 版本。
#
# 工具名统一命名空间:mcp__<server>__<tool>,避免与本地工具(如 add)重名,
# 也便于 tool_call_node 按 mcp__ 前缀分流。
#
# 多服务支持(本次改造):
#   以前这里的 SERVER_FILE / SERVER_NAME 是**硬编码**成 mcp_server.py 的 ——
#   .env 里的 MCP_SERVERS 从来没被读过,所以那条配置一直是死的,整个 MCP 层
#   实际只能接一个服务。现在改成从 MCP_SERVERS 读服务清单:
#     · 命名空间带上 server 名(mcp__order__query_order),按 server 路由;
#     · **单个服务挂掉不影响其他服务** —— 拉工具列表时逐个降级,
#       订单服务没起来不该让通用工具一起消失。
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from mcp import Client, StdioServerParameters

logger = logging.getLogger(__name__)

# ⚠️ 必须在这里 load_dotenv：本模块在**调用时**读 MCP_SERVERS，而不配的话会
# 静默退回内置默认服务（工具看起来正常，只是少了几个）。以前硬编码 SERVER_FILE
# 所以不需要这行；现在配置来自环境变量，就不能再依赖"入口恰好先加载过 .env"
# 这种 import 顺序的巧合（config.py 顶部有同类教训）。
load_dotenv(encoding="utf-8-sig")  # utf-8-sig:兼容带 BOM 的 .env

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ServerSpec:
    """一个 MCP 服务子进程的启动描述。"""

    name: str  # 决定工具命名空间 mcp__<name>__<tool>
    command: str
    args: tuple[str, ...]


def _resolve_command(command: str) -> str:
    """把 "python"/"python3" 换成**当前解释器**。

    直接用 "python" 依赖 PATH，在 Windows 上常常指向另一个环境（或根本不存在），
    表现是 MCP 服务起不来、工具全消失 —— 用 sys.executable 才是确定的。
    """
    return sys.executable if command in ("python", "python3", "py") else command


def _resolve_arg(arg: str) -> str:
    """相对路径的脚本按**项目根目录**解析。

    .env 里写的是 "mcp_order_server.py"，而子进程的工作目录不保证是项目根，
    不解析的话换个启动方式就找不到脚本。
    只处理看起来像脚本的参数，避免把 -m / run 这类参数当路径改坏。
    """
    if not arg.endswith(".py"):
        return arg
    p = Path(arg)
    return str(p if p.is_absolute() else ROOT / p)


def _spec_from_entry(entry: object) -> ServerSpec | None:
    if not isinstance(entry, dict):
        return None
    name = str(entry.get("serverName") or "").strip()
    command = str(entry.get("command") or "").strip()
    if not name or not command:
        return None
    raw_args = entry.get("args") or []
    if not isinstance(raw_args, list):
        return None
    return ServerSpec(
        name=name,
        command=_resolve_command(command),
        args=tuple(_resolve_arg(str(a)) for a in raw_args),
    )


def load_server_specs() -> list[ServerSpec]:
    """从环境变量 MCP_SERVERS 读服务清单(JSON)。

    接受**单个对象**或**数组**两种写法(老配置是单对象)。
    任何解析失败都退回内置默认服务并告警 —— 配置写错不该让整条 MCP 链路消失。
    """
    raw = (os.getenv("MCP_SERVERS") or "").strip()
    if not raw:
        return _default_specs()

    try:
        data = json.loads(raw)
    except Exception as e:  # noqa: BLE001
        logger.warning("MCP_SERVERS 解析失败，退回默认服务: %s", e)
        return _default_specs()

    entries = data if isinstance(data, list) else [data]
    specs = [s for s in (_spec_from_entry(e) for e in entries) if s is not None]
    if not specs:
        logger.warning("MCP_SERVERS 里没有合法条目，退回默认服务")
        return _default_specs()
    return specs


def _default_specs() -> list[ServerSpec]:
    """内置默认:通用工具服务 mcp_server.py(与改造前行为一致)。"""
    return [
        ServerSpec(
            name="mcp_server",
            command=sys.executable,
            args=(str(ROOT / "mcp_server.py"),),
        )
    ]


def _server_params(spec: ServerSpec) -> StdioServerParameters:
    """描述如何启动服务端子进程。"""
    return StdioServerParameters(command=spec.command, args=list(spec.args))


def split_namespaced(tool_name: str, specs: list[ServerSpec]):
    """mcp__<server>__<tool> → (spec, 真名)。

    按**已知 server 名**匹配而不是按 "__" 切分:工具名本身可能含下划线,
    简单切分会在这种名字上切错(路由到错误的服务)。
    返回 (None, 原串) 表示这个名字不对应任何已配置的服务。
    """
    for spec in specs:
        prefix = f"mcp__{spec.name}__"
        if tool_name.startswith(prefix):
            return spec, tool_name[len(prefix) :]
    return None, tool_name


async def _list_tools_openai(specs: list[ServerSpec]) -> list[dict]:
    """异步内芯:逐个服务列出工具并转成 OpenAI/智谱 function 格式。

    **逐个降级**:某个服务起不来只跳过它并告警,不影响其余服务的工具。
    反过来的话,一个没起来的订单服务会让整条 MCP 工具链全线消失。
    """
    out: list[dict] = []
    for spec in specs:
        try:
            async with Client(_server_params(spec)) as client:
                tools_result = await client.list_tools()
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "MCP 服务 %s 不可用，跳过其工具（根因 -> %s）", spec.name, _explain(e)
            )
            continue
        out += [
            {
                "type": "function",
                "function": {
                    "name": f"mcp__{spec.name}__{tool.name}",
                    "description": tool.description or "",
                    "parameters": tool.input_schema
                    or {"type": "object", "properties": {}},
                },
            }
            for tool in tools_result.tools
        ]
    return out


async def _call_tool_text(spec: ServerSpec, raw_name: str, arguments: dict) -> str:
    """异步内芯:调用工具,返回文本结果;工具报错(is_error)则抛异常。"""
    async with Client(_server_params(spec)) as client:
        result = await client.call_tool(raw_name, arguments=arguments)
        if result.is_error:
            raise RuntimeError(f"MCP 工具 {raw_name} 执行失败: {result.content}")
        texts = [c.text for c in result.content if getattr(c, "text", None)]
        return "\n".join(texts) if texts else str(result.content)


def _explain(exc: BaseException) -> str:
    """把异常(含 TaskGroup/ExceptionGroup 包装)展开成可读的根因文本。

    mcp 2.1.1 内部用 TaskGroup 管理后台任务,出错时会把真正的原因包成
    "unhandled errors in a TaskGroup (1 sub-exception)",直接抛出来人看不懂。
    这里递归展开每一层子异常,取到底层原因。
    """
    if isinstance(exc, BaseExceptionGroup):
        return "; ".join(_explain(e) for e in exc.exceptions)
    return f"{type(exc).__name__}: {exc}"


def _run_bridge(coro, what: str):
    """asyncio.run + 统一拆包:任何底层异常都以带根因的 RuntimeError 抛出。"""
    try:
        return asyncio.run(coro)
    except BaseException as e:
        raise RuntimeError(f"MCP {what} 失败,根因 -> {_explain(e)}") from e


def get_mcp_tools_definition() -> list[dict]:
    """同步:拉取**所有已配置 MCP 服务**的工具并转 OpenAI function 格式。"""
    return _run_bridge(_list_tools_openai(load_server_specs()), "工具列表获取")


def call_mcp_tool(tool_name: str, arguments: dict) -> str:
    """同步:让对应的 MCP 服务执行工具。

    tool_name 支持带命名空间(mcp__<server>__<tool>)—— 按 server 路由到正确的
    子进程;不带命名空间的裸名则交给第一个服务(兼容改造前的调用方式)。
    """
    specs = load_server_specs()
    spec, raw_name = split_namespaced(tool_name, specs)

    if spec is None:
        if tool_name.startswith("mcp__"):
            # 命名空间指向一个没配置的服务:报错要说清原因,别静默路由到别的服务
            raise RuntimeError(
                f"MCP 工具 {tool_name} 对应的服务未在 MCP_SERVERS 中配置"
                f"（已配置: {', '.join(s.name for s in specs)}）"
            )
        spec = specs[0]

    return _run_bridge(
        _call_tool_text(spec, raw_name, arguments or {}), f"工具调用 {raw_name}"
    )


def main():
    """演示:列出所有服务的工具清单,并调用一个命名空间工具。

    ⚠️ 必须是**同步**函数:它调用的 get_mcp_tools_definition / call_mcp_tool 内部
    自己 asyncio.run,在已运行的事件循环里再调会直接抛
    "asyncio.run() cannot be called from a running event loop"。
    （原实现是 `async def main` + `asyncio.run(main())`,所以这个演示脚本一直跑不起来。）
    """
    specs = load_server_specs()
    print("==== ① 已配置的 MCP 服务 ====")
    for s in specs:
        print(f"    - {s.name}: {s.command} {' '.join(s.args)}")

    print("==== ② 工具清单(mcp__ 命名空间) ====")
    for t in get_mcp_tools_definition():
        print(f"    - {t['function']['name']}: {t['function']['description'][:60]}")

    print("==== ③ 调用订单工具 ====")
    try:
        text = call_mcp_tool(
            "mcp__order__query_order", {"order_no": "ORD-20250820-001"}
        )
        print(text)
    except Exception as e:  # noqa: BLE001
        print(f"    调用失败: {e}")

    print("==== 结束 ====")


if __name__ == "__main__":
    main()
