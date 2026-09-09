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
import asyncio
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters

# 服务端脚本绝对路径 + 服务端名字(与 mcp_server.py 里 MCPServer("mcp_server") 一致)
SERVER_FILE = Path(__file__).resolve().parent / "mcp_server.py"
SERVER_NAME = "mcp_server"


def _server_params() -> StdioServerParameters:
    """描述如何启动服务端子进程。"""
    return StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_FILE)],
    )


async def _list_tools_openai() -> list[dict]:
    """异步内芯:列出工具并转成 OpenAI/智谱 function 格式(带 mcp__ 命名空间)。"""
    async with Client(_server_params()) as client:
        tools_result = await client.list_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": f"mcp__{SERVER_NAME}__{tool.name}",
                    "description": tool.description or "",
                    "parameters": tool.input_schema
                    or {"type": "object", "properties": {}},
                },
            }
            for tool in tools_result.tools
        ]


async def _call_tool_text(raw_name: str, arguments: dict) -> str:
    """异步内芯:调用工具,返回文本结果;工具报错(is_error)则抛异常。"""
    async with Client(_server_params()) as client:
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
    """同步:拉取 MCP 工具清单并转 OpenAI function 格式(喂给 LLM)。"""
    return _run_bridge(_list_tools_openai(), "工具列表获取")


def call_mcp_tool(tool_name: str, arguments: dict) -> str:
    """同步:让 MCP 服务端执行工具。

    tool_name 支持带命名空间(mcp__<server>__<tool>),内部自动剥前缀后调用。
    """
    raw_name = tool_name.split("__")[-1] if tool_name.startswith("mcp__") else tool_name
    return _run_bridge(_call_tool_text(raw_name, arguments or {}), f"工具调用 {raw_name}")


async def main():
    """演示 5 步流程(现在走同步桥,调用处与图节点一致)。"""
    print("==== ① 工具清单(mcp__ 命名空间) ====")
    tools = get_mcp_tools_definition()
    for t in tools:
        print(f"    - {t['function']['name']}: {t['function']['description']}")

    print("==== ② 调用命名空间工具 ====")
    text = call_mcp_tool(f"mcp__{SERVER_NAME}__add", {"a": 3, "b": 5})
    print(f"    返回: {text}")

    print("==== 结束 ====")


if __name__ == "__main__":
    asyncio.run(main())
