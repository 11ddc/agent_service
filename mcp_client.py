# mcp_client.py
import asyncio

from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client

# 全局变量（懒加载）
_client = None
_read = None
_write = None
_lock = asyncio.Lock()

# mcp客户端负责启动服务端子进程，通过管道通信，调用工具
# 因为 stdio 模式下，通信双方必须由同一个父进程管理管道，
# 所以客户端必须主动拉起服务端进程。如果你的服务端改为 HTTP 模式（mcp.run(transport="http")）
# 客户端就只需指定 URL，不再需要启动命令了。


async def get_client():
    """获取全局唯一的客户端实例（懒加载）"""
    global _client, _read, _write
    async with _lock:
        if _client is None:
            # MCP 服务器子进程
            # 当需要连接时，执行 python mcp_server.py 来启动服务器
            server_params = StdioServerParameters(
                command="python",
                args=["mcp_server.py"],  # 你的服务器文件名，确保路径正确
            )
            # 启动子进程创建标准输入输出的管道，返回一个异步上下文管理器
            #   _read  ← 从 mcp_server.py 的 stdout 读取数据
            #   _write → 向 mcp_server.py 的 stdin 写入数据
            _read, _write = await stdio_client(server_params).__aenter__()
            # 创建 MCP 协议客户端 对象
            # __aenter__手动进入上下文管理器
            _client = Client(_read, _write, mode="auto")
            await _client.__aenter__()
            print(" MCP 客户端初始化成功！")
        return _client


async def get_mcp_tools_definition():
    """
    获取 MCP 工具列表，并转换为 OpenAI / 智谱 API 的 function 格式。
    返回一个列表，每个元素为：
    {
        "type": "function",
        "function": {
            "name": ...,
            "description": ...,
            "parameters": {...}   # JSON Schema
        }
    }
    """
    # 初始化客户端
    client = await get_client()
    # 获取 MCP 工具列表
    tools_result = await client.list_tools()
    openai_tools = []
    for tool in tools_result.tools:
        # 直接使用 tool.inputSchema（已经是 JSON Schema）
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema
                    or {"type": "object", "properties": {}},
                },
            }
        )
        print(f"mcp tool——name: {openai_tools}")

    return openai_tools


async def call_mcp_tool(tool_name: str, arguments: dict):
    """调用 MCP 工具并返回文本结果"""
    client = await get_client()
    result = await client.call_tool(tool_name, arguments=arguments)
    # 提取文本内容（根据实际返回结构调整）
    if result.content and hasattr(result.content[0], "text"):
        return result.content[0].text
    return str(result.content)
