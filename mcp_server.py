# mcp_server.py — MCP 服务端(stdio 模式)
#
# 作用:定义"我提供哪些工具"。运行方式:不需要手动启动,
#       由客户端(mcp_client.py)以子进程方式自动拉起,通过 stdin/stdout 管道通信。
#
# 用法:python mcp_server.py      # 也可以手动单独跑(会一直等待客户端的请求)

from mcp.server import MCPServer

# 服务端实例:名字 mcp_server 会在工具命名空间里用到(见 mcp_client.py)
mcp = MCPServer("mcp_server")


# @mcp.tool() 把普通函数注册成 MCP 工具,三样东西会变成工具的"说明书":
#   1. 函数名 → 工具名(客户端用这个名字调用)
#   2. 类型注解 → 参数的 JSON Schema(LLM 靠它知道传什么参数)
#   3. docstring → 工具描述(LLM 靠它判断何时该调用这个工具)
@mcp.tool()
def add(a: int, b: int) -> int:
    """两个整数相加,返回它们的和。

    使用场景:用户问"3 加 5 等于几"这类算术问题时调用。
    """
    # 注意:stdio 模式下 stdout 是 MCP 协议通道,绝对不能 print!
    # 需要看日志请用 sys.stderr 或 logging(否则会污染协议流,客户端 UTF-8 解码崩溃)。
    return a + b


if __name__ == "__main__":
    mcp.run()  # 监听 stdin、回复 stdout,一直服务直到客户端断开
