from mcp.server import MCPServer

mcp = MCPServer("mcp_server")


# mcp服务端，提供工具，监听 stdin，回复 stdout


@mcp.tool()
def add(data) -> str:
    """返回两个整数的和"""
    return "两数之和:", data


if __name__ == "__main__":
    mcp.run()
