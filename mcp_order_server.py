# mcp_order_server.py — 订单域 MCP 服务端(stdio 模式)
#
# 为什么单独一个 server 而不是塞进 mcp_server.py：
#   MCP 的设计就是"一个 server 一个能力边界"。订单/物流/退款属于**外部业务系统**，
#   和 add 这类通用工具不是一回事 —— 接真实系统时它是另一个团队、另一套鉴权、
#   另一个部署单元。分开之后：
#     · 工具名带独立命名空间 mcp__order__<tool>，不会与通用工具混淆；
#     · 订单服务挂掉不影响通用工具（客户端按 server 逐个降级，见 mcp_client.py）。
#
# ⚠️ 数据是**模拟的**（见 mcp_order_data.py）：不连任何真实订单系统。
#    它存在是为了把"工具调用链路"跑通并可回归测试，不是为了提供业务真相。
#
# 运行方式：不需要手动启动，由 mcp_client.py 以子进程方式自动拉起。
# 单独调试：python mcp_order_server.py
#
# ⚠️ stdio 模式下 stdout 是 MCP 协议通道，**绝对不能 print**！
#    需要日志请用 sys.stderr 或 logging，否则会污染协议流让客户端解码崩溃。

import os

from mcp.server import MCPServer

from mcp_order_data import (
    CALLER_ENV_VAR,
    UNIDENTIFIED,
    describe_logistics,
    describe_order,
    describe_recent_orders,
    describe_refund,
)

# 服务端名字：客户端据此拼命名空间 mcp__order__<tool>，
# 必须与 .env 的 MCP_SERVERS 里这个 server 的 serverName 一致。
SERVER_NAME = "order"

mcp = MCPServer(SERVER_NAME)


def _caller() -> str | None:
    """当前调用方身份。

    ⚠️ 来源是**本进程的环境变量**（由 `mcp_client.call_mcp_tool` 注入），
    而不是工具参数 —— 模型无法影响子进程的环境，所以也无法伪造成别人。
    真实部署里这个值应当来自认证后的 principal；本仓库由 graph 节点传入会话 id，
    再由数据层的 `resolve_caller` 决定映射（演示模式 = 单租户）。
    """
    return (os.getenv(CALLER_ENV_VAR) or "").strip() or None


@mcp.tool()
def query_order(order_no: str) -> str:
    """按订单号查询订单详情：状态、下单时间、商品、金额。

    使用场景：
    - 用户问"我的订单到哪了"、"订单状态是什么"、"我买的东西发货了吗"
    - 用户问订单金额、下单时间、买了什么商品
    - 用户提供了订单号（形如 ORD-20250820-001）

    不适用场景：
    - 只问物流轨迹/运单号 → 用 query_logistics
    - 只问退款到没到账 → 用 query_refund
    - 用户不记得订单号 → 用 query_recent_orders 查**他本人名下**的订单

    Args:
        order_no: 订单号，格式 ORD-YYYYMMDD-NNN（如 ORD-20250820-001）。
            大小写和前后空格会被自动处理

    返回：订单详情文本；订单号格式错误或查不到时会明确说明是哪一种。
    """
    caller = _caller()
    if caller is None:
        # 拿不到身份就**失败关闭**：数据层的 customer_id=None 是"不过滤"的
        # 宽松路径（只该给单测直调用），绝不能从这里走下去。
        return UNIDENTIFIED
    return describe_order(order_no, customer_id=caller)


@mcp.tool()
def query_logistics(order_no: str) -> str:
    """按订单号查询物流进度：承运商、运单号、发货与签收时间、运输轨迹。

    使用场景：
    - 用户问"物流到哪了"、"什么时候能到"、"快递单号是多少"
    - 用户说快递很久没动了（本工具会在在途超期时给出提醒）

    不适用场景：
    - 问订单本身的状态/金额 → 用 query_order
    - 问退款进度 → 用 query_refund

    Args:
        order_no: 订单号，格式 ORD-YYYYMMDD-NNN。大小写与空格会自动处理

    返回：物流进度文本；订单未发货时会说明原因，在途超期时会给出建议。
    """
    caller = _caller()
    if caller is None:
        return UNIDENTIFIED
    return describe_logistics(order_no, customer_id=caller)


@mcp.tool()
def query_refund(order_no: str) -> str:
    """按订单号查询退款进度：退款状态、金额、申请时间、预计到账时间。

    使用场景：
    - 用户问"退款到账了吗"、"退款要多久"、"为什么还没退款"
    - 用户说退款拖了很久（本工具会在超过预计到账时间时给出提醒）

    不适用场景：
    - 问订单状态或物流 → 用 query_order / query_logistics
    - 用户还没申请退款，只是想了解退货政策 → 查知识库，不要调用本工具

    Args:
        order_no: 订单号，格式 ORD-YYYYMMDD-NNN。大小写与空格会自动处理

    返回：退款进度文本；该订单没有退款记录时会明确说明。
    """
    caller = _caller()
    if caller is None:
        return UNIDENTIFIED
    return describe_refund(order_no, customer_id=caller)


@mcp.tool()
def query_recent_orders(limit: int = 3) -> str:
    """列出**当前用户本人名下**的最近订单，用于用户不记得订单号的场景。

    使用场景：
    - 用户说"我最近买的东西到哪了"但没给订单号

    不适用场景：
    - 用户已给出订单号 → 直接用 query_order / query_logistics / query_refund

    安全边界：
    - 本工具**不接受手机号、也不接受任何"查谁"的参数**：查询对象由服务端身份
      决定，模型只能决定"返回几笔"。以前这里有一个 phone 参数，那等于开放了
      一个"按手机号枚举他人订单"的接口。

    Args:
        limit: 最多返回几笔，默认 3（上限 20）

    返回：订单号 + 状态 + 下单时间 + 金额的列表。
    """
    caller = _caller()
    if caller is None:
        return UNIDENTIFIED
    return describe_recent_orders(limit=limit, customer_id=caller)


if __name__ == "__main__":
    mcp.run()  # 监听 stdin、回复 stdout，一直服务直到客户端断开
