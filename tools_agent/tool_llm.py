import os

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages.utils import convert_to_openai_messages
from langchain_core.utils.function_calling import convert_to_openai_tool

# ── MCP 接入点①（新增）：外部 MCP 工具桥，见 tools_agent/mcp_client.py
# from tools_agent import mcp_client
from openai import OpenAI

from mcp_client import get_mcp_tools_definition

# 有这个才能进env文件读取内容
load_dotenv(encoding="utf-8-sig")  # utf-8-sig:兼容带 BOM 的 .env

TOOL_MAX = 5


zhipu_client = OpenAI(
    api_key=os.getenv("ZHI_PU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)


# tool装饰器会把函数包装成一个工具对象（StructuredTool实例），方便llm调用
@tool
def searchOrder(query: str, session_id: str) -> str:
    """查询用户的订单信息，返回订单状态、物流进度和商品明细。

    使用场景：
    - 用户询问订单状态、物流进度、到货时间、订单详情时调用
    - 用户提供了订单号（如 ORD-20260831-001）希望查询时调用
    - 用户说"查一下我的订单""我买的东西到哪了"等类似表达时调用

    不适用场景：
    - 用户咨询商品参数、价格、库存时不要调用（使用商品查询工具）
    - 用户要求修改、取消、退款时不要调用（使用订单变更工具）
    - 闲聊或与订单无关的问题不要调用

    返回内容：
    订单号、当前状态（待支付/已发货/已完成）、物流公司及单号、
    商品清单、下单时间、订单金额。

    Args:
        query: 查询关键词，可以是订单号（如 ORD-20260831-001）、
            商品名称，或"最近的订单""上周买的东西"等自然语言描述
        session_id: 当前会话唯一标识，用于关联该用户的订单数据，
            由系统运行时自动传入，模型无需生成
    """

    return "搜索成功，您的订单为。。。。。。。。。。。。"


@tool
def add(query: str, session_id: str) -> int:
    """Adds `query` and `session_id`.

    Args:
        query: First int
        session_id: Second int
    """
    return "调用add函数成功"


tools = [
    # 将 langchain 工具转换为 openai 工具，方便llm调用
    convert_to_openai_tool(t)
    for t in [add, searchOrder]
]


def call_zhipu_chat(messages: list):
    # 格式转换(同步:内部用 asyncio.run 拉 MCP 工具,见 mcp_client.py)
    mcp_tools = get_mcp_tools_definition()
    print(f"MCP工具列表，mcp_tools: {mcp_tools}")

    payload = convert_to_openai_messages(messages)
    print("调用智谱chat模型，messages:", payload)
    res = zhipu_client.chat.completions.create(
        model="glm-4.5-air",
        messages=payload,
        # ── MCP 接入点①（新增）：本地工具 + MCP 工具合并喂给 LLM；未配置 MCP 时等于原 tools
        tools=[*tools, *mcp_tools],
        # 让llm自动调用工具
        tool_choice="auto",
    )

    return res
