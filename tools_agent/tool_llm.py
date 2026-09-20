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


# ⚠️ 下面两个工具是**占位实现**，没有接真实订单系统。
#
# 它们曾经被注册进文件末尾那个喂给模型的 tools 列表，于是用户问"我的订单到哪了"时：
# 模型调用 searchOrder → 拿到 "搜索成功，您的订单为。。。。。。。。。。。。" 这种假数据
# → 再据此编出一段语气自信的订单状态回答。而接口返回 200、日志一切正常 ——
# 对客服机器人来说这是最坏的一类失败：用户被喂了编造的业务数据，且无人察觉。
#
# 现在的约定（两层防护）：
#   ① 不进 tools 列表（见文件末尾 _LOCAL_TOOLS）；
#   ② 函数体直接抛错 —— 即使将来有人误把它注册回去，也会当场炸出来，
#      而不是静默返回假数据。
# 接入真实订单系统后：把实现补上，并把名字加进 _LOCAL_TOOLS 即可。
@tool
def searchOrder(query: str, session_id: str) -> str:
    """（占位，未实现）查询用户的订单信息，返回订单状态、物流进度和商品明细。

    留档的接口约定（接入真实系统时按此实现）：

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
    raise NotImplementedError(
        "searchOrder 尚未接入真实订单系统：请勿注册给模型，也不得返回假数据"
    )


@tool
def add(query: str, session_id: str) -> int:
    """（占位，未实现）早期调试用的示例工具，没有任何业务语义。"""
    raise NotImplementedError("add 是示例占位工具，没有实际实现")


# 暴露给模型的**本地**工具列表：只放有真实实现的工具。
# 上面两个占位工具刻意不在这里 —— 它们的存在曾让用户拿到编造的订单答案。
_LOCAL_TOOLS: list = []

tools = [
    # 将 langchain 工具转换为 openai 工具，方便llm调用
    convert_to_openai_tool(t)
    for t in _LOCAL_TOOLS
]


def call_zhipu_chat(messages: list):
    # 格式转换(同步:内部用 asyncio.run 拉 MCP 工具,见 mcp_client.py)
    mcp_tools = get_mcp_tools_definition()
    print(f"MCP工具列表，mcp_tools: {mcp_tools}")

    payload = convert_to_openai_messages(messages)
    print("调用智谱chat模型，messages:", payload)

    # ── MCP 接入点①：本地工具 + MCP 工具合并喂给 LLM
    # 合并后为空时**不要传 tools/tool_choice**：部分 OpenAI 兼容端点对空数组直接回 400。
    # 本地工具目前是空的占位状态（见 _LOCAL_TOOLS），所以这条分支是常态而不是边角。
    merged_tools = [*tools, *mcp_tools]
    extra = {"tools": merged_tools, "tool_choice": "auto"} if merged_tools else {}
    res = zhipu_client.chat.completions.create(
        model="glm-4.5-air",
        messages=payload,
        **extra,
    )

    return res
