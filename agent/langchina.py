import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from tools_agent.kb_tools import KB_TOOLS

load_dotenv(encoding="utf-8-sig")  # utf-8-sig:兼容带 BOM 的 .env(键名不会被 \ufeff 污染)

# ── 初始化模型（全局单例）────────────────────────────────
model = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    # 概率分布 越高越容易胡言乱语
    temperature=0.7,
)


# ── Agent 上下文 ─────────────────────────────────────────
@dataclass
class Context:
    question: str
    user_id: str = "default_user"
    messages: list = field(default_factory=list)


# ── 工具集 ───────────────────────────────────────────────
# 知识库相关工具统一来自 tools_agent/kb_tools.py（基于真实数据、有单测守着）。
# ⚠️ 这里原本自己写了一个 search_knowledge_base，但 `return retrieve(query)`
# 返回的是**未 await 的协程**（retrieve 是 async def）；而且 create_agent 当时
# 没有传 tools，两个 @tool 从未注册 —— 兜底 Agent 其实查不了知识库，
# 而它的职责恰恰是"检索/路由失败时兜底回答"。
@tool
def get_knowledge_base_status() -> str:
    """
    查看当前知识库的状态：是否已初始化、包含多少文档块。

    使用场景：
    - 用户问"知识库里有多少资料 / 初始化了吗"
    - 你想确认库里到底有没有内容可查时

    Args:
        （无参数）

    返回：初始化状态、文档块数量与存储目录。
    """
    from rag.rag import get_status

    status = get_status()
    if not status["initialized"]:
        return "知识库尚未初始化。"
    if status["document_count"] == 0:
        return f"知识库已初始化，但目录（{status['knowledge_base_dir']}）中暂无文档。"
    return (
        f"知识库状态：已初始化\n"
        f"- 文档块数量：{status['document_count']}\n"
        f"- 存储目录：{status['knowledge_base_dir']}"
    )


# 兜底 Agent 的工具列表：**必须显式传给 create_agent**，否则工具不生效。
AGENT_TOOLS = [*KB_TOOLS, get_knowledge_base_status]


# ── 创建 Agent（全局单例）────────────────────────────────
agent = create_agent(
    model=model,
    tools=AGENT_TOOLS,
    checkpointer=InMemorySaver(),
    context_schema=Context,
)
