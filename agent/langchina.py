from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from dataclasses import dataclass, field
import os
from dotenv import load_dotenv

load_dotenv()

# ── 初始化模型（全局单例）────────────────────────────────
model = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    #概率分布 越高越容易胡言乱语
    temperature=0.7
)

# ── Agent 上下文 ─────────────────────────────────────────
@dataclass
class Context:
    question: str
    user_id: str = "default_user"
    messages: list = field(default_factory=list)

# ── 定义 RAG 工具 ────────────────────────────────────────
@tool
def search_knowledge_base(query: str) -> str:
    """
    搜索本地知识库，获取与查询相关的文档内容。

    使用场景：
    - 用户询问的是特定文档/专业知识领域的问题
    - 需要从公司内部文档、技术手册、规章制度等资料中查找答案
    - 你不确定某个知识点时，优先查询知识库

    参数:
        query: 搜索查询文本（可以是关键词或完整问题）

    返回:
        知识库中检索到的相关文档片段，或提示未找到。
    """
    from rag.rag import retrieve
    return retrieve(query)


@tool
def get_knowledge_base_status() -> str:
    """
    查看当前知识库的状态：是否已初始化、包含多少文档。

    当你需要了解知识库有什么内容可用时调用此工具。
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


# ── 创建 Agent（全局单例）────────────────────────────────
agent = create_agent(
    model=model,
    # tools=[search_knowledge_base, get_knowledge_base_status],
    checkpointer=InMemorySaver(),
    context_schema=Context
)
