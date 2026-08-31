from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from agent.langchina import agent
from intent.classifier import classify
from intent.problemdecomposition import split_questions
from intent.schemas import IntentName, IntentResult
from query_rewrite.history import get_history
from query_rewrite.rewriter import rewrite_query
from rag.generatellm import RAGGenerator
from rag.rag import KnowledgeBaseError, get_status, reordering, retrieve_sync

# ── 非知识库意图的短路回复话术 ────────────────────────────
SHORT_CIRCUIT_REPLIES = {
    IntentName.CHITCHAT: "你好呀，我是智能助手，可以回答产品、售后、规则、业务流程等方面的问题，有什么可以帮您？",
    IntentName.HUMAN_HANDOFF: "好的，正在为您转接人工客服，请稍候。您也可以先留下问题，我会一并转达。",
    IntentName.OUT_OF_SCOPE: "抱歉，这个问题超出了我的服务范围。您可以咨询产品、售后、业务流程相关的问题，或者让我查一下知识库里的资料。",
}


# 定义规则
class AgentState(TypedDict):
    question: str  # 单问题
    session_id: str
    sub_questions: list[str] | None  # 多问题 Optional 或的意思
    idx: int | None  # 问题序号
    intent_result: IntentResult | None  # 意图
    contexts: str | None  # 检索得到的答案
    answer: str | None  # 响应
    meta: dict | None  # 元数组数据


Generator = RAGGenerator()


# query改写节点
def rewrite_node(state: AgentState) -> dict:
    session_id = state.get("session_id") or ""
    question = state.get("question") or ""

    if not session_id:
        return {"question": question}
    # 返回会话历史
    history = get_history(session_id)

    return {"question": rewrite_query(question, history)}


def splitter_node(state: AgentState) -> dict:
    """问题拆分节点：多问题 → 子问题列表；单问题 → 空列表。"""
    querys = split_questions(state["question"])
    return {"sub_questions": list(querys) if querys else []}


def route_after_split(state: AgentState) -> str:
    """拆分后判断：多问题 → multi_loop；单问题 → 单问题子图。question_flow"""
    return (
        "multi_loop" if len(state.get("sub_questions") or []) > 1 else "question_flow"
    )


def intent_router_node(state: AgentState) -> dict:
    """意图识别节点：三级漏斗（规则→Embedding→LLM）；失败置 None 交给兜底。"""
    try:
        result = classify(state["question"])
    except Exception as e:
        print(f"意图识别失败：{e}")
        result = None
    return {"intent_result": result}


# 根据意图返回对应状态
def route_by_intent(state: AgentState) -> str:
    """路由器：读 intent_result 决定下一个节点（对应原 _handle_intent 分支）。"""
    result = state.get("intent_result")
    if result is None:
        return "agent_flow"  # 识别失败 → Agent 兜底
    intent = result.intent
    if intent == IntentName.TOOL_CALL:
        return "tool_agent"
    if intent == IntentName.AMBIGUOUS:
        if result.confidence == 0:
            return "short_circuit"  # 空问题
        return "agent_flow"  # 意图模糊 → LLM 兜底
    if intent == IntentName.KB_QUESTION:
        return "rag_flow"  # 知识库问答 → RAG
    return "short_circuit"  # chitchat / handoff / out_of_scope / status


# 工具调用意图
def tool_call_node(state: AgentState) -> dict:

    return True


# llm判断什么时候调用工具，调用什么工具
def llm_call_node(state: AgentState) -> dict:
    # llm需要用一个单独输出  来让多问题循环的时候接收使用
    # 从身份推导（最好）：用户是登录态，user_id 在 config 里——你可以直接查他的订单列表，根本不该让用户报订单号
    # 从对话历史里拿：用户三轮前说过的订单号，LLM 能从 messages 里捡回来
    # 问用户（兜底）：实在没有再问，而且要告诉用户去哪找（“在 我的-订单 页面可查看”）

    return True


# llm判断是否需要调用
def tool_continue(state: AgentState) -> dict:

    return True


# 3 个字典回复 + 2 个特判分支"覆盖了 5 个意图值
def short_circuit_node(state: AgentState) -> dict:
    result = state.get("intent_result")
    intent = result.intent if result else None
    if intent == IntentName.STATUS_QUERY:  # 状态查询：实时查库拼话术
        status = get_status()
        text = (
            (
                f"知识库已初始化，目前包含 {status['document_count']} 个文档块，"
                f"存储目录：{status['knowledge_base_dir']}。"
            )
            if status.get("initialized")
            else "知识库尚未初始化，请先上传文档。"
        )
        return {"answer": text}
    if intent == IntentName.AMBIGUOUS and result and result.confidence == 0:
        return {"answer": "您好，我没收到您的问题，请重新输入您想查询的内容。"}
    return {
        "answer": SHORT_CIRCUIT_REPLIES.get(intent, "抱歉，我暂时无法回答这个问题。")
    }


# 调用检索接口
def rag_node(state: AgentState) -> dict:
    """RAG 节点（薄图版）：直接复用 retrieve_sync（多路召回+RRF 融合），行为与现在一致。"""
    try:
        answer = retrieve_sync(state["question"])
        if not answer:
            return {"answer": "未在知识库中找到与问题相关的内容。"}
        # 将检索到的答案进行重排序
        docs = reordering(state["question"], answer)
        # 生成
        result = Generator.generate_answer(state["question"], docs)

    except KnowledgeBaseError as e:
        print(f"知识库不可用: {e}")
        result = str(e)  # 连接失败/空库 → 把具体提示语原样返回给用户
    except Exception as e:
        print(f"知识库检索失败: {e}")
        result = "知识库检索失败，请稍后重试。"
    return {"answer": result}


def agent_node(state: AgentState) -> dict:
    """Agent/LLM 兜底节点：DeepSeek Agent 单独回答（与原 _answer_by_agent 一致）。"""
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": state["question"]}]},
            config={"configurable": {"thread_id": state["session_id"]}},
        )
        return {"answer": result["messages"][-1].content}
    except Exception as e:
        print(f"Agent 回答失败: {e}")
        return {"answer": "抱歉，我暂时无法回答这个问题，请稍后重试。"}


def merge_node(state: AgentState) -> dict:
    """汇总节点：组装响应附带信息（对齐原 _response_with_intent）。"""
    result = state.get("intent_result")
    meta = {"intent": "unknown", "method": "fallback"}
    if result is not None:
        meta = {
            "intent": result.intent.value,
            "method": result.method,
            "confidence": round(result.confidence, 4),
        }
        slots = result.slots
        if slots is not None and (slots.source or slots.keyword or slots.time_range):
            meta["slots"] = slots.model_dump()
    return {"answer": state.get("answer") or "", "meta": meta}


def multi_loop_node(state: AgentState) -> dict:
    """多问题节点（薄图版）：顺序调用单问题子图，每个子问题走完整流程，最后合并。"""
    parts, last_meta = [], None
    for q in state["sub_questions"]:
        print("多问题：", q)
        # 循环走单问题子图
        r = question_graph.invoke({"question": q, "session_id": state["session_id"]})
        parts.append(r.get("answer") or "")
        last_meta = r.get("meta") or last_meta
    return {"answer": "\n".join(parts) or None, "meta": last_meta or {}}


# tool_agent子图
def build_tool_agent_graph():
    g = StateGraph(AgentState)
    g.add_node("tool_call", tool_call_node)
    g.add_node("llm_call", llm_call_node)

    g.add_edge(START, "llm_call")
    g.add_conditional_edges("llm_call", tool_continue, ["tool_call", END])
    # 从tool->llm
    g.add_edge("tool_call", "llm_call")
    return g.compile()


def build_question_graph():
    """单问题子图：意图识别(路由器) → 各意图分支 → 汇总。"""
    g = StateGraph(AgentState)
    g.add_node("intent_router", intent_router_node)
    g.add_node("short_circuit", short_circuit_node)
    g.add_node("rag_flow", rag_node)  # 版本 B：换成 rag_subgraph
    g.add_node("agent_flow", agent_node)
    g.add_node("merge", merge_node)
    g.add_node("tool_agent", tool_agent)

    g.add_edge(START, "intent_router")
    # 从intent_router节点出发，route_by_intent返回哪个节点 就去哪个节点
    # 意图判定后续路线
    g.add_conditional_edges(
        "intent_router",
        route_by_intent,
        {
            "tool_agent": "tool_agent",
            "rag_flow": "rag_flow",
            "agent_flow": "agent_flow",
            "short_circuit": "short_circuit",
        },
    )
    # 这里是循环给每一个节点都定义上一条指向merge节点的边
    for n in ("rag_flow", "agent_flow", "short_circuit", "tool_agent"):
        g.add_edge(n, "merge")
    g.add_edge("merge", END)
    # 编译成可执行的工作流对象
    return g.compile()


# 如果子图是作为节点嵌入到主图中 那么状态就是共享的 如question_flow节点
# question_graph.invoke({"question": q, "session_id": state["session_id"]}) 这个是独立状态  新的AgentState参数
def build_main_graph():
    """主图：拆分 → 路由（单问题直达 / 多问题循环）。"""
    g = StateGraph(AgentState)
    g.add_node("splitter", splitter_node)
    g.add_node("question_flow", question_graph)  # 编译后的子图直接作为节点
    g.add_node("multi_loop", multi_loop_node)
    g.add_node("rewrite_node", rewrite_node)

    g.add_edge(START, "rewrite_node")
    g.add_edge("rewrite_node", "splitter")
    g.add_conditional_edges(
        "splitter",
        route_after_split,
        {
            "question_flow": "question_flow",
            "multi_loop": "multi_loop",
        },
    )
    g.add_edge("question_flow", END)
    g.add_edge("multi_loop", END)
    return g.compile()


tool_agent = build_tool_agent_graph()
question_graph = build_question_graph()
graph = build_main_graph()
