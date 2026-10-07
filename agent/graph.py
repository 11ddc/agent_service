import json
import logging
import os
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agent.events import (  # 流式事件旁路（同步链路下 emit 是空操作）
    current_emitter,
    emit,
)
from agent.langchina import agent
from context_budget import input_budget
from intent.classifier import classify
from intent.problemdecomposition import split_questions
from intent.schemas import IntentName, IntentReason, IntentResult
from mcp_client import call_mcp_tool
from query_rewrite.history import get_history
from query_rewrite.rewriter import rewrite_query
from rag.generatellm import RAGGenerator
from rag.rag import KnowledgeBaseError, get_status, reordering, retrieve_sync

logger = logging.getLogger(__name__)

# ── 图的步数硬上限 ────────────────────────────────────────────
# 工具循环 `llm_call ⇄ tool_call` 必须有个上限：不设的话，langgraph 的
# DEFAULT_RECURSION_LIMIT（10007）就是实际生效值 —— 模型只要反复请求工具，
# 就能打出上千轮 LLM 调用（每轮都要花钱、都要占住一个工作线程）。
# `TOOL_MAX` 限的是"工具轮次"，这里限的是"整图步数"，两者互为兜底。
GRAPH_RECURSION_LIMIT = int(os.getenv("GRAPH_RECURSION_LIMIT", "25"))

# 兜底 Agent（LangChain `create_agent`）自己的步数上限。
# 不显式传的话用的是它自己的默认值（langchain 内部是 9999）—— 与工具循环同理，
# 一个想反复调工具的模型能跑上千步。实测：编译图**自己的** config 会生效，
# 所以这里必须传，不能指望上层 invoke 的 recursion_limit 兜住。
AGENT_RECURSION_LIMIT = int(os.getenv("AGENT_RECURSION_LIMIT", "12"))

# 单个请求最多处理几个子问题。子问题数量由 LLM 输出决定，
# 不设上限 = "一次请求跑 N 遍完整 RAG"，成本与延迟都被模型放大。
MAX_SUB_QUESTIONS = int(os.getenv("MAX_SUB_QUESTIONS", "5"))

# ── MCP 接入点②（新增）：外部 MCP 工具桥，见 tools_agent/mcp_client.py
# from tools_agent import mcp_client
from tools_agent.tool_llm import LOCAL_TOOL_MAP, TOOL_MAX, call_zhipu_chat

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
    tool_rounds: int | None  # 工具循环已进行的轮次（硬上限见 GRAPH_RECURSION_LIMIT）
    principal_id: str | None  # **已认证的**身份标识（由 API 层注入；绝不来自客户端）
    customer_id: str | None  # 已解析的业务客户号（由 API 层注入；工具据此查业务数据）
    messages: Annotated[list, add_messages]


Generator = RAGGenerator()

# ── RAG 资料预算（token）──────────────────────────────────
# 输入总预算 = 模型窗口 - 输出预留(max_tokens) - 安全边际（见 context_budget.input_budget）
# 再扣掉 RAG_PROMPT 模板与用户问题的预留，剩下的才是能给"检索资料"的额度。
# 模板约 400 token、问题约 100 token，留 1000 足够，不用精算。
RAG_DOC_BUDGET = max(1000, input_budget() - 1000)


# query改写节点
def rewrite_node(state: AgentState) -> dict:
    session_id = state.get("session_id") or ""
    question = state.get("question") or ""

    if not session_id:
        return {"question": question}
    # 返回会话历史
    history = get_history(session_id)
    # 历史原文只放 DEBUG：INFO 级别下每轮都打印整段对话既吵又涉及用户隐私
    logger.debug("会话历史 %d 条: %s", len(history), history)

    rewritten = rewrite_query(question, history)
    if rewritten != question:
        logger.info("本轮问题已改写: %r", rewritten)
    return {"question": rewritten}


def splitter_node(state: AgentState) -> dict:
    """问题拆分节点：多问题 → 子问题列表；单问题 → 空列表。"""
    querys = split_questions(state["question"])
    logger.info(f"问题拆分结果: {querys}")
    return {"sub_questions": list(querys) if querys else []}


def route_after_split(state: AgentState) -> str:
    """拆分后判断：多问题 → multi_loop；单问题 → 单问题子图。question_flow"""
    logger.info(f"最终问题：{state['question']}，拆分结果：{state.get('sub_questions')}")
    return (
        "multi_loop" if len(state.get("sub_questions") or []) > 1 else "question_flow"
    )


def intent_router_node(state: AgentState) -> dict:
    """意图识别节点：三级漏斗（规则→Embedding→LLM）；失败置 error 结果交给兜底。"""
    try:
        result = classify(state["question"])
    except Exception as e:
        # 分类器整体不可用（单层失败已在 classify 内部降级，走到这里说明更严重）
        logger.warning("意图识别异常，交 Agent 兜底: %s", e)
        result = IntentResult(
            intent=IntentName.AMBIGUOUS,
            confidence=0.0,
            method="error",
            reason=IntentReason.LLM_ERROR,
        )
    else:
        logger.info(
            "意图识别: intent=%s method=%s conf=%.3f reason=%s",
            result.intent.value,
            result.method,
            result.confidence,
            result.reason.value if result.reason else "-",
        )
    return {"intent_result": result}


# 根据意图返回对应状态
def route_by_intent(state: AgentState) -> str:
    """路由器：读 intent_result 决定下一个节点（对应原 _handle_intent 分支）。

    注意只看 reason 判断"空问题"：以前用 confidence == 0 代替，而 LLM 仲裁
    失败时 confidence 也是 0，于是真实问题会被当成空问题回一句
    "我没收到您的问题"。现在失败/低置信一律走 agent_flow 兜底。
    """
    result = state.get("intent_result")
    if result is None:
        return "agent_flow"  # 识别失败 → Agent 兜底
    intent = result.intent
    if intent == IntentName.TOOL_CALL:
        return "tool_agent"
    if intent == IntentName.AMBIGUOUS:
        if result.reason == IntentReason.EMPTY:
            return "short_circuit"  # 空问题
        return "agent_flow"  # 意图模糊 / 分类器不可用 → LLM 兜底
    if intent == IntentName.KB_QUESTION:
        return "rag_flow"  # 知识库问答 → RAG
    return "short_circuit"  # chitchat / handoff / out_of_scope / status


# 工具调用
def _as_text(result: object) -> str:
    """工具的返回值不保证是 str（可能返回 dict / 数字），而 ToolMessage 只吃 str。"""
    return result if isinstance(result, str) else str(result)


def tool_call_node(state: AgentState) -> dict:
    last_msg = state["messages"][-1]

    # 打印工具调用信息（现在 tc 是字典，直接打印即可）
    logger.info(f"tool_call_node——msg: {last_msg}")

    tool_messages = []
    # 如果 tool_calls 为 None 或空列表，直接返回空消息列表
    if not last_msg.tool_calls:
        return {"messages": []}

    for tc in last_msg.tool_calls:
        # 现在 tc 是 LangChain 格式的字典：{"name": ..., "args": {...}, "id": ...}
        tool_name = tc["name"]  # 或者 tc.get("name")
        tool_args = tc["args"]  # 已经是字典，不需要 json.loads
        tool_id = tc["id"]

        logger.info(f"调用工具：{tool_name}，参数：{tool_args}")  # 清晰打印

        # ── MCP 接入点②:外部 MCP 工具(mcp__<server>__<tool>)由 mcp_client 同步执行
        if tool_name.startswith("mcp__"):
            try:
                # 身份有两个来源，都不是模型能控制的：
                #   · principal_id  —— 已认证的账号（state 由 API 层注入）
                #   · customer_id   —— 后端已解析好的业务客户号
                # 绝不用客户端传来的 session_id：它可控，等于让调用方自选客户
                # （这正是上一轮真实越权的成因）。
                result = call_mcp_tool(
                    tool_name,
                    tool_args,
                    caller_id=state.get("principal_id"),
                    resolved_customer=state.get("customer_id"),
                )
            except Exception as e:
                logger.warning(f"MCP 工具调用失败: {e}")
                result = f"[MCP 工具执行失败: {e!s}]"
            tool_messages.append(ToolMessage(content=result, tool_call_id=tool_id))

        elif tool_name in LOCAL_TOOL_MAP:
            # 本地工具**统一分派**：只要工具进了 tool_llm._LOCAL_TOOLS 就自动可用。
            # 以前这里是两个硬编码的 elif（searchOrder / add），新增一个工具就得
            # 回来改一遍 —— 漏改的后果是工具被当成"未知工具"、模型永远调不动。
            try:
                result = LOCAL_TOOL_MAP[tool_name].invoke(tool_args)
            except Exception as e:  # noqa: BLE001
                # 工具失败要回填错误、而不是中断链路：模型看到失败原因还能换个说法
                logger.warning(f"本地工具调用失败 {tool_name}: {e}")
                result = f"[工具 {tool_name} 执行失败: {e!s}]"
            tool_messages.append(
                ToolMessage(content=_as_text(result), tool_call_id=tool_id)
            )

        elif tool_name in ("searchOrder", "add"):
            # ⚠️ 这两个是占位工具（tools_agent/tool_llm.py 里没有真实实现），已从喂给
            # 模型的 tools 列表里摘掉；这里再兜一层 —— 即使被误调用，也绝不返回假数据。
            # 以前 searchOrder 返回 "搜索成功，您的订单为。。。。。。。。。。。。"，
            # 模型会据此编出一段语气自信的订单状态回答，用户完全看不出是编的。
            result = f"[{tool_name} 暂不支持：订单查询尚未接入真实订单系统]"
            tool_messages.append(ToolMessage(content=result, tool_call_id=tool_id))

        else:
            # 未知工具**必须回一条 ToolMessage**，不能只 print 就当没事：
            # tool_continue 判的是"最后一条消息有没有 tool_calls"，如果这里什么都不追加，
            # 最后一条仍是那条带 tool_calls 的 AIMessage → tool_call → llm_call →
            # tool_call …，**无限循环**。回一条"不可用"让模型自己收尾。
            logger.info(f"未知工具：{tool_name}，忽略")
            tool_messages.append(
                ToolMessage(content=f"[工具 {tool_name} 不可用]", tool_call_id=tool_id)
            )

    return {"messages": tool_messages}


def convert_zhipu_tool_calls(zhipu_tool_calls):
    """将智谱的工具调用转换为 LangChain ToolCall 格式"""
    if not zhipu_tool_calls:
        return []

    langchain_tool_calls = []
    for tc in zhipu_tool_calls:
        # 解析 arguments 字符串为字典

        args = json.loads(tc.function.arguments)
        langchain_tool_calls.append(
            ToolCall(name=tc.function.name, args=args, id=tc.id)
        )
    return langchain_tool_calls


# 入口函数 修格式使用
def entry_node(state: AgentState) -> dict:
    return {"messages": [HumanMessage(content=state["question"])]}


# llm判断什么时候调用工具，调用什么工具
def llm_call_node(state: AgentState) -> dict:
    # llm需要用一个单独输出  来让多问题循环的时候接收使用
    # 从身份推导（最好）：用户是登录态，user_id 在 config 里——你可以直接查他的订单列表，根本不该让用户报订单号
    # 从对话历史里拿：用户三轮前说过的订单号，LLM 能从 messages 里捡回来
    # 问用户（兜底）：实在没有再问，而且要告诉用户去哪找（“在 我的-订单 页面可查看”）
    # history_msg = state["messages"]
    # user_msg = {"role": "user", "content": state["question"]}
    # full_messages = history_msg + [user_msg]
    try:
        res = call_zhipu_chat(state["messages"])
        logger.info("调用智谱chat模型完成， %s", res)
        zhipi_msg = res.choices[0].message

        logger.info("toolcalls_zhipu %s", zhipi_msg)
        new_ai_msg = AIMessage(
            content=zhipi_msg.content or "",
            # 将智谱的工具调用转换为 LangChain ToolCall 格式
            tool_calls=convert_zhipu_tool_calls(zhipi_msg.tool_calls),
        )
        logger.info("llm_call_node返回的AIMessage: %s", new_ai_msg)

        # 每进一次 llm_call 就是一轮"模型 ⇄ 工具"协商；tool_continue 用它兜住循环
        rounds = int(state.get("tool_rounds") or 0) + 1
        return {"messages": new_ai_msg, "tool_rounds": rounds}

    except Exception as e:
        logger.warning(f"调用智谱chat模型失败: {e}")
        # 上抛而不是静默返回 None:否则消息停在 HumanMessage,
        # 会被 tool_continue/data_node 当成"用户原问题"回显;抛出让外层兜底接管。
        raise


# llm判断是否需要调用工具
def tool_continue(state: AgentState) -> Literal["tool_call", "data_node"]:

    last_msg = state["messages"][-1]
    logger.info("tool_continue检查是否需要调用工具，last_msg: %s", last_msg)
    # 用 getattr 判空:最后一条可能是 HumanMessage(没有 tool_calls 属性),
    # 直接访问会抛 AttributeError 把整图带崩(原实现就是这么挂的)。
    if not getattr(last_msg, "tool_calls", None):
        return "data_node"

    # ⚠️ 工具轮次硬上限：模型每轮都可以继续请求工具（尤其是工具刚回了一条
    # "执行失败 / 请稍后重试" 之后 —— 它会换个参数再试一次）。没有这个上限，
    # 循环的唯一终点就是 langgraph 的 DEFAULT_RECURSION_LIMIT（默认 10007），
    # 也就是上千轮 LLM 调用。超过 TOOL_MAX 就交回 data_node 收尾。
    if int(state.get("tool_rounds") or 0) >= TOOL_MAX:
        return "data_node"

    return "tool_call"


# 用来和单问题图之间对接数据 和mager一样
def data_node(state: AgentState) -> dict:
    last_msg = state["messages"][-1]
    answer = last_msg.content if hasattr(last_msg, "content") else ""
    # 工具轮次已到上限、而模型仍然只想调工具（content 为空）时，必须给一句
    # 诚实的收尾，而不是把空串交给上层：空 answer 会让 api/chat.py 再调一次
    # 兜底 Agent（白跑一轮），最终还可能把空字符串发给前端。
    if not answer and getattr(last_msg, "tool_calls", None):
        answer = (
            f"抱歉，这个问题需要查询的步骤超过了上限（{TOOL_MAX} 轮），"
            f"我没能给出可靠答案。请补充更具体的信息（例如订单号）后重试。"
        )
    logger.info(f"data_node——answer: {answer}")
    # 返回包含 answer 的字典，更新状态
    return {"answer": answer}


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
    if intent == IntentName.AMBIGUOUS and result and result.reason == IntentReason.EMPTY:
        return {"answer": "您好，我没收到您的问题，请重新输入您想查询的内容。"}
    return {
        "answer": SHORT_CIRCUIT_REPLIES.get(intent, "抱歉，我暂时无法回答这个问题。")
    }


# 调用检索接口
def rag_node(state: AgentState) -> dict:
    """RAG 节点（薄图版）：直接复用 retrieve_sync（多路召回+RRF 融合），行为与现在一致。

    流式：RAG 是唯一会产出 token 流的路径。emitter 为 None 时（同步 /chat 链路）
    下面所有 emit() 都是空操作，因此行为与改造前一致。
    """
    emitter = current_emitter()  # None = 同步链路（/chat）
    try:
        # 检索+重排+装箱是纯同步耗时，先给前端一个阶段提示，避免用户盯着空白
        # 推事件给前端
        emit({"type": "status", "stage": "retrieving", "text": "正在检索知识库…"})
        answer = retrieve_sync(state["question"])
        if not answer:
            return {"answer": "未在知识库中找到与问题相关的内容。"}
        # 将检索到的答案进行重排序
        docs = reordering(state["question"], answer)

        # ── 预检：重排之后、生成之前，按 token 预算装箱 ──────────────
        # 1. pack_docs 按相关性顺序装箱，装不下的从尾部丢弃（重排已排序，丢尾=丢最不相关）
        # 2. 先丢弃后编号，保证 prompt 里的 [docN] 引用不错位
        # 3. 丢块时自动附"资料不完整"声明，避免模型给出看似完整实则缺项的答案
        # 4. 万一仍被判超限 → 预算降 1/2、1/4 重试 → 最后退到 map-reduce 分段生成
        #    （实现见 rag/generatellm.generate_answer_within_budget）
        emit({"type": "status", "stage": "generating", "text": "正在生成回答…"})
        gen = Generator.generate_answer_within_budget(
            state["question"],
            docs,
            # token预算
            budget_tokens=RAG_DOC_BUDGET,
            emit=emitter,  # None → 非流式（/chat 链路）；回调 → 边生成边推 delta
        )
        result = gen["answer"]
        logger.info(f"[RAG预算] 预算={gen['budget_tokens']} 实耗={gen['used_tokens']} "
            f"资料={gen['kept']}/{gen['kept'] + gen['dropped']} 丢弃={gen['dropped']} "
            f"截断={gen['truncated']} 降级={gen['degraded']} 重试={gen['retries']}")

        logger.info(f"RAG 生成结果: {result}")
    except KnowledgeBaseError as e:
        logger.warning(f"知识库不可用: {e}")
        result = str(e)  # 连接失败/空库 → 把具体提示语原样返回给用户
    except Exception as e:
        logger.warning(f"知识库检索失败: {e}")
        result = "知识库检索失败，请稍后重试。"
    return {"answer": result}


def agent_node(state: AgentState) -> dict:
    """Agent/LLM 兜底节点：DeepSeek Agent 单独回答（与原 _answer_by_agent 一致）。"""
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": state["question"]}]},
            config={
                "configurable": {"thread_id": state["session_id"]},
                # ⚠️ 显式给步数上限：不传就用 langchain 自己的默认（9999）。
                # 这个兜底 Agent 同样带工具，模型可以反复调用 —— 与工具循环同理，
                # 必须在这里收紧，不能指望上层的 recursion_limit 兜住。
                "recursion_limit": AGENT_RECURSION_LIMIT,
            },
        )
        return {"answer": result["messages"][-1].content}
    except Exception as e:
        # 这是 /chat 两条失败路径的最终兜底，必须留痕（原来只有 print → 线上查不到）
        logger.warning("Agent 回答失败: %s", e, exc_info=True)
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
        if result.reason is not None:
            # 兜底原因（空问题/低置信/分类器不可用）：便于线上回答"为什么走了兜底"
            meta["reason"] = result.reason.value
        slots = result.slots
        if slots is not None and (slots.source or slots.keyword or slots.time_range):
            meta["slots"] = slots.model_dump()

    return {"answer": state.get("answer") or "", "meta": meta}


def multi_loop_node(state: AgentState) -> dict:
    """多问题节点（薄图版）：顺序调用单问题子图，每个子问题走完整流程，最后合并。

    子问题数量**必须设上限**：数量来自 LLM 输出，不设限就是"一次请求跑 N 遍完整
    RAG"，成本与延迟都被模型放大。超出的部分明确告知用户，而不是假装全答了。
    """
    all_questions = list(state.get("sub_questions") or [])
    questions = all_questions[:MAX_SUB_QUESTIONS]
    dropped = len(all_questions) - len(questions)
    if dropped > 0:
        logger.warning(
            "子问题过多：收到 %d 个，只处理前 %d 个，丢弃 %d 个",
            len(all_questions),
            MAX_SUB_QUESTIONS,
            dropped,
        )

    parts, last_meta = [], None
    for i, q in enumerate(questions):
        logger.info("多问题循环： %s", q)
        if i:
            # 必须与下面 "\n".join(parts) 的分隔保持一致，否则流式正文会缺
            # 子问题之间的换行（同步链路下 emit 是空操作，无影响）。
            emit({"type": "delta", "content": "\n"})
        # 循环走单问题子图。子图是同步调用、同一个线程 → 子图节点里的 emit()
        # 能直接读到本线程的 emitter，不需要额外透传参数。
        # config 也带上传：每次 invoke 是一份**独立**的步数预算，不带就退回 10007。
        r = question_graph.invoke(
            {
                "question": q,
                "session_id": state["session_id"],
                # 身份必须透传：否则子图里的工具调用拿不到 principal（会失败关闭）
                "principal_id": state.get("principal_id"),
                "customer_id": state.get("customer_id"),
            },
            config={"recursion_limit": GRAPH_RECURSION_LIMIT},
        )
        parts.append(r.get("answer") or "")
        last_meta = r.get("meta") or last_meta

    answer = "\n".join(parts) or None
    if answer and dropped:
        answer = (
            f"{answer}\n\n（本次问题较多，我只处理了前 {len(questions)} 个，"
            f"其余 {dropped} 个请分次提问。）"
        )
    return {"answer": answer, "meta": last_meta or {}}


# tool_agent子图
def build_tool_agent_graph():
    g = StateGraph(AgentState)
    g.add_node("tool_call", tool_call_node)
    g.add_node("llm_call", llm_call_node)
    g.add_node("entry_node", entry_node)
    g.add_node("data_node", data_node)

    g.add_edge(START, "entry_node")
    g.add_edge("entry_node", "llm_call")
    g.add_conditional_edges("llm_call", tool_continue, ["tool_call", "data_node"])
    # 从tool->llm
    g.add_edge("tool_call", "llm_call")
    g.add_edge("data_node", END)
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
