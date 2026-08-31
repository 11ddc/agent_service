import asyncio
from uuid import uuid4

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent.graph import graph  # LangGraph 编排图（拆分→意图→路由→处理→汇总）
from agent.langchina import agent  # DeepSeek Agent（流式 + 图异常兜底）
from intent.examples import HANDOFF_RE
from query_rewrite import aappend_history
from redis_client import redis_client

router = APIRouter()


class ChatRequest(BaseModel):
    question: str = Field(..., alias="message")
    session_id: str = Field(default_factory=lambda: str(uuid4()))


async def servercustomer(query: str, session_id: str) -> str | None:
    """
    转人工检测：10 分钟窗口内累计 3 次转人工信号 → 返回转人工提示。

    - 不要求连续（窗口内累计即可），每次命中刷新窗口
    - 只返回提示文本，不中断、不影响主流程
    - Redis 异常由调用方兜底，这里正常抛出
    """
    query = query.strip()
    if not HANDOFF_RE.search(query):
        return None

    key = f"user:{session_id}:message_count"
    count = await redis_client.incr(key)
    await redis_client.expire(key, 600)  # 10分钟滑动窗口

    print("count:", count)
    if count >= 3:
        await redis_client.delete(key)  # 触发后重置，下一次重新累计
        return "正在转人工，请稍候.................."
    return None


def _answer_by_agent(question: str, session_id: str) -> str:
    """让 Agent 单独回答一个问题（图异常兜底 / 多问题全空时降级）。"""
    agent_result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"configurable": {"thread_id": session_id}},
    )
    return agent_result["messages"][-1].content


@router.post("/chat")
async def chat(request: ChatRequest):
    """同步聊天 — 编排图执行：拆分→意图路由→(短路/RAG/Agent)→汇总"""
    # 转人工检测：先于一切执行，但只附加提示、不短路主流程；
    # Redis 异常时静默降级，聊天照常（不影响当前聊天接口）
    handoff_msg = None
    try:
        handoff_msg = await servercustomer(request.question, request.session_id)
    except Exception as e:
        print(f"转人工检测失败，忽略: {e}")

    # 整条「问题拆分→意图识别→按意图路由→处理→汇总」由 LangGraph 编排完成
    try:
        # 这里是子线程执行，跑到这里不等了去执行其他用户的请求
        # 其实就是因为事件循环就一个线程（多个协程轮流切换执行） 为了不阻塞当前事件循环  把他丢到线程池中让别的线程去执行这个协程
        # 事件循环里同时有很多协程（一个请求一个协程），不是只有一个；
        # 但循环只有一个线程在跑它们，所以同一时刻只能执行一个协程；
        # 协程多但线程一，所以要靠"让位"轮流跑——这正是 async 的本质。
        # "让事件循环去执行这个协程" ✅ —— 准确说：把 await 后面的可等待对象（协程/Task/Future）交给事件循环调度；
        # "当前协程挂起等待" ✅ —— 当前代码停在这一行，不往下走；
        # "等待时去执行其他东西" ✅ —— 事件循环单线程不闲着，去跑别的协程（其他请求）。
        # “创建了一个 Task（任务）对象”，然后将同步函数 graph.invoke 丢给线程池去执行。
        ####
        # 因为dense_task，sparse_task 这两个在上面是创建任务然后等下面await gather两个一起运行完
        # 并发执行器gather
        # gather  如果传进去的是协程对象 则会调用create_task包装成协程任务  如果传进去的是任务 就相当于await 则直接返回该任务
        # gather  和 create_task会创建任务（并将协程对象放入到事件循环中等待await执行）
        # 如果不考虑并发以及一些情况，gather 和await差不多
        # 协程对象就是你调用一个 async def 函数时，返回的那个东西
        # create_task 只能接收协程对象（接收任务会报错，gather是两个都可以）创建task 放入事件循环 等待执行
        result = await asyncio.to_thread(
            graph.invoke,
            {"question": request.question, "session_id": request.session_id},
        )
        answer = result.get("answer") or ""
        meta = result.get("meta") or {}
    except Exception as e:
        print(f"编排图执行失败，降级直连 Agent: {e}")
        answer = _answer_by_agent(request.question, request.session_id)
        meta = {"intent": "unknown", "method": "fallback"}

    # 与旧逻辑对齐：多问题全部检索为空时，用原问题整体兜底给 Agent
    if not answer:
        answer = _answer_by_agent(request.question, request.session_id)
        meta = {"intent": "unknown", "method": "fallback"}

    if handoff_msg:
        answer = f"{answer}\n\n{handoff_msg}" if answer else handoff_msg
    # 异步写入历史会话消息
    # 这里使用的是 主线程的事件循环 即main里面的run
    await aappend_history(request.session_id, request.question, answer)

    resp = {"answer": answer, "session_id": request.session_id, **meta}
    if handoff_msg:
        resp["handoff"] = True
    return resp


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式聊天 — 调用 DeepSeek Agent 并 SSE 流式返回（暂未按意图分流）"""

    async def generate():
        try:
            async for event in agent.astream(
                {"messages": [{"role": "user", "content": request.question}]},
                config={"configurable": {"thread_id": request.session_id}},
                stream_mode="messages",
            ):
                # stream_mode="messages" 返回 (message_chunk, metadata) 元组
                if isinstance(event, tuple) and len(event) >= 1:
                    chunk = event[0]
                    if hasattr(chunk, "content") and chunk.content:
                        yield f"data: {chunk.content}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: 出错了: {e!s}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
