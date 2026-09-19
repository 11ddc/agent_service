import asyncio
import json
from uuid import uuid4

from fastapi import APIRouter
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from agent.events import (  # 流式事件旁路（同步链路下为空操作）
    reset_emitter,
    set_emitter,
)
from agent.graph import graph  # LangGraph 编排图（拆分→意图→路由→处理→汇总）
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
    # agent_result = agent.invoke(
    #     {"messages": [{"role": "user", "content": question}]},
    #     config={"configurable": {"thread_id": session_id}},
    # )
    # return agent_result["messages"][-1].content
    return "兜底agent回答: "


@router.post("/chat")
async def chat(request: ChatRequest):
    print("✅ 请求已进入接口！")
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

        print(f"编排图执行完成，answer: {answer}, meta: {meta}")
    except Exception as e:
        print(f"编排图执行失败，降级直连 Agent: {e}")
        answer = _answer_by_agent(request.question, request.session_id)
        meta = {"intent": "unknown", "method": "fallback"}

    # 与旧逻辑对齐：多问题全部检索为空时，用原问题整体兜底给 Agent
    if not answer:
        print("编排图执行结果为空，降级直连 Agent")
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


def _sse(event: dict) -> dict:
    """把内部事件包成 SSE 帧。

    payload 走 JSON（而不是裸文本）有两个理由：
    1. 正文里必然出现换行（markdown 列表/表格），裸文本会让 SSE 逐行解析时把
       第二行起当未知字段静默丢掉；JSON 会把 \\n 转义掉。
    2. status / meta / error / reset 这些结构化事件需要携带多个字段。
    """
    return {"event": event["type"], "data": json.dumps(event, ensure_ascii=False)}


def _run_graph_with_emitter(emitter, payload: dict):
    """在工作线程里跑图，并把 emitter 装进**该线程**的上下文。

    装在 worker 函数内部是刻意的：langgraph 提交节点时会在当前线程
    copy_context()（langgraph/pregel/_executor.py:64），所以这里设置的值能被
    节点内部的 emit() 读到。若改在接口协程里设置，就额外依赖 asyncio.to_thread
    的上下文拷贝行为 —— 那样也能work，但不如此处直接。
    """
    token = set_emitter(emitter)
    try:
        return graph.invoke(payload)
    finally:
        reset_emitter(token)


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式聊天 — 与 /chat 走**同一条编排图**，RAG 路径真 token 流式。

    与 /chat 的两处**有意差异**（为严格隔离，不改动同步链路的任何行为）：
    - 不写 Redis 历史：否则会改变 /chat 下一轮 rewrite_node 读到的历史；
    - 不做转人工计数：否则会共享 user:{session_id}:message_count 这个副作用。
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def emitter(event: dict) -> None:
        # 节点跑在工作线程，而 asyncio.Queue 属于事件循环 → 必须线程安全地回推
        loop.call_soon_threadsafe(queue.put_nowait, event)

    # 客户端"当前应该持有"的正文；节点推 reset 时会清空，用于流末对账
    streamed: list[str] = []

    def tracking_emitter(event: dict) -> None:
        kind = event.get("type")
        if kind == "reset":
            streamed.clear()  # 节点宣告前面推的 token 作废（预算降级重试等）
        elif kind == "delta":
            streamed.append(event.get("content") or "")
        emitter(event)

    async def run() -> None:
        try:
            result = await asyncio.to_thread(
                _run_graph_with_emitter,
                tracking_emitter,
                {"question": request.question, "session_id": request.session_id},
            )
            final_answer = result.get("answer") or ""

            # ── 流末对账 ─────────────────────────────────────────────
            # 只有 RAG 路径会吐 delta。若 intent 路由到 agent_flow / tool_agent /
            # short_circuit，或 RAG 降级成 map-reduce，流式正文就是不完整的，
            # 一律以图返回的完整答案为准。
            # 比较时用 split() 归一化空白：流式侧不做 strip，非流式侧 strip 过。
            if not streamed:
                emitter({"type": "delta", "content": final_answer})
            elif "".join(streamed).split() != final_answer.split():
                emitter({"type": "reset", "reason": "部分内容未走流式，改用完整答案"})
                emitter({"type": "delta", "content": final_answer})

            emitter(
                {
                    "type": "meta",
                    "session_id": request.session_id,
                    **(result.get("meta") or {}),
                }
            )
            emitter({"type": "done"})
        except Exception as e:
            print(f"流式编排图执行失败: {e}")
            emitter({"type": "error", "message": f"{e!s}"})
        finally:
            # 哨兵：无论成功还是失败都必须发，否则下面的生成器会永久 await
            emitter(None)

    task = asyncio.create_task(run())

    async def generate():
        try:
            while True:
                # 取出 队列中的事件，如果是None就跳出循环
                event = await queue.get()
                if event is None:
                    break
                # SSE 帧化后 yield 给 交给长连接EventSourceResponse，前端就能收到
                yield _sse(event)
        finally:
            # 客户端断线时 EventSourceResponse 会取消本生成器；图在工作线程里
            # 没法真正中断，让它跑完即可（因为不写历史，所以没有副作用残留）。
            if not task.done():
                task.cancel()

    return EventSourceResponse(
        generate(),
        ping=15,  # 反代理保活：nginx proxy_read_timeout 默认 60s，长检索会被掐断
        headers={"X-Session-Id": request.session_id},
    )
