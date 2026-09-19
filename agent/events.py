"""流式事件旁路（emit）—— 让节点能把 token 送出图，同时**不干扰同步链路**。

核心设计：**空操作默认值**
    emitter 存在 contextvar 里，默认 None。
    - `/chat`（同步链路）从不设置 emitter → `emit()` 直接 return，
      节点的行为、调用次数、时序与改造前完全一致；
    - `/chat/stream` 在 worker 线程里设置 emitter → 节点内部推的事件
      经 `loop.call_soon_threadsafe` 回到事件循环，最终变成 SSE 帧。

为什么 contextvar 能穿过线程池抵达节点（已核对源码，Python 3.12 / langgraph 1.2.11）：
    1. `asyncio.to_thread` 复制当前 context：`ctx = copy_context()` + `ctx.run(...)`
       （标准库 asyncio/threads.py，文档字符串明确写了 context 会被传播）
    2. langgraph 提交节点时**在提交线程里**再复制一次：
       `ctx = copy_context(); self.executor.submit(ctx.run, fn, ...)`
       （langgraph/pregel/_executor.py:64,71）
    两次都是显式 `ctx.run`，所以不依赖 ThreadPoolExecutor 自身是否继承 context
    （它默认**不继承**）—— 这也是为什么 emitter 必须设在 worker 线程内、而不是
    设在接口协程里。

约定：事件出口故障绝不允许带崩主流程，`emit()` 内部吞掉异常。
"""

from collections.abc import Callable
from contextvars import ContextVar, Token

Emitter = Callable[[dict], None]

_emitter: ContextVar[Emitter | None] = ContextVar("chat_stream_emitter", default=None)


def set_emitter(fn: Emitter) -> Token:
    """在当前上下文设置事件出口；返回 token 供 reset_emitter 还原。"""
    return _emitter.set(fn)


def reset_emitter(token: Token) -> None:
    """还原到设置前的状态（配套 set_emitter，务必放在 finally 里）。"""
    _emitter.reset(token)


def current_emitter() -> Emitter | None:
    """取出当前上下文的事件出口。

    同步链路下是 None —— 可以直接当"当前是不是流式调用"的判据用
    （rag_node 就是这么判断要不要把 emit 透传给生成器的）。
    """
    return _emitter.get()


def emit(event: dict) -> None:
    """节点内推送一个流式事件；没有 emitter 时（同步 /chat 链路）是空操作。"""
    fn = _emitter.get()
    if fn is None:
        return
    try:
        fn(event)
    except Exception as e:
        # 出口故障（例如事件循环已关闭）只影响流式观感，不能影响答案生成
        print(f"流式事件推送失败，忽略: {e}")
