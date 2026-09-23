import os
import traceback
from collections.abc import Callable

from dotenv import load_dotenv
from openai import OpenAI

# ── 新增：预算装箱工具（纯函数模块，见项目根目录 context_budget.py）──
# 不改动本文件既有逻辑，只在下面新增的 *_within_budget / _map_reduce 方法里使用
from context_budget import (
    DEFAULT_MAX_OUTPUT,
    DEFAULT_MODEL_CONTEXT,
    DEFAULT_SAFETY,
    count_tokens,
    doc_text,
    input_budget,
    is_context_overflow,
    pack_docs,
)

load_dotenv(encoding="utf-8-sig")  # utf-8-sig:兼容带 BOM 的 .env

# ⚠️ 这是**通用**模板，所有 RAG 问题都会用它。
# 不要再往里塞针对某个具体问题/某份文档的约束（例如"必须合并某表的字段清单"）：
# 那会污染每一次回答；更糟的是 pack_docs 丢弃后会**重新编号**（见 context_budget.py），
# 写死"doc2、doc5"这类编号时那些块可能根本不存在，等于诱导模型编造。
# 场景化约束请另建模板，按需追加。
RAG_PROMPT = """你是一个严谨的知识助手。请严格基于以下【参考资料】回答用户问题，不要编造资料中不存在的信息。

【参考资料】
{context}

【回答要求】
1. 只用参考资料里的内容作答，不得引入外部知识
2. 在每个关键论断后用 [doc{{i}}] 标注来源（i 对应上面文档编号）
3. 资料与问题**部分相关**时，先把能对上的部分答出来（照常标注来源），再说明还缺什么。
   不要因为"资料不完整"就整段拒答 —— 用户问的是排查思路，能给出相关现象、原因、
   处理建议都是有价值的；只答一句"无法回答"等于把已有的信息也一起丢掉了
4. 资料里没有直接出现用户问的那个词/型号时，不要立刻判定无关：先看它讲的是不是
   同一类问题（同类现象、同类原因、通用排查或售后流程）。是就把对得上的部分答出来，
   并说明这是资料里的通用说明、未专门针对该型号
5. 只有资料与问题**完全无关**时，才说"根据现有资料无法回答"，不要猜测
6. 答案结构清晰，分点叙述

【用户问题】
{question}
"""

# ── 新增：预算内生成用的模型参数与分段生成提示词 ──
# （RAG_PROMPT / _format_docs / generate_answer 一个字都没动）
GENERATE_MODEL = "qwen3-32b"  # 与上面 generate_answer 里写死的模型保持一致

# map 阶段：每批资料各提取一次局部结果（输出短、可合并）
MAP_PROMPT = """你是严谨的知识助手。下面是【参考资料】的其中一部分，不是全部。
请只针对【用户问题】，从这一部分资料里提取相关信息。

【参考资料】
{context}

【用户问题】
{question}

要求：
1. 只输出与问题有关的结论、字段、表格行，不要解释过程，不要写引导语。
2. 不得编造资料里没有的内容。
3. 字段/清单类问题，要完整列出这部分资料里出现的**所有**字段或条目，不要省略。
4. 这部分资料确实与问题无关时，只输出"本部分资料未涉及"。"""

# reduce 阶段：把多份局部结果合并成一份完整答案
REDUCE_PROMPT = """你是严谨的知识助手。下面是对同一个问题的多份局部提取结果（分别来自不同的资料片段，内容会有重复，也可能各有遗漏）。

【局部结果】
{context}

【用户问题】
{question}

要求：
1. **合并去重**：同一信息只保留一次。
2. **不得遗漏**：任何一份局部结果里出现过的字段、条目、数值都必须出现在最终答案里。
3. 用 [part{{i}}] 标注每条信息来自哪份局部结果。
4. 局部结果之间矛盾时全部列出，并注明存在冲突。
5. 合并后仍不足以完整回答时，直接说明还缺少什么。"""


def _format_docs(reranked_docs: list) -> str:
    """把重排后的文档拼成带 [doc{i}] 编号的上下文文本，供提示词里的引用要求对应上。

    兼容 Document 对象（取 page_content）和普通字符串。
    """
    parts = []
    for i, doc in enumerate(reranked_docs, 1):
        content = doc.page_content if hasattr(doc, "page_content") else str(doc)
        parts.append(f"[doc{i}] {content}")
    return "\n\n".join(parts)


class RAGGenerator:
    def __init__(self):
        # 创建生成模型客户端
        self.generate_client = OpenAI(
            api_key=os.getenv("GENERATE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def generate_answer(self, query: str, reranked_docs: list) -> str:
        """
        基于重排序后的文档，生成最终回答
        :param query: 用户原始问题
        :param reranked_docs: 重排序后的文档列表
        """
        print("问题：query：", query)
        # 先拼成带编号的纯文本，直接把 Document 列表塞进 format 会得到一堆
        # Document(page_content=..., metadata=...) 的对象字符串，污染上下文
        context_text = _format_docs(reranked_docs)
        try:
            res = self.generate_client.chat.completions.create(
                model="qwen3-32b",
                messages=[
                    {"role": "system", "content": "你是严谨的中文知识助手。"},
                    {
                        "role": "user",
                        "content": RAG_PROMPT.format(
                            context=context_text, question=query
                        ),
                    },
                ],
                extra_body={"enable_thinking": False},
                stream=False,
                # 温度越高llm越自由发挥
                temperature=0.3,
                # 限制生成模型的输出长度，避免生成过长的回答
                max_tokens=2000,
            )
        except Exception as e:
            # 打印完整异常类型和堆栈（之前 KeyError 只打印出 'i' 就是吃了这个亏），
            # 然后抛给上层：graph.py 的 rag_node 会接住并走降级文案
            print(f"生成发生问题：{e!r}")
            traceback.print_exc()
            raise

        print("经过生成之后的文档：", res)

        return res.choices[0].message.content

    # ══════════════════════════════════════════════════════════════
    # 新增：预算内生成（以下全是纯新增方法，上面的 generate_answer
    # 行为与签名完全不变；调用方想用新版就改调 generate_answer_within_budget）
    # ══════════════════════════════════════════════════════════════

    def _chat(
        self,
        messages: list,
        *,
        max_tokens: int = DEFAULT_MAX_OUTPUT,
        emit: Callable[[dict], None] | None = None,
    ) -> str:
        """统一的底层调用（新增方法共用，避免重复那段 create 参数）。

        emit=None（默认）→ 走**原有非流式实现**，/chat 链路用的就是这条，
                          线上行为与改造前完全一致；
        emit=callback     → 流式路径：边生成边逐块解析并推送 delta 事件，
                          最后仍返回**完整字符串**（节点契约要求 return 完整
                          answer，所以"边推边攒"必须并存）。
        """
        if emit is None:
            # ── 原有实现，未做任何改动 ──────────────────────────────
            res = self.generate_client.chat.completions.create(
                model=GENERATE_MODEL,
                messages=messages,
                extra_body={"enable_thinking": False},
                stream=False,
                temperature=0.3,
                max_tokens=max_tokens,
            )
            return (res.choices[0].message.content or "").strip()

        # ── 新增：流式路径 ─────────────────────────────────────────
        # 1. 生成模型返回 SSE 流式帧（每帧可能只带 delta，也可能只带 usage，甚至空帧）
        # 2. 遍历每帧，解析出 delta 内容 → emit({"type":"delta","content":piece}) 推给前端
        stream = self.generate_client.chat.completions.create(
            model=GENERATE_MODEL,
            messages=messages,
            extra_body={"enable_thinking": False},
            stream=True,
            temperature=0.3,
            max_tokens=max_tokens,
        )
        parts: list[str] = []
        for chunk in stream:
            # OpenAI 兼容层在末尾可能补一个只带 usage、choices 为空的帧，必须跳过
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            piece = getattr(delta, "content", None) if delta is not None else None
            if piece:
                parts.append(piece)
                emit({"type": "delta", "content": piece})
        return "".join(parts).strip()

    def _generate_once(
        self,
        query: str,
        context_text: str,
        emit: Callable[[dict], None] | None = None,
    ) -> str:
        """单次生成（资料已经装箱好）；与 generate_answer 用的是同一个 RAG_PROMPT。"""
        return self._chat(
            [
                {"role": "system", "content": "你是严谨的中文知识助手。"},
                {
                    "role": "user",
                    "content": RAG_PROMPT.format(context=context_text, question=query),
                },
            ],
            emit=emit,
        )

    def generate_answer_within_budget(
        self,
        query: str,
        reranked_docs: list,
        *,
        budget_tokens: int | None = None,
        model_context: int = DEFAULT_MODEL_CONTEXT,
        max_output: int = DEFAULT_MAX_OUTPUT,
        safety: float = DEFAULT_SAFETY,
        declare_partial: bool = True,
        emit: Callable[[dict], None] | None = None,
    ) -> dict:
        """【预检版生成】先按 token 预算装箱，再调模型；真超限自动降级。

        流程（重排之后 → 生成之前这一跳应该做的事）：
            1. 算预算：input_budget(窗口, 输出预留, 安全边际)
            2. 装箱：pack_docs 按相关性丢尾 —— 保证输入**一定**装得下
            3. 生成：丢块时 prompt 里带上"资料不完整"的声明
            4. 兜底：万一仍被判超限（tokenizer 差异），预算降到 1/2、1/4 重试
            5. 最后：缩到 1/4 还不行 → map-reduce 分段生成（信息不丢）

        Returns:
            dict（而不是 str），方便调用方直接打日志：
            {answer, used_tokens, kept, dropped, truncated, budget_tokens,
             degraded, retries}
            degraded=True 表示走了分段生成（延迟更高）

        emit：可选的流式出口（见 agent/events.py）。传 None（默认）时本方法行为
            与改造前**完全一致** —— /chat 链路走的就是这条。传了回调才走流式，
            并且会在预算降级重试前先推一个 reset 事件，让前端丢弃已经收到的
            作废 token（流式下"先吐后发现超限"是回退不了的，只能通知前端丢弃）。
        """
        docs = list(reranked_docs or [])
        # 预算换算
        budget = budget_tokens or input_budget(model_context, max_output, safety)

        # 流式对账状态：已经推给客户端的正文是否还有效
        streamed = False

        def _emit(ev: dict) -> None:
            nonlocal streamed
            if ev.get("type") == "delta":
                streamed = True
            if emit is not None:
                emit(ev)

        # 传给生成层的出口 —— 必须是"调用方的 emit"，而不是恒为可调用的 _emit：
        #   - emit is None（同步 /chat 链路）→ 传 None，_chat 才走非流式分支，
        #     与 docstring 承诺的"与改造前完全一致"相符；
        #     曾经这里恒传 _emit，导致同步链路也在 stream=True 下跑（4 个测试因此变红）。
        #   - emit 有值（流式链路）→ 传 _emit：它负责转发事件，并顺带记账 streamed，
        #     供下面"超限降级前先推 reset"的判断使用。
        gen_emit = None if emit is None else _emit

        # 没有资料：保持与 generate_answer 一致的行为（照样让模型按 prompt 回答）
        if not docs:
            return {
                "answer": self._generate_once(query, "", emit=gen_emit),
                "used_tokens": 0,
                "kept": 0,
                "dropped": 0,
                "truncated": False,
                "budget_tokens": budget,
                "degraded": False,
                "retries": 0,
            }

        retries = 0
        last_pack = pack_docs(docs, budget)

        for attempt_budget in (budget, budget // 2, budget // 4):
            if attempt_budget <= 0:
                continue
            pack = pack_docs(docs, attempt_budget)
            if not pack.docs:
                continue
            last_pack = pack
            context_text = pack.format(declare_partial=declare_partial)
            try:
                answer = self._generate_once(query, context_text, emit=gen_emit)
            except Exception as e:
                if not is_context_overflow(e):
                    raise  # 不是超限 → 交给上层原有异常处理，不掩盖真实故障
                retries += 1
                print(
                    f"[预算] 输入被判超限，预算 {attempt_budget} → 降至 1/2 重试: {e}"
                )
                if streamed:
                    # 流式下"先吐出去才发现超限"是收不回来的，只能通知前端丢弃
                    _emit(
                        {
                            "type": "reset",
                            "reason": "输入超限，正在用更小的资料重新生成",
                        }
                    )
                    streamed = False
                continue
            if retries:
                print(
                    f"[预算] 降级重试成功（重试 {retries} 次，预算 {attempt_budget}）"
                )
            return {
                "answer": answer,
                "used_tokens": pack.used_tokens,
                "kept": pack.kept,
                "dropped": pack.dropped,
                "truncated": pack.truncated,
                "budget_tokens": attempt_budget,
                "degraded": False,
                "retries": retries,
            }

        # 缩到 1/4 仍超限 → 分段生成
        print("[预算] 缩到 1/4 预算仍超限，改用 map-reduce 分段生成")
        if streamed:
            # map-reduce 是 N+1 次调用，不做流式（见 _map_reduce_generate 注释），
            # 所以先让前端丢弃已推的作废内容，完整答案在最后一次性下发。
            _emit({"type": "reset", "reason": "资料较多，改用分段生成，请稍候"})
            streamed = False
        answer = self._map_reduce_generate(query, docs, budget)
        return {
            "answer": answer,
            "used_tokens": last_pack.used_tokens,
            "kept": last_pack.kept,
            "dropped": last_pack.dropped,
            "truncated": last_pack.truncated,
            "budget_tokens": budget,
            "degraded": True,
            "retries": retries,
        }

    def _map_reduce_generate(self, query: str, docs: list, budget_tokens: int) -> str:
        """资料总量真的超过窗口时的正解：分批提取 → 合并。

        用「模型的输出」换「模型的输入预算」：每批只喂窗口装得下的一部分，
        局部结果先被压缩成要点，合并阶段的输入自然就小了。

        代价：N+1 次 LLM 调用，延迟随之上升 —— 所以只在预检失败时才走这条路。
        """
        docs = list(docs or [])
        if not docs:
            return self._generate_once(query, "")

        # 每批只用 1/3 预算：还要留出 prompt 模板和局部答案的输出空间
        batch_budget = max(1, budget_tokens // 3)
        batches: list[list] = []
        buf: list = []
        used = 0
        for doc in docs:
            cost = count_tokens(doc_text(doc))
            if buf and used + cost > batch_budget:
                batches.append(buf)
                buf, used = [], 0
            buf.append(doc)
            used += cost
        if buf:
            batches.append(buf)

        print(f"[预算] map-reduce：{len(docs)} 段资料分成 {len(batches)} 批")

        partials: list[str] = []
        for i, batch in enumerate(batches, 1):
            context = pack_docs(batch, batch_budget).format()
            if not context:
                continue
            try:
                part = self._chat(
                    [
                        {"role": "system", "content": "你是严谨的中文知识助手。"},
                        {
                            "role": "user",
                            "content": MAP_PROMPT.format(
                                context=context, question=query
                            ),
                        },
                    ],
                    max_tokens=max(
                        256, DEFAULT_MAX_OUTPUT // 2
                    ),  # 局部答案要短，否则合并阶段又超
                )
            except Exception as e:
                print(f"[预算] 第 {i}/{len(batches)} 批局部生成失败，跳过该批: {e}")
                continue
            if part:
                partials.append(part)

        if not partials:
            raise RuntimeError("map-reduce 局部生成全部失败")
        if len(partials) == 1:
            return partials[0]

        # 合并阶段的输入也可能超（局部结果自身很长）→ 同样套一层装箱
        merge_pack = pack_docs(partials, budget_tokens)
        merge_context = merge_pack.format(declare_partial=False)
        if merge_pack.dropped:
            print(f"[预算] 合并阶段丢弃 {merge_pack.dropped} 份局部结果")
        return self._chat(
            [
                {"role": "system", "content": "你是严谨的中文知识助手。"},
                {
                    "role": "user",
                    "content": REDUCE_PROMPT.format(
                        context=merge_context, question=query
                    ),
                },
            ]
        )
