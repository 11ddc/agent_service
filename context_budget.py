"""
上下文预算 —— token 计数、装箱、超限判定。

定位：**纯函数模块**，无网络、无 LLM、无副作用；不改动任何既有代码，
只在「重排之后 → 生成之前」这一段被调用（rag/graph.py 的 rag_node 里那一跳）。

解决三个具体问题：

1. **算账而不是等报错**：重排后的资料总量不受控（TOP_N 是硬编码常量），
   进生成模型前必须先算 token，否则超限只能靠 API 抛异常才发现。
2. **字符数不能当预算**：实测（cl100k_base，本项目真实语料）
   - CJK 汉字/中文标点：**1.17 token/字**（干净中文散文样本约 0.80）
   - 表格/数字/日期等非 CJK：**0.46 token/字符**（最密集的表格行 0.59）
   - 英文散文：0.22 token/字符
   同样 1000 字符，纯中文与英文的实际 token 差 3.6 倍以上，用字符数当预算会严重失真。
   > 实测还推翻了一个想当然的优化：原本打算"粗估筛一遍、只在接近边界时才精算"，
   > 但粗估在这些表格语料上会**低估**（比值低到 0.50），筛出来的"离预算还远"
   > 结论不可靠，会导致装箱塞多。所以装箱路径**全程精确计数**——
   > tiktoken 编码 2 万字符只要几毫秒，相对一次 1-3 秒的 LLM 调用可忽略。
3. **先丢弃、后编号**：RAG_PROMPT 要求用 [doc{i}] 标注来源
   （rag/generatellm.py:16）。编号一断，模型的引用就指错片段，
   所以装箱必须在编号之前完成。
3. **先丢弃、后编号**：RAG_PROMPT 要求用 [doc{i}] 标注来源
   （rag/generatellm.py:16）。编号一断，模型的引用就指错片段，
   所以装箱必须在编号之前完成。

为什么丢块要显式声明：把「必须合并全部字段」的资料砍掉一半，
模型会给出**看似完整、实则缺项**的答案，用户无从察觉 —— 这比超限报错更危险。
因此 pack_docs 会附一段"资料不完整"的说明，给模型一个承认资料不足的机会。

用法（两步，不碰现有代码）：
    from context_budget import pack_docs, input_budget

    budget = input_budget(model_context=32768, max_output=2000)
    result = pack_docs(reranked_docs, budget)
    prompt = RAG_PROMPT.format(context=result.format(), question=query)
    # result.used_tokens / result.dropped / result.truncated 可直接打日志
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

# ── 默认参数 ─────────────────────────────────────────────
# qwen3-32b 原生上下文窗口（rag/generatellm.py 生成模型）
DEFAULT_MODEL_CONTEXT = 32768
# 与现有 generate_answer 里的 max_tokens=2000 保持一致
DEFAULT_MAX_OUTPUT = 2000
# 安全边际：tiktoken(cl100k) 与 qwen/deepseek 的 tokenizer 有 ±15% 偏差，
# 预算不能顶满，否则"算得下"和"真的装得下"是两回事
DEFAULT_SAFETY = 0.15

# 粗估系数 —— **只在 tiktoken 不可用时兜底**，且刻意取偏大值。
# 实测（cl100k_base，本项目 53 个真实 chunk 聚合）：
#   CJK 汉字/中文标点：1.166 token/字（干净中文散文样本约 0.80）
#   表格/数字/日期等非 CJK：0.461 token/字符（最密集的表格行 0.59）
# 取 1.2 / 0.6 覆盖以上实测上界：高估=少装=安全，低估才会真超限。
_CJK_TO_TOKEN = 1.2
_OTHER_TO_TOKEN = 0.6

# CJK 汉字 + 中文标点/全角符号
_CJK_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3000-\u303f\uff00-\uffef]"
)

_encoder = None
_encoder_unavailable = False


# ==================== token 计数 ====================
def _get_encoder():
    """惰性拿 tiktoken 编码器；不可用时返回 None（自动退化为字符粗估）。"""
    global _encoder, _encoder_unavailable
    if _encoder is None and not _encoder_unavailable:
        try:
            import tiktoken

            _encoder = tiktoken.get_encoding("cl100k_base")
        except Exception as e:  # 没装 tiktoken 也要能跑，只是估得粗
            _encoder_unavailable = True
            print(f"[预算] tiktoken 不可用，改用字符粗估: {e}")
    return _encoder


def estimate_tokens(text: str) -> int:
    """字符粗估：按 CJK / 非 CJK 分档加权（快，且对中英混排都偏保守）。

    为什么不用固定系数：同一个系数在纯中文上会低估、在英文上会高估 4 倍，
    分档之后两边都能贴合，误差主要只剩在边界处 —— 那部分由
    count_tokens_for_budget 的精算兜住。
    """
    text = text or ""
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    other = len(text) - cjk
    return math.ceil(cjk * _CJK_TO_TOKEN + other * _OTHER_TO_TOKEN)


def count_tokens(text: str) -> int:
    """token 计数（预算装箱用的就是它）；tiktoken 不可用时退化为保守粗估。

    注意：cl100k 不是 deepseek/qwen 的官方分词器，用于**预算装箱**足够
    （偏差已被 DEFAULT_SAFETY 覆盖），但不要拿它做计费对账
    —— 计费要用 API 返回的 usage。
    """
    text = text or ""
    if not text:
        return 0
    enc = _get_encoder()
    if enc is None:
        return estimate_tokens(text)
    return len(enc.encode(text))


def doc_text(doc: Any) -> str:
    """Document 或裸字符串都能取文本（与 _format_docs 的兼容方式一致）。"""
    return doc.page_content if hasattr(doc, "page_content") else str(doc)


def count_docs_tokens(docs) -> int:
    """一组文档的 token 总量（用于打日志/离线统计）。"""
    return sum(count_tokens(doc_text(d)) for d in (docs or []))


# ==================== 预算换算 ====================
def input_budget(
    model_context: int = DEFAULT_MODEL_CONTEXT,
    max_output: int = DEFAULT_MAX_OUTPUT,
    safety: float = DEFAULT_SAFETY,
) -> int:
    """留给「输入」的 token 预算 = 窗口 - 输出预留 - 安全边际。

    输出必须提前扣掉：max_tokens 和输入**共用同一个上下文窗口**，
    只算输入是最常见的越界原因。
    """
    if model_context <= 0:
        raise ValueError("model_context 必须为正数")
    if max_output < 0:
        raise ValueError("max_output 不能为负")
    if not 0 <= safety < 1:
        raise ValueError("safety 应在 [0, 1) 之间")
    usable = model_context - max_output
    if usable <= 0:
        return 1
    return max(1, int(usable * (1 - safety)))


# ==================== 句子级截断 ====================
# 中英文句末标点（含换行）——按句子切，绝不切在半句上
_SENT_BOUNDARY_RE = re.compile(r"(?<=[。！？；!?;\n])")


def truncate_to_budget(text: str, budget_tokens: int, *, suffix: str = "…") -> str:
    """把一段文本按**句子边界**截断到预算内。

    单块超预算时的兜底：按字符硬切会切在半句上，模型会脑补后半句；
    按句子切虽然丢信息，但语义是完整的。
    """
    text = text or ""
    if budget_tokens <= 0 or not text:
        return ""
    if count_tokens(text) <= budget_tokens:
        return text  # 装得下就原样返回，不加省略号

    # 预留后缀自身的 token，否则"截断后 + …"会又超出去 1 个 token
    effective = max(1, budget_tokens - count_tokens(suffix))

    kept: list[str] = []
    used = 0
    for seg in _SENT_BOUNDARY_RE.split(text):
        if not seg:
            continue
        cost = count_tokens(seg)
        if used + cost > effective:
            break
        kept.append(seg)
        used += cost

    out = "".join(kept).rstrip()
    if not out:
        # 单句就超预算（超长无标点串）：只能按字符粗切兜底
        approx = max(1, int(effective / _CJK_TO_TOKEN))
        out = text[:approx].rstrip()
    if not out:
        return ""
    return out + suffix


# ==================== 装箱 ====================
@dataclass
class PackResult:
    """装箱结果：装进去的资料 + 用了多少 token + 丢了多少。"""

    docs: list = field(default_factory=list)
    used_tokens: int = 0
    total: int = 0
    truncated: bool = False  # 是否对某一块做了句子级截断

    @property
    def kept(self) -> int:
        return len(self.docs)

    @property
    def dropped(self) -> int:
        """被完整丢弃的块数（被截断保留的那块不算丢弃）。"""
        return max(0, self.total - self.kept)

    @property
    def partial(self) -> bool:
        """资料是否不完整（丢了块或截断过）。"""
        return self.dropped > 0 or self.truncated

    def notice(self) -> str:
        """资料不完整时的显式声明 —— 让模型有机会说"资料不足"而不是硬编。"""
        if not self.partial:
            return ""
        reason = []
        if self.dropped:
            reason.append(f"已丢弃相关性最低的 {self.dropped} 段")
        if self.truncated:
            reason.append("其中 1 段因过长做了截断")
        return (
            f"\n【资料完整性说明】本次仅提供 {self.kept}/{self.total} 段资料"
            f"（按相关性从高到低截取，{'；'.join(reason)}）。"
            f"若现有资料不足以完整回答，请明确指出缺少哪些信息，不要凭猜测补全。\n"
        )

    def format(self, *, declare_partial: bool = True, with_source: bool = True) -> str:
        """渲染成 prompt 上下文：**重新从 1 连续编号**，附来源与完整性说明。

        与 rag/generatellm.py 的 _format_docs 的差别就在这里：
        那个是"按原顺序编号"，丢弃后就出现 [doc1][doc3][doc7] 这样的空洞，
        模型的来源标注会错位；这里是先丢后编号，永远连续。

        with_source=True 时每条带「（来源：面包屑 (p.N)）」：RAG_PROMPT 要求模型
        用 [doc{i}] 标注来源，但没有出处标签的话用户无从核验 —— 这是引用可信的前提。
        来源标签约 10~20 token/条，相对 15% 的安全边际可忽略。

        Args:
            declare_partial: 是否附上"资料不完整"的声明（默认附，见模块头说明）
            with_source: 是否在每条资料前标注来源（默认标）
        """
        if not self.docs:
            return ""
        parts = []
        for i, d in enumerate(self.docs, 1):
            label = _source_label(d) if with_source else ""
            head = f"[doc{i}]" + (f"（来源：{label}）" if label else "")
            parts.append(f"{head} {doc_text(d)}")
        text = "\n\n".join(parts)
        return text + self.notice() if declare_partial else text


def pack_docs(
    docs,
    budget_tokens: int,
    *,
    allow_truncate_tail: bool = True,
) -> PackResult:
    """按给定顺序装箱，装不下的从尾部丢弃。

    前提：传入的 docs **已按相关性排序**（reordering() 的输出）。
    所以「丢尾」丢的就是最不相关的，不需要再额外判断相关性。

    截断规则（只在一种情况下截断，避免行为不可预测）：
    - 第一块就装不下 → 把它按句子截断塞进去（总比什么都不给模型好）；
    - 后续某块装不下 → 直接丢弃并停止（丢尾比截断更划算，也更可预测）。

    Args:
        docs: 已排序的检索结果（Document 或字符串）
        budget_tokens: 留给资料的 token 预算
        allow_truncate_tail: 是否允许"第一块就装不下"时的句子级截断

    Returns:
        PackResult（.format() 直接产出可用的上下文文本）
    """
    docs = list(docs or [])
    if not docs or budget_tokens <= 0:
        return PackResult(docs=[], used_tokens=0, total=len(docs))

    kept: list = []
    used = 0
    truncated = False

    for doc in docs:
        text = doc_text(doc)
        cost = count_tokens(text)
        if used + cost <= budget_tokens:
            kept.append(doc)
            used += cost
            continue

        # 装不下了：只有"第一块就超"才截断，其余情况丢尾
        if not kept and allow_truncate_tail:
            trimmed = truncate_to_budget(text, budget_tokens)
            if trimmed:
                kept.append(trimmed)
                used += count_tokens(trimmed)
                truncated = True
        break

    return PackResult(
        docs=kept, used_tokens=used, total=len(docs), truncated=truncated
    )


def _source_label(doc: Any) -> str:
    """从 metadata 里取**可核验**的出处：面包屑（含文件名+章节）＞裸路径，有页码带上。

    没有这一步，prompt 里只有 `[doc1] 正文` —— 模型按要求标了 [doc1]，用户却无法
    把 [doc1] 对应回任何文件，引用等于不可核验。
    """
    meta = getattr(doc, "metadata", None) or {}
    label = str(meta.get("breadcrumb") or meta.get("source") or "").strip()
    if not label:
        return ""
    page = meta.get("page_start")
    if page:
        label = f"{label} (p.{page})"
    return label


def format_docs_within_budget(docs, budget_tokens: int) -> tuple[str, PackResult]:
    """便捷入口：一步拿到 (上下文文本, 装箱详情)。

    想把现有 _format_docs 换成预算版时，改这一行就够：
        context_text = _format_docs(reranked_docs)
        context_text, pack = format_docs_within_budget(reranked_docs, budget)
    """
    result = pack_docs(docs, budget_tokens)
    return result.format(), result


# ==================== 超限判定 ====================
# 各家 SDK 的错误文案不统一，这里做关键字兜底匹配
_OVERFLOW_MARKERS = (
    "context_length_exceeded",
    "maximum context length",
    "context window",
    "too many tokens",
    "reduce the length",
    "exceeds the maximum",
)


def is_context_overflow(exc: BaseException) -> bool:
    """判断异常是不是"超出上下文长度"（用于决定是否降级重试）。

    刻意保守：只认明确的超限文案。把普通 400（比如参数非法）也当成超限
    会导致"降级重试 → 仍然失败 → 暴露成资料不足"的误判，掩盖真实故障。
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    text = str(exc).lower()

    if any(m in text for m in _OVERFLOW_MARKERS):
        return True
    # 兜底：400 且同时提到 token 和长度/上限
    if status == 400 and "token" in text and any(
        k in text for k in ("length", "limit", "exceed")
    ):
        return True
    return False
