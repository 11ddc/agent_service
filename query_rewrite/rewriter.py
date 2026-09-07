"""
查询改写核心 —— 结合会话历史做上下文补全（短期记忆的消费方）。

流程（门控设计，尽量少调 LLM，控制延迟与成本）：
    rewrite_query(question, history)
        ├─ question 为空 / 无历史       → 原样返回（零成本，无 LLM 调用）
        ├─ 无指代/省略信号（正则粗筛）  → 原样返回（不调 LLM）
        └─ 有历史且有指代信号            → 低温 LLM 补全；任何异常 → 原问题
"""

import os
import re

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_MODEL = "deepseek-chat"
_TEMPERATURE = 0.2  # 改写要稳定，低温；与 agent 的 0.7 区分开

# 指代/省略信号粗筛：命中才值得调 LLM（宁可漏改，不可多改）
_DEIXIS_RE = re.compile(
    r"它|这个|那个|这些|那些|刚才|上面|下面|那笔|那单|该单|这一单|上次|之前"
)

_REWRITE_PROMPT = """\
你是"查询改写助手"。用户正在持续对话，他最新的一句话可能依赖前面的对话内容（用了"它、这个、那个、刚才、上面"等指代，或省略了主语/对象）。

你的任务：把用户最新问题改写成一个**不依赖上下文、可以独立理解**的完整问题。

规则：
1. 只做"补全"：把指代、省略替换成对话历史里具体指代的对象（商品名、订单号、单据、话题等）。
2. 用户问题本身已经完整、不依赖上下文时，**原样输出，一个字都不要改**。
3. 禁止添加历史里不存在的信息；禁止猜测用户没说过的事实；禁止回答问题；禁止输出解释、引号或任何前后缀。
4. 输出只允许有一行：改写后的问题本身。

对话历史（按时间先后）：
{history}
"""

# 惰性创建：DEEPSEEK_API_KEY 缺失时不抛导入错误，改写自然降级为原样返回
_rewrite_client = None


def _get_client() -> OpenAI | None:
    global _rewrite_client
    if _rewrite_client is None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            return None
        _rewrite_client = OpenAI(
            api_key=api_key, base_url="https://api.deepseek.com/v1"
        )
    return _rewrite_client


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def _format_history(history: list[dict]) -> str:
    """历史渲染成对话文本；助手回答截断，防止 prompt 过长。"""
    lines = []
    for m in history:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            lines.append(f"用户：{content}")
        else:
            lines.append(f"助手：{_cut(content, 200)}")
    return "\n".join(lines) if lines else "（无）"


def rewrite_query(question: str, history: list[dict]) -> str:
    """查询改写入口：门控后决定是否调用 LLM；任何异常 → 返回原问题。

    Args:
        question: 用户最新问题
        history: 最近对话历史，如 [{"role": "user", "content": "..."}, ...]

    Returns:
        改写后的问题；无需改写或失败时原样返回。
    """
    question = (question or "").strip()
    if not question or not history:
        return question  # 空问题 / 无历史：无从补全，零成本返回
    if not _DEIXIS_RE.search(question):
        return question  # 无指代/省略信号：不调 LLM
    client = _get_client()
    if client is None:
        return question  # 未配置 API Key：降级原样返回
    try:
        response = client.chat.completions.create(
            model=_MODEL,
            # 温度低 需要稳定性
            temperature=_TEMPERATURE,
            messages=[
                {
                    "role": "system",
                    "content": _REWRITE_PROMPT.format(
                        history=_format_history(history),
                    ),
                },
                {"role": "user", "content": question},
            ],
        )
        rewritten = (response.choices[0].message.content or "").strip()

        print(f"查询改写：\n原问题: {question}\n改写后: {rewritten}")

        # 防止乱改 回退原问题
        if not rewritten or len(rewritten) > len(question) * 3 + 50:
            return question

        return rewritten
    except Exception as e:
        print(f"查询改写失败，使用原问题: {e}")
        return question
