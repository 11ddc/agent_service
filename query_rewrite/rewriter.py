"""
查询改写核心 —— 结合会话历史做上下文补全（短期记忆的消费方）。

流程（门控设计，尽量少调 LLM，控制延迟与成本）：
    rewrite_query(question, history)
        ├─ question 为空 / 无可用历史        → 原样返回（零成本，无 LLM 调用）
        ├─ needs_rewrite() 判为"自洽"        → 原样返回（不调 LLM）
        └─ 判为"依赖上下文"                   → 低温 LLM 补全 → 校验通过才采用

门控取舍（2026-xx 调整）：早期版本只认指代词且"宁可漏改不可多改"，
结果"那退货呢 / 换成红色的呢 / 多久能到 / 还有别的吗 / 继续"这类最高频的
省略式追问全部漏补。现在把省略句式、追问词开头、以及"有历史 + 极短问题"
也纳入信号：多调一次 LLM 很便宜（prompt 已要求"本就完整就原样输出"），
而漏补全的代价是把"它保修多久"直接丢给检索/路由，错得很隐蔽。
"""

import logging
import os
import re
import threading

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(encoding="utf-8-sig")  # utf-8-sig:兼容带 BOM 的 .env

logger = logging.getLogger(__name__)

_MODEL = "deepseek-chat"
_TEMPERATURE = 0.2  # 改写要稳定，低温；与 agent 的 0.7 区分开
# 改写是同步阻塞节点（graph 跑在线程池里），必须限时：
# 不设 timeout 时 SDK 默认 600s + 2 次重试，一次抖动就能把用户请求挂几分钟。
_TIMEOUT = float(os.getenv("REWRITE_TIMEOUT", "8"))
_MAX_RETRIES = int(os.getenv("REWRITE_MAX_RETRIES", "1"))

# ── 需要上下文的信号：命中任一 → 值得付一次 LLM 调用 ──────────
_CONTEXT_SIGNAL_RES = (
    # 1) 指代/指示词（强信号）
    re.compile(
        r"它|他|她|这个|那个|这些|那些|这种|那种|那笔|那单|那款|那台"
        r"|上述|上面|刚才|之前|上次|该单|该订单|该商品|该产品"
    ),
    # 2) 省略式追问句式：结尾"呢"（"换成红色的呢"）；"那/那么"开头（"那退货呢"）
    re.compile(r"呢[?？!！。~\s]*$|^(?:那|那么)"),
    # 3) 追问词开头：多久能到 / 为什么 / 继续 / 再详细点 / 还有别的吗
    re.compile(
        r"^(?:继续|接着|再来|再(?:说|讲|详细|具体|补充)|详细|还有|换成|改成"
        r"|为什么|为何|多久|多长|多少|怎么(?:样|办)?$|可以吗|行吗)"
    ),
)
# 4) 有历史时的极短问题：省略主语/宾语是常态（"多久能到"、"继续"）。
#    设 0 可关闭这条规则（回到"只认信号词"的保守策略）。
_SHORT_QUERY_LEN = int(os.getenv("REWRITE_SHORT_QUERY_LEN", "8"))

# ── 结果校验 ──────────────────────────────────────────────
# 比对"实词保留"时先剥掉指代/虚词/标点：像"它呢"这种整句都由虚词组成，
# 剥完为空，就不该用实词规则去卡它（否则合法改写永远过不了）。
_STOPWORD_RE = re.compile(r"[它他她这那哪些么的呢吗吧啊呀哦嗯?？!！。，,、;；:：\s]+")
_ENTITY_RE = re.compile(r"[A-Za-z0-9]+")  # 单号/型号/数字必须原样保留
_LINE_PREFIX_RE = re.compile(r"^(?:改写后的问题|改写后|问题|回答|答案|输出)\s*[:：]\s*")
_MIN_CONTENT_KEEP = 0.6  # 原问题实词至少要有这么多出现在改写结果里

_REWRITE_PROMPT = """\
你是"查询改写助手"。用户正在持续对话，他最新的一句话可能依赖前面的对话内容（用了"它、这个、那个、刚才、上面"等指代，或省略了主语/对象）。

你的任务：把用户最新问题改写成一个**不依赖上下文、可以独立理解**的完整问题。

规则：
1. 只做"补全"：把指代、省略替换成对话历史里具体指代的对象（商品名、订单号、单据、话题等）。
2. 用户问题本身已经完整、不依赖上下文时，**原样输出，一个字都不要改**。
3. 禁止添加历史里不存在的信息；禁止猜测用户没说过的事实；禁止回答问题；禁止输出解释、引号或任何前后缀。
4. 输出只允许有一行：改写后的问题本身。
5. 不要改变问句类型（选择疑问、对比疑问保持原样），也不要把一个问题里的多个点拆成多句。

对话历史（按时间先后）：
{history}
"""

# 惰性创建：DEEPSEEK_API_KEY 缺失时不抛导入错误，改写自然降级为原样返回
_rewrite_client = None
_client_lock = threading.Lock()


def _get_client() -> OpenAI | None:
    """惰性 + 双检锁创建客户端（图节点在线程池里并发执行，避免重复建连）。"""
    global _rewrite_client
    if _rewrite_client is not None:
        return _rewrite_client
    with _client_lock:
        if _rewrite_client is None:
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                return None
            _rewrite_client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com/v1",
                timeout=_TIMEOUT,
                max_retries=_MAX_RETRIES,
            )
    return _rewrite_client


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def _format_history(
    history: list[dict], max_messages: int = 8, assistant_limit: int = 200
) -> str:
    """历史渲染成对话文本；助手回答截得更短，防止 prompt 过长。"""
    lines = []
    for m in history[-max_messages:]:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            lines.append(f"用户：{_cut(content, 300)}")
        else:
            lines.append(f"助手：{_cut(content, assistant_limit)}")
    return "\n".join(lines) if lines else "（无）"


def _content(text: str) -> str:
    """剥掉指代/虚词/标点，只留实词（用于判断改写有没有丢信息）。"""
    return _STOPWORD_RE.sub("", text or "")


def needs_rewrite(question: str, history: list[dict] | None) -> bool:
    """门控：这句话是否依赖上文（依赖才值得调用 LLM 补全）。"""
    question = (question or "").strip()
    if not question or not history:
        return False
    if _SHORT_QUERY_LEN and len(question) <= _SHORT_QUERY_LEN:
        return True
    return any(p.search(question) for p in _CONTEXT_SIGNAL_RES)


def _sanitize(rewritten: str) -> str:
    """只取第一行，去掉"改写后："这类前缀和包裹引号（模型偶尔不守"只输出一行"）。"""
    for line in (rewritten or "").splitlines():
        line = _LINE_PREFIX_RE.sub("", line.strip()).strip()
        line = line.strip("\"'“”‘’《》「」").strip()
        if line:
            return line
    return ""


def _accepts(question: str, candidate: str) -> bool:
    """改写结果是否可信：长度没爆，且没丢掉原问题的单号/型号与实词。"""
    if not candidate:
        return False
    # 防止乱改（把一个问题写成一段话）
    if len(candidate) > len(question) * 3 + 50:
        return False
    # 单号/型号/英文数字必须原样保留（最容易在补全时被"顺手改掉"的东西）
    for token in _ENTITY_RE.findall(question):
        if token.lower() not in candidate.lower():
            return False
    # 实词保留：原问题的实词至少 60% 出现在改写里（"它/这个/呢"这些不算实词）
    src = set(_content(question))
    if src and len(src & set(_content(candidate))) / len(src) < _MIN_CONTENT_KEEP:
        return False
    return True


def rewrite_query(question: str, history: list[dict]) -> str:
    """查询改写入口：门控后决定是否调用 LLM；任何异常/校验不过 → 返回原问题。

    Args:
        question: 用户最新问题
        history: 最近对话历史，如 [{"role": "user", "content": "..."}, ...]

    Returns:
        改写后的问题；无需改写、失败或校验不过时原样返回。
    """
    question = (question or "").strip()
    if not needs_rewrite(question, history):
        return question  # 空问题 / 无历史 / 本身自洽：不调 LLM

    client = _get_client()
    if client is None:
        logger.warning("未配置 DEEPSEEK_API_KEY，查询改写跳过（原样返回）")
        return question

    try:
        response = client.chat.completions.create(
            model=_MODEL,
            temperature=_TEMPERATURE,  # 温度低 需要稳定性
            timeout=_TIMEOUT,  # 再显式给一次：即使上层换掉 client 也不会无超时
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
        raw = response.choices[0].message.content or ""
    except Exception as e:
        logger.warning("查询改写失败，使用原问题: %s", e)
        return question

    candidate = _sanitize(raw)
    if not _accepts(question, candidate):
        # 被拒的样本要留痕：这是调 prompt / 调门控的唯一数据来源
        logger.warning(
            "查询改写结果被拒（丢信息或超长），使用原问题: 原=%r 改写=%r",
            question,
            raw[:200],
        )
        return question

    if candidate != question:
        logger.info("查询改写生效: %r -> %r", question, candidate)
    return candidate
