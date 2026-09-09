"""
问题拆分 —— 多问题检测（divide）+ LLM 拆分（decompose）。

流程：
    split_questions(query)
        ├─ divide(query) == False → 单问题，原样返回
        └─ divide(query) == True  → decompose(query)：LLM 拆成子问题列表

兼容说明：api/chat.py 里是 `from intent.problemdecomposition import divide`，
所以模块级保留 divide() 函数；ProblemdeComposition 类作为可选封装。
"""

import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(encoding="utf-8-sig")  # utf-8-sig:兼容带 BOM 的 .env

client = OpenAI(
    api_key=os.getenv("QIAN_WEN_QUERYSTION_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1",
)

_MODEL = "qwen3.5-flash"

# ── 规则判断（divide）用的多问题信号 ──
_CONJUNCTIONS_RE = re.compile(
    r"并且|而且|同时|还有|另外|顺便|以及|再问|再帮|另一个|然后|首先|其次|和|还要"
)
_ENUMERATION_RE = re.compile(r"第一|第二|第三|第四|几个问题|两个问题|2个问题")

# ── LLM 拆分提示词：只做问题拆分 ──
_DECOMPOSE_PROMPT = """\
你是"问题拆分器"。用户可能在一句话里同时问了多个问题，你的任务是把它们拆成一个个独立、完整、语义自洽的子问题。

规则：
1. 只做拆分：不要回答任何问题，不要补充新问题，不要遗漏原问题里的任何信息。
2. 每个子问题要独立完整，能单独交给下游处理；尽量保留用户原始问法和关键信息（商品名、时间、单据号等）。
3. 边界：以下情况是一个问题，不要拆——
   - 选择疑问句："你是想退货还是换货？"
   - 对比疑问句："退货和换货有什么区别？"
   - "还有别的退款方式吗？" 这类带"还有…吗"的单问句
4. 寒暄语（"你好"、"谢谢"、"在吗"）不算问题，直接丢弃，不要输出。
5. 子问题按原顺序输出，不要打乱顺序。
6. 去重：如果多个子问题内容完全相同，或只是对同一件事的重复强调（如"我要退款和退款"），只保留一个子问题。

输出要求：
- 只输出一个 JSON 数组，例如：["子问题1", "子问题2", "子问题3"]
- 不要输出任何解释、前后缀、markdown 代码块标记。
- 如果输入本身只有一个问题，输出包含一个元素的数组。"""


def divide(query: str) -> bool:
    """规则判断是否多问题。返回 True → 需要 LLM 拆分。"""
    # 两个及以上问号：强多问题信号（"什么是退货？什么是换货？"）
    if query.count("？") + query.count("?") >= 2:
        return True
    # 连接词：即使只有一个问号也是多问题
    if _CONJUNCTIONS_RE.search(query):
        return True
    # 字数太多：长文本（弱信号，配合上面规则）
    if len(query) > 50:
        return True
    # 显式枚举："第一…第二" / "两个问题"
    if _ENUMERATION_RE.search(query):
        return True
    return False


def decompose(query: str) -> list[str]:
    """当 divide() 为 True 时调用：LLM 拆分子问题；失败降级为规则拆分。"""
    text = ""
    try:
        print("问题拆分模型调用")
        response = client.responses.create(
            model=_MODEL,
            input=[
                {"role": "system", "content": _DECOMPOSE_PROMPT},
                {"role": "user", "content": query},
            ],
            extra_body={"enable_thinking": False},  # 关闭思考模式，拆分要快
        )
        # 取出文本并去掉 ```json 包裹（LLM 偶尔会带）
        text = response.output_text if response else ""
        # 替换字符 忽略大小写
        cleaned = re.sub(
            r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE
        )
        questions = json.loads(cleaned)
        # 返回列表就好
        if isinstance(questions, list):
            result = [str(q).strip() for q in questions if str(q).strip()]
            if result:
                return result
        raise ValueError(f"LLM 输出不是 JSON 数组: {cleaned[:200]}")

    except Exception as e:
        print(f"LLM 问题拆分失败，降级规则拆分: {e}")
        # 兜底：从返回文本里抓引号中的子问题
        try:
            raw = re.findall(r'["“]([^"”]+)["”]', text)
            result = [q.strip() for q in raw if q.strip()]
            if result:
                return result
        except Exception:
            pass
        # 拆分失败先统一当单问题走
    return False


def split_questions(query: str) -> list[str]:
    """对外入口：多问题 → LLM 拆分；单问题 → 原样返回。"""
    query = (query or "").strip()
    if divide(query):
        return decompose(query)
    return False
