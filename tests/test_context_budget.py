"""上下文预算测试 —— 零服务、零网络：全是纯函数 + 假客户端。

覆盖四件事：
1. token 计数与预算换算（含"输出必须提前扣掉"这条最容易搞错的规则）；
2. 装箱：不超预算、丢尾、**先丢弃后编号**（编号连续，引用不错位）；
3. 超限判定的保守性（普通 400 不能被误判成超限）；
4. 预检版生成的三条路径：一次成功 / 降级重试成功 / map-reduce 兜底。

另附一个"切分方式"的取证测试：证实 RecursiveCharacterTextSplitter
的默认 separators 末位是 ""（逐字符硬切），中文长段落会被切在词中间。
"""
import re

import pytest
from langchain_core.documents import Document

from context_budget import (
    DEFAULT_MAX_OUTPUT,
    DEFAULT_MODEL_CONTEXT,
    PackResult,
    count_tokens,
    doc_text,
    estimate_tokens,
    format_docs_within_budget,
    input_budget,
    is_context_overflow,
    pack_docs,
    truncate_to_budget,
)
from rag.generatellm import RAGGenerator

# ── 中文语料：纯中文（1 字 ≈ 1.17 token）+ 表格行（ASCII 密集）──
ZH = "退货流程需要用户在订单页面提交申请，客服在二十四小时内审核，审核通过后原路退款。"
TABLE = "| 10 | AG-010 | 郑婷 | 2026-07 | 500 | 84.60 | 172,300.00 | A | 2800.00 |"


# ==================== token 计数 ====================
def test_count_tokens_chinese_is_about_one_per_char():
    """纯中文 ≈1 token/字（实测 0.8~1.17），字符数与 token 数同量级。"""
    ratio = count_tokens(ZH) / len(ZH)
    assert 0.6 < ratio < 1.6


def test_count_tokens_empty_is_zero():
    assert count_tokens("") == 0
    assert count_tokens(None) == 0


def test_estimate_tokens_never_underestimates_chinese_prose():
    """粗估是对数装箱的兜底，必须偏大 —— 低估会让装箱塞多。"""
    assert estimate_tokens(ZH) >= count_tokens(ZH)


def test_estimate_tokens_never_underestimates_table_rows():
    """表格/数字行 token 密度最高（实测 0.59 token/字符），粗估同样不能低于它。"""
    assert estimate_tokens(TABLE) >= count_tokens(TABLE)


def test_doc_text_accepts_document_and_plain_string():
    assert doc_text(Document(page_content="文档内容")) == "文档内容"
    assert doc_text("裸字符串") == "裸字符串"


# ==================== 预算换算 ====================
def test_input_budget_reserves_output_and_safety():
    """预算 = (窗口 - 输出预留) × (1 - 安全边际)。"""
    expected = int((DEFAULT_MODEL_CONTEXT - DEFAULT_MAX_OUTPUT) * (1 - 0.15))
    assert input_budget(DEFAULT_MODEL_CONTEXT, DEFAULT_MAX_OUTPUT, 0.15) == expected
    # 输出预留必须被扣掉：max_tokens 和输入共用同一个窗口
    assert input_budget(10000, 2000, 0.0) == 8000
    assert input_budget(10000, 4000, 0.0) == 6000


def test_input_budget_rejects_bad_params():
    with pytest.raises(ValueError):
        input_budget(model_context=0)
    with pytest.raises(ValueError):
        input_budget(model_context=1000, max_output=-1)
    with pytest.raises(ValueError):
        input_budget(model_context=1000, safety=1.5)


def test_input_budget_never_returns_non_positive():
    """输出预留比窗口还大时也不能返回 0（否则调用方会拿 0 当预算）。"""
    assert input_budget(model_context=1000, max_output=5000, safety=0.0) == 1


# ==================== 装箱 ====================
def _docs(*texts):
    return [Document(page_content=t, metadata={}) for t in texts]


def test_pack_docs_respects_budget():
    docs = _docs(*[TABLE + str(i) for i in range(20)])
    budget = 200
    result = pack_docs(docs, budget)

    assert result.used_tokens <= budget, "装箱结果不能超过预算"
    assert result.kept >= 1
    assert result.total == 20
    assert result.dropped == 20 - result.kept


def test_pack_docs_renumbers_contiguously_after_dropping():
    """核心：先丢弃、后编号 —— 编号必须连续，否则 [docN] 引用会错位。"""
    docs = _docs(*[f"第{i}段资料内容{TABLE}" for i in range(1, 11)])
    result = pack_docs(docs, 400)

    assert result.dropped > 0, "这个预算下应该丢块，否则测不到编号问题"
    text = result.format()
    numbers = [int(n) for n in re.findall(r"\[doc(\d+)\]", text)]
    assert numbers == list(range(1, result.kept + 1)), "编号必须连续且从 1 开始"
    # 被丢掉的内容不能出现在上下文里
    for i in range(result.kept + 1, 11):
        assert f"第{i}段资料内容" not in text


def test_pack_docs_declares_partial_context():
    """丢块必须显式声明，否则模型会给出"看似完整实则缺项"的答案。"""
    docs = _docs(*[f"资料{i}{TABLE}" for i in range(10)])
    result = pack_docs(docs, 300)

    assert result.partial is True
    text = result.format()
    assert "资料完整性说明" in text
    assert f"{result.kept}/{result.total}" in text
    # 不想要这段声明时可以关掉
    assert "资料完整性说明" not in result.format(declare_partial=False)


def test_pack_docs_complete_context_has_no_notice():
    docs = _docs("短资料一", "短资料二")
    result = pack_docs(docs, 10000)

    assert result.dropped == 0 and result.truncated is False
    assert result.partial is False
    assert "资料完整性说明" not in result.format()


def test_pack_docs_truncates_when_nothing_fits_yet():
    """第一块就装不下 → 按句子截断塞进去（总比什么都不给模型好）。"""
    long_doc = ZH * 50  # 远超预算
    result = pack_docs([Document(page_content=long_doc)], 100)

    assert result.kept == 1
    assert result.truncated is True
    assert result.dropped == 0
    assert result.used_tokens <= 100
    # 装进去的是被截断后的文本（结尾带省略号）
    assert result.docs[0].endswith("…")
    assert "资料完整性说明" in result.format()


def test_pack_docs_drops_tail_instead_of_truncating_later_blocks():
    """后续块装不下时丢尾，不做截断（行为可预测：只截第一块）。"""
    docs = _docs(TABLE, ZH * 50, "尾巴资料")
    result = pack_docs(docs, 120)

    assert result.kept == 1
    assert result.truncated is False
    assert result.dropped == 2


def test_pack_docs_empty_inputs():
    assert pack_docs([], 1000).format() == ""
    assert pack_docs(_docs("资料"), 0).format() == ""
    assert pack_docs([], 1000).dropped == 0


def test_format_docs_within_budget_returns_text_and_detail():
    text, result = format_docs_within_budget(_docs("资料甲", "资料乙"), 10000)

    assert isinstance(result, PackResult)
    assert text.startswith("[doc1] 资料甲")
    assert "[doc2] 资料乙" in text


# ==================== 句子级截断 ====================
def test_truncate_to_budget_cuts_at_sentence_boundary():
    """按句号切，不切在半句上。"""
    out = truncate_to_budget(ZH * 20, 60)

    assert count_tokens(out) <= 60
    assert out.endswith("…")
    # 去掉省略号后，最后一个实义字符应是句末标点
    assert out[:-1].rstrip().endswith("。")


def test_truncate_to_budget_returns_original_when_it_fits():
    assert truncate_to_budget("短文本。", 1000) == "短文本。"
    assert truncate_to_budget("", 100) == ""
    assert truncate_to_budget("文本", 0) == ""


def test_truncate_to_budget_falls_back_to_chars_without_punctuation():
    """超长无标点串（如 OCR 出来的一整段）只能按字符兜底，但不能返回空。"""
    out = truncate_to_budget("字" * 5000, 200)

    assert out
    assert count_tokens(out) <= 200


# ==================== 超限判定 ====================
def test_is_context_overflow_recognises_real_messages():
    assert is_context_overflow(
        RuntimeError("This model's maximum context length is 32768 tokens")
    )
    assert is_context_overflow(
        RuntimeError("Error code: 400 - context_length_exceeded")
    )

    class _Err(Exception):
        status_code = 400

    assert is_context_overflow(_Err("too many tokens, please reduce the length"))


def test_is_context_overflow_is_conservative():
    """普通错误不能被当成超限 —— 否则会误降级，掩盖真实故障。"""
    assert not is_context_overflow(RuntimeError("dashscope 500 internal error"))

    class _Bad400(Exception):
        status_code = 400

    assert not is_context_overflow(_Bad400("invalid temperature parameter"))
    assert not is_context_overflow(ValueError("接口密钥无效"))


# ==================== 预检版生成 ====================
class _FakeCompletions:
    """假客户端：记录每次调用的 messages，可按规则抛错。"""

    def __init__(self, fail_on=None, raise_times=0):
        self.calls = []
        self._fail_on = fail_on  # 命中该子串的调用抛超限错
        self._raise_times = raise_times  # 只抛前 N 次（模拟"降级后成功"）

    def create(self, **kwargs):
        content = kwargs["messages"][-1]["content"]
        self.calls.append(kwargs)
        if self._fail_on and self._fail_on in content:
            if self._raise_times == 0 or len(self.calls) <= self._raise_times:
                raise RuntimeError(
                    "Error code: 400 - context_length_exceeded: maximum context length"
                )
        return type(
            "_Result",
            (),
            {
                "choices": [
                    type("_Choice", (), {"message": type("_Msg", (), {"content": "模型答案"})()})()
                ]
            },
        )()


def _generator(completions):
    gen = RAGGenerator()
    client = type("_Client", (), {})()
    client.chat = type("_Chat", (), {"completions": completions})()
    gen.generate_client = client
    return gen


def test_generate_within_budget_happy_path(monkeypatch):
    completions = _FakeCompletions()
    gen = _generator(completions)

    out = gen.generate_answer_within_budget("问题", _docs(TABLE, TABLE), budget_tokens=500)

    assert out["answer"] == "模型答案"
    assert out["degraded"] is False
    assert out["retries"] == 0
    assert out["used_tokens"] <= 500
    assert len(completions.calls) == 1
    # 送出去的上下文确实带编号，且 prompt 用的是原有 RAG_PROMPT 模板
    sent = completions.calls[0]["messages"][-1]["content"]
    assert "[doc1]" in sent and "【参考资料】" in sent


def test_generate_within_budget_drops_tail_and_declares_partial():
    completions = _FakeCompletions()
    gen = _generator(completions)
    docs = _docs(*[f"资料{i}{TABLE}" for i in range(30)])

    out = gen.generate_answer_within_budget("问题", docs, budget_tokens=200)

    assert out["dropped"] > 0
    sent = completions.calls[0]["messages"][-1]["content"]
    assert "资料完整性说明" in sent, "丢块后必须告诉模型资料不完整"


def test_generate_within_budget_retries_with_smaller_budget():
    """被判超限 → 预算减半重试，且第二次送出的上下文更短。"""
    completions = _FakeCompletions(fail_on="【参考资料】", raise_times=1)
    gen = _generator(completions)
    docs = _docs(*[f"资料{i}{TABLE}" for i in range(30)])

    out = gen.generate_answer_within_budget("问题", docs, budget_tokens=1000)

    assert out["answer"] == "模型答案"
    assert out["retries"] == 1
    assert out["degraded"] is False
    assert len(completions.calls) == 2
    first = completions.calls[0]["messages"][-1]["content"]
    second = completions.calls[1]["messages"][-1]["content"]
    assert len(second) <= len(first)


def test_generate_within_budget_falls_back_to_map_reduce():
    """三次都超限 → 走 map-reduce：分段提取 + 合并，最终仍给出答案。"""
    # 只让"最终生成"（带【回答要求】的 RAG_PROMPT）超限，map/reduce 的提示词放行
    completions = _FakeCompletions(fail_on="【回答要求】")
    gen = _generator(completions)
    docs = _docs(*[f"资料{i}{TABLE}" for i in range(30)])

    out = gen.generate_answer_within_budget("问题", docs, budget_tokens=300)

    assert out["degraded"] is True
    assert out["answer"] == "模型答案"
    # 3 次预算尝试 + 至少 2 批 map + 1 次 reduce
    assert len(completions.calls) >= 6
    assert any("【局部结果】" in c["messages"][-1]["content"] for c in completions.calls), \
        "应该走到 reduce 合并阶段"


def test_generate_within_budget_reraises_non_overflow_errors():
    """非超限异常原样上抛 —— 不掩盖真实故障（rag_node 的兜底逻辑照旧生效）。"""
    completions = _FakeCompletions(fail_on="接口密钥", raise_times=0)

    class _Boom(_FakeCompletions):
        def create(self, **kwargs):
            raise RuntimeError("接口密钥无效")

    gen = _generator(_Boom())

    with pytest.raises(RuntimeError, match="接口密钥无效"):
        gen.generate_answer_within_budget("问题", _docs("资料"), budget_tokens=1000)


def test_generate_within_budget_without_docs():
    """没有资料时行为与 generate_answer 一致：照样让模型按 prompt 回答。"""
    completions = _FakeCompletions()
    gen = _generator(completions)

    out = gen.generate_answer_within_budget("问题", [])

    assert out["answer"] == "模型答案"
    assert out["kept"] == 0 and out["dropped"] == 0
    assert out["degraded"] is False


def test_existing_generate_answer_still_works(monkeypatch):
    """回归保护：原有 generate_answer 的行为不受新增代码影响。"""
    completions = _FakeCompletions()
    gen = _generator(completions)

    answer = gen.generate_answer("问题", _docs("资料"))

    assert answer == "模型答案"
    # 老方法不做预算装箱：30 段资料会被原样全塞进去
    sent = completions.calls[0]["messages"][-1]["content"]
    assert "[doc1]" in sent


# ==================== 切分方式取证 ====================
def test_default_separators_hard_split_chinese_mid_sentence():
    """取证：默认 separators 末位是 ""（逐字符硬切），中文会被切在词中间。

    中文没有空格，默认列表 ['\\n\\n', '\\n', ' ', ''] 的前三级都命中不了，
    直接掉到最后一层按字符硬切 —— 所以"不按语义切"是真的。
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text = ZH * 10  # 无换行的中文长段落
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)

    assert splitter._separators[-1] == "", "默认最后一层就是字符级兜底"
    chunks = splitter.split_text(text)
    assert len(chunks) > 1
    # 切点落在句子中间：前一块不以任何句末标点结尾
    assert not chunks[0].rstrip().endswith(("。", "！", "？", "；"))


def test_chinese_punctuation_separators_cut_at_sentence_boundary():
    """对策：把中文标点加进 separators，切点就落回句末。"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text = ZH * 10
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=0,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        keep_separator="end",
    )
    chunks = splitter.split_text(text)

    assert len(chunks) > 1
    for chunk in chunks[:-1]:
        assert chunk.rstrip().endswith(("。", "！", "？", "；")), (
            f"非末块必须切在句末，实际结尾: {chunk[-12:]!r}"
        )
