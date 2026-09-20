"""子块切分（structure.split_children）的单测。

零服务、零模型：新的切分是纯规则（结构语义边界），不加载 cross-encoder，
所以这些用例跑起来没有任何外部依赖。

守住四条不变量：
1. **不丢字符** —— 所有子块拼回去（忽略空白）等于原文；
2. **切点只落在语义边界上** —— 不会有块以半句话/半行开头；
3. **表格不被切坏** —— 含表格行的块必须自带表头分隔行，且没有行被截断；
4. **章节缝优先** —— 块不横跨两个小节，标题行不会被孤立在块尾。
"""
import re

from rag import structure as st

BOUND = "。！？；，、：\n \t"


def _nows(text: str) -> str:
    return re.sub(r"\s+", "", text)


def test_empty_and_short_text():
    assert st.split_children("") == []
    assert st.split_children("   ") == []
    assert st.split_children("短文本") == ["短文本"]


def test_pure_sentences_split_at_size():
    text = "客户在收到货后七天内可以申请无理由退货。" * 60  # 1200 字

    chunks = st.split_children(text)

    assert [len(c) for c in chunks] == [300, 300, 300, 300]
    assert all(c.endswith("。") for c in chunks)  # 切在句号后，不切在句子中间


def test_no_character_is_lost():
    text = "\n\n".join(
        [
            "第一章 退货政策。" + "七天之内可以无理由退货。" * 12,
            "第二章 运费说明。" + "运费由责任方承担。" * 15,
            "| 型号 | 价格 |\n| --- | --- |\n" + "".join(
                f"| Z9-{i:03d} | {2000 + i} |\n" for i in range(20)
            ),
        ]
    )

    chunks = st.split_children(text)

    assert _nows("".join(chunks)) == _nows(text)


def test_cut_points_are_semantic_boundaries():
    text = "第一节内容。" + "本节的说明文字都比较长。" * 25
    text += "\n\n" + "第二节内容。" + "换了一节之后说的是另一件事。" * 22

    chunks = st.split_children(text, size=300)

    for chunk in chunks[1:]:
        pos = text.find(chunk[:30])
        assert pos > 0
        assert text[pos - 1] in BOUND, f"块从半句话开始: {chunk[:20]!r}"


def test_chunk_never_spans_two_sections():
    """章节缝即使落在 size 之后，也优先于窗口内的弱缝。"""
    a = "第一节内容。" + "本节的说明文字都比较长。" * 25
    b = "第二节内容。" + "换了一节之后说的是另一件事。" * 22

    chunks = st.split_children(a + "\n\n" + b, size=300)

    assert not any("第一节" in c and "第二节" in c for c in chunks), "有块横跨了两个小节"


def test_heading_line_is_not_orphaned_at_chunk_end():
    doc = (
        "## 2.1 计费规则\n"
        + "按调用次数计费，每次零点零一元。" * 20
        + "\n\n## 2.2 退款规则\n"
        + "退款三个工作日到账，原路返回。" * 20
    )

    chunks = st.split_children(doc, size=300)

    for chunk in chunks:
        assert not chunk.rstrip().split("\n")[-1].startswith("#"), "标题行被孤立在块尾"
    # 标题本身一个字都不能丢
    body = "".join(chunks)
    assert "## 2.1 计费规则" in body
    assert "## 2.2 退款规则" in body


def test_table_rows_keep_their_header():
    """数据行脱离表头就没法理解：含表格行的块必须带分隔行，且不许有行被截断。"""
    text = (
        "以下是参数说明，请仔细阅读。\n表头与数据见下表。\n\n"
        "| 型号 | 价格 |\n| --- | --- |\n"
        + "".join(f"| Z9-{i:03d} | {2000 + i} |\n" for i in range(40))
    )

    chunks = st.split_children(text, size=150)

    for chunk in chunks:
        lines = chunk.splitlines()
        if not any(ln.lstrip().startswith("|") for ln in lines):
            continue
        assert any(st.is_table_separator(ln) for ln in lines), "表格块丢了表头"
        for line in lines:
            if line.lstrip().startswith("|"):
                assert line.rstrip().endswith("|"), f"表格行被截断: {line!r}"


def test_no_boundary_long_text_is_capped_not_lost():
    """整段没有一个分隔符时没有语义缝可切，由 rebalance 按 max_chars 兜底切开
    —— 但必须一个字都不丢。"""
    text = "A" * 400 + "B" * 400

    chunks = st.split_children(text)

    assert all(len(c) <= 500 for c in chunks), "超过 max_chars 说明兜底硬切没生效"
    assert "".join(chunks) == text
