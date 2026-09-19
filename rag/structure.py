"""文档结构解析 + 两级切分。

## 两级切分

检索粒度和生成粒度天生矛盾：
- **embedding 要小块**：大块里只有一小部分和问题相关时，向量的语义重心被无关
  内容拉偏，BM25 也会被文档长度归一化压低命中词权重。所以打分单元要小（子块）。
- **LLM 要大块**：几百字的碎片缺章节标题、缺表头，单独读不懂（"费用为 300 元"
  到底指哪一项）。所以送给生成侧的单元要大（父块）。

一件事拆成两件事，各用各的最优粒度：**小块打分，大块生成。**

    split_parents   文本 → 父块（~1200 字）：写 MySQL，给 LLM 读
    split_children  父块 → 子块（~300 字）：写 Chroma，参与检索打分

检索时用命中的子块拿到 parent_id，再按主键把父块全文取回来（small-to-big）。

## 解析层

解析层产出 `Section`，切分器只认这个协议，与文件类型解耦：

    text          正文（Markdown 标题行已写回开头，保证块自解释）
    section_path  标题路径 ["第2章", "2.3 计费规则"]
    order         文档内顺序
    page_start/page_end  页码（PDF 有，其他类型没有）
    atomic        True = 不可切分（表格窗口 / 图片 OCR）
    kind          "text" | "table" | "image"
    extra         文件类型特有的元数据（如 Excel 的 sheet/rows）
"""

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

# 切分参数版本：改了切分参数就要改它，documents.chunk_schema_ver 靠它
# 定位"哪些文档需要重新解析"，不必全库盲灌
CHUNK_SCHEMA_VER = "v3-simple-1"

# ── 切分参数（就是两个数字，写在函数默认参数里）─────────────
# split_parents  target=1200  父块软目标：切到这么多字左右就断（给 LLM 读）
# split_children size=300     子块目标长度（给检索打分）

# 子块切分不看模型、不打分：切点由文本自身的结构决定（见 split_children）。

# 父块/子块共用的分隔符层级，从"最该切的地方"到"最不该切的地方"逐级降级。
# **中文句读必须显式列出**：RecursiveCharacterTextSplitter 的默认分隔符是
# ["\n\n", "\n", " ", ""]，没有中文标点；中文长段落里空格极少，于是会一路
# 降到 "" 那一层按字符硬砍 —— 句子被拦腰截断。
SPLIT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", "、", " ", ""]

# ── 子块切分的"语义边界强度" ───────────────────────────────
# 强度 = "这个位置有多该切"，数字越大越优先：
#   3 章节缝：空行、或紧跟着一行标题 —— 块不该横跨两个小节
#   2 句子缝：。！？； 与换行
#   1 从句缝：，、： —— 只在窗口里找不到更强的缝时才用
#   0 禁用：标题行之后（会把标题孤立在上一块末尾）、文本结尾
_STRENGTH_SECTION = 3
_STRENGTH_SENTENCE = 2
_STRENGTH_CLAUSE = 1
_STRENGTH_NONE = 0

# 行内分隔符：句末标点 / 从句标点。**换行不进这个正则** —— 行边界由
# splitlines(keepends=True) 负责，这样"表格行"才能整体当一个不可分单元。
_CLAUSE_DELIM_RE = re.compile(r"[。！？；]+|[，、：]+")

# ── 表格 ───────────────────────────────────────────────────
# docx/pdf 表格的切窗字符预算（Excel 另见 rag.XLSX_MAX_CHARS）
TABLE_WINDOW_CHARS = 1200
TABLE_MAX_ROWS = 50

# 标题行的最大长度：超过就不当标题（防止整段话被误判成标题）
MAX_HEADING_CHARS = 80

# ── 标题识别 ───────────────────────────────────────────────
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_CN_CHAPTER_RE = re.compile(r"^第\s*[一二三四五六七八九十百零〇\d]+\s*[章篇部]")
_CN_ORDINAL_RE = re.compile(r"^[一二三四五六七八九十]+\s*、")
# 多级数字标题（2.3 / 2.3.1），要求至少两段，避免把 "2999.5 元" 误判成标题
_NUM_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)+)[\s、.]")
# 句末标点：以这些结尾的行不是标题
_SENTENCE_END = ("。", "！", "？", "；", "，")

# ── Markdown 表格识别 ──────────────────────────────────────
_TABLE_SEP_CHARS_RE = re.compile(r"^[\s|:\-]+$")
_DIGITS_RE = re.compile(r"\d+")


# ==================== 数据结构 ============================


@dataclass
class Section:
    """解析层产出的结构单元。切分器只认这个协议，与文件类型解耦。"""

    text: str
    section_path: list[str]
    order: int
    page_start: int | None = None
    page_end: int | None = None
    atomic: bool = False
    kind: str = "text"
    # 文件类型特有的附加元数据。
    # 例：Excel 带 {"sheet": "参数表", "rows": "2-51"}，为"只查某张表"的
    # 元数据过滤留字段。**值必须是 str/int/float/bool，不能是 None**
    # （Chroma 拒绝 None）
    extra: dict[str, object] = field(default_factory=dict)


# ==================== 稳定主键 ============================
# 全部用确定性 hash（而非随机 id）：同一个文件重新入库得到同样的 id。
# 列宽都是 varchar(32)，所以 "前缀(2) + sha1前30位" 正好 32。


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def doc_id_of(source: str) -> str:
    """文档主键 = sha1(source)。source 是绝对路径，所以不要换目录/改文件名。"""
    return "d_" + _sha1(source)[:30]


def parent_id_of(source: str, order: int) -> str:
    """父块主键 = sha1(source | 文档内第几个父块)。

    **只用 source + 序号，不用正文**：正文改一个字 id 就全变，重新上传
    会变成"新增一份"而不是覆盖旧的那份，库里越堆越多。
    """
    return "p_" + _sha1(f"{source}|{order}")[:30]


def child_id_of(parent_id: str, index: int) -> str:
    """子块主键。用作 Chroma 的 id，让重复入库天然幂等。"""
    return "c_" + _sha1(f"{parent_id}|{index}")[:30]


# ==================== 标题识别 ============================


def heading_of(line: str, *, markdown: bool = True) -> tuple[int, str] | None:
    """判断一行是不是标题，返回 (层级, 标题文本)。不是则返回 None。

    两条防线避免误判：
    1. 长度上限 MAX_HEADING_CHARS —— 整段话不会因为以"一、"开头就被当标题
    2. 句末标点 —— 以 。！？；， 结尾的是句子，不是标题
    """
    s = line.strip()
    if not s or len(s) > MAX_HEADING_CHARS:
        return None

    if markdown:
        m = _MD_HEADING_RE.match(s)
        if m:
            title = m.group(2).strip()
            if title:
                # Markdown 的 # 是显式声明，不需要上面的句末标点防线
                return min(len(m.group(1)), 6), title

    if s.endswith(_SENTENCE_END):
        return None

    if _CN_CHAPTER_RE.match(s) or _CN_ORDINAL_RE.match(s):
        return 1, s

    m = _NUM_HEADING_RE.match(s)
    if m:
        # 2.3 → 层级 2；2.3.1 → 层级 3
        return min(m.group(1).count(".") + 1, 4), s

    return None


# ==================== Markdown 表格识别与切窗 ============================


def is_table_separator(line: str) -> bool:
    """判断是不是 Markdown 表格的分隔行（|---|---|）。

    只允许 | - : 空白四种字符，且必须同时出现 | 和 -，
    避免把正文里的普通竖线误判成表格。
    """
    s = line.strip()
    if not s or "|" not in s or "-" not in s:
        return False
    return bool(_TABLE_SEP_CHARS_RE.match(s))


def starts_table(lines: list[str], index: int) -> bool:
    """从 index 处是否开始一张 Markdown 表格（当前行 + 下一行是分隔行）。"""
    if index + 1 >= len(lines):
        return False
    if "|" not in lines[index]:
        return False
    return is_table_separator(lines[index + 1])


def take_table(lines: list[str], index: int) -> tuple[list[str], int]:
    """收集从 index 开始的整张表格，返回 (表格行, 下一行下标)。"""
    block: list[str] = []
    n = len(lines)
    while index < n and "|" in lines[index]:
        block.append(lines[index].rstrip())
        index += 1
    return block, index


def window_table_lines(
    head_md: str,
    row_lines: list[str],
    *,
    max_chars: int = TABLE_WINDOW_CHARS,
    max_rows: int = TABLE_MAX_ROWS,
) -> list[tuple[str, int, int]]:
    """把表格数据行按"字符预算 + 行数上限"切成若干窗口，**每个窗口都重复表头**。

    返回 [(窗口 Markdown, 起始行下标, 结束行下标)]，下标是相对 row_lines 的 0 基下标，
    由调用方映射回真实行号（Excel 行号 / 表格行号）。

    为什么必须重复表头：块是逐块入库、逐块被检索出来的。数据行脱离表头就没法理解
    —— "2999 | 24 个月" 到底是价格还是库存？
    """
    windows: list[tuple[str, int, int]] = []
    buf: list[str] = []
    buf_chars = len(head_md)
    start = 0

    for index, line in enumerate(row_lines):
        if buf and (len(buf) >= max_rows or buf_chars + len(line) + 1 > max_chars):
            windows.append((head_md + "\n" + "\n".join(buf), start, index - 1))
            buf, buf_chars = [], len(head_md)
            start = index
        if not buf:
            start = index
        buf.append(line)
        buf_chars += len(line) + 1

    if buf:
        windows.append((head_md + "\n" + "\n".join(buf), start, len(row_lines) - 1))
    return windows


def window_markdown_table(
    block: list[str],
    *,
    max_chars: int = TABLE_WINDOW_CHARS,
    max_rows: int = TABLE_MAX_ROWS,
) -> list[str]:
    """把一张完整的 Markdown 表格按预算切窗，返回若干段可直接入库的表格文本。

    预算内则原样返回一段 —— 短表格不受任何影响。
    """
    if len(block) < 2:
        joined = "\n".join(block).strip()
        return [joined] if joined else []

    head_md = "\n".join(block[:2])  # 表头行 + 分隔行
    rows = block[2:]
    total = len(head_md) + sum(len(r) + 1 for r in rows)

    if total <= max_chars and len(rows) <= max_rows:
        return ["\n".join(block)]

    return [
        piece
        for piece, _, _ in window_table_lines(
            head_md, rows, max_chars=max_chars, max_rows=max_rows
        )
    ]


# ==================== 结构解析 ============================


def _tokenize(text: str, *, markdown: bool, detect_tables: bool) -> list[tuple]:
    """把文本切成三类 token：heading / table / text。

    先 token 化再组装，是为了让"标题路径跨页延续"这类状态管理变得简单
    （PDF 是逐页解析的，但章节标题在页之间是连续的）。
    """
    items: list[tuple] = []
    buf: list[str] = []
    lines = text.split("\n")
    i, n = 0, len(lines)

    while i < n:
        line = lines[i]
        h = heading_of(line, markdown=markdown)
        if h:
            if buf:
                items.append(("text", buf))
                buf = []
            items.append(("heading", h[0], h[1]))
            i += 1
            continue
        if detect_tables and starts_table(lines, i):
            if buf:
                items.append(("text", buf))
                buf = []
            block, i = take_table(lines, i)
            items.append(("table", block))
            continue
        buf.append(line)
        i += 1

    if buf:
        items.append(("text", buf))
    return items


def path_of(levels: dict[int, str]) -> list[str]:
    """把"按层级索引的标题"拍平成路径列表（层级从小到大）。"""
    return [levels[level] for level in sorted(levels)]


def sections_from_text(
    text: str,
    *,
    markdown: bool = True,
    detect_tables: bool = True,
    page: int | None = None,
    start_levels: dict[int, str] | None = None,
    start_order: int = 0,
) -> tuple[list[Section], dict[int, str], int]:
    """把一段文本解析成结构单元。

    返回 (sections, 结束时的层级映射, 下一个可用的 order)。
    后两个返回值是为了 PDF 逐页解析时能把标题路径和序号**跨页延续** ——
    否则每页都会从"没有标题"重新开始，长 PDF 的结构就全丢了。

    为什么用"层级→标题"的映射，而不是一个按位置截断的列表：
    文档的第一级标题未必是 `#`（很多文档直接从 `##` 开始，Word 的 Heading 1
    常常就是文档标题）。若用 path[:level-1] 截断，会出现"2.1 是深度 2、
    2.2 是深度 3"这种深度与层级错位的情况 —— 后果是同级判断失效，
    该合并的不合并。用层级做键之后，同层级的 section 深度必然一致。

    markdown=False 时只认中文编号标题（用于纯文本 txt）。

    **标题行会写回 block 正文开头**（不只在 metadata 里）：
    标题行如果被丢掉，小节名字就彻底消失了。写回正文后，每个块都
    自带自己的标题，单独读起来还是自解释的。
    """
    items = _tokenize(text, markdown=markdown, detect_tables=detect_tables)
    levels: dict[int, str] = dict(start_levels or {})
    sections: list[Section] = []
    order = start_order
    pending = ""  # 当前标题行，挂到下一个内容块前面

    for item in items:
        kind = item[0]
        if kind == "heading":
            _, level, title = item
            # 只保留比当前标题更浅的层级，再挂上新标题（等价于"层级回退"）
            levels = {l: t for l, t in levels.items() if l < level}
            levels[level] = title
            pending = f"{'#' * level} {title}"
            continue

        if kind == "table":
            for index, piece in enumerate(window_markdown_table(item[1])):
                # 标题行只挂到这张表的第一个窗口上，后续窗口靠重复表头自解释
                body = f"{pending}\n{piece}" if (pending and index == 0) else piece
                sections.append(
                    Section(
                        text=body,
                        section_path=path_of(levels),
                        order=order,
                        page_start=page,
                        page_end=page,
                        atomic=True,  # 表格不参与二次切分
                        kind="table",
                    )
                )
                order += 1
            pending = ""
            continue

        body = "\n".join(item[1]).strip()
        if body:
            if pending:
                body = f"{pending}\n{body}"
            sections.append(
                Section(
                    text=body,
                    section_path=path_of(levels),
                    order=order,
                    page_start=page,
                    page_end=page,
                )
            )
            order += 1
            pending = ""

    return sections, levels, order


# ==================== PDF 页眉页脚去噪 ============================


def drop_repeated_lines(
    pages: list[str],
    *,
    min_pages: int = 4,
    ratio: float = 0.6,
    max_len: int = 60,
) -> tuple[list[str], list[str]]:
    """删掉"几乎每页都出现"的短行 —— 也就是页眉页脚。

    长文档里每页都印着公司名/文档标题/页码，几百页就是几百份重复文本进库，
    同时污染两路检索：BM25 被高频无意义词拉偏、向量空间里这些块互相靠近。

    三重保守约束，避免误删正文：
    1. 页数 >= min_pages（样本太少时"重复"没有说服力）
    2. 出现页数 >= ratio * 总页数
    3. 行长 <= max_len，且跳过表格行（以 | 开头）

    计数时把数字归一成 #，这样"第 3 页""第 4 页"能归为同一条。

    返回 (处理后的页文本, 被删掉的模式)，模式用于打日志核对。
    """
    if len(pages) < min_pages:
        return pages, []

    counts: Counter[str] = Counter()
    for text in pages:
        seen = set()
        for line in text.split("\n"):
            s = line.strip()
            if not s or len(s) > max_len or s.startswith("|"):
                continue
            norm = _DIGITS_RE.sub("#", s)
            if norm not in seen:
                seen.add(norm)
                counts[norm] += 1

    threshold = max(2, int(len(pages) * ratio))
    drop = {k for k, v in counts.items() if v >= threshold}
    if not drop:
        return pages, []

    cleaned: list[str] = []
    for text in pages:
        kept = []
        for line in text.split("\n"):
            s = line.strip()
            norm = _DIGITS_RE.sub("#", s)
            if s and len(s) <= max_len and not s.startswith("|") and norm in drop:
                continue
            kept.append(line)
        cleaned.append("\n".join(kept))

    return cleaned, sorted(drop)


# ==================== 两级切分 ============================


def split_parents(text: str, target: int = 1200) -> list[str]:
    """父块：**递归切分**。

    递归 = 先尽量在"段落"边界切，单段太长才降级到换行 → 句号 → 逗号 → 最后按
    字符硬切。所以它既保留了"短段落会被合并成一块"的效果，又给单段长度上了限。

    为什么不能只按空行聚合：那样一个 8000 字的段落会原样变成一个 8000 字的父块，
    没有任何上限。父块是给 LLM 读的，超长会挤掉别的内容。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=target,
        chunk_overlap=0,  # 父块之间不重叠：重叠会让同一段正文进两个父块，白占库和白占上下文
        separators=SPLIT_SEPARATORS,
        keep_separator="end",  # 分隔符留在前一块末尾，块不会以标点开头
    )
    return [s for s in splitter.split_text(text) if s.strip()] or [text]


def split_children(parent_text: str, size: int = 300, overlap: int = 50) -> list[str]:
    """子块：**结构语义切分**

    切点完全由文本自身的结构决定，三级强度见 _STRENGTH_*：
    章节缝（空行 / 标题前）> 句子缝（。！？；与换行）> 从句缝（，、：）。

    算法两步：
    1. 把父块切成"从句单元"（每段文字带上结束它的分隔符；表格行、标题行各自
       整体成单元，不可再分）—— 见 _clause_units；
    2. 按目标大小合并成块、只在高强度缝处断开 —— merge_units；再把过短的并进
       前一块、过长的兜底切开 —— rebalance。
    """
    text = parent_text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    # 父块本身就装着整张 Markdown 表格 → 不切。
    # 理由：数据行脱离表头就没法理解（"2999 | 24个月"是价格还是库存？），
    # 而且实测表格是这套链路里最容易被切坏的东西（11.docx 18 张表）。
    # 宁可让它超一点长度，也不把它切碎。
    # 判断是不是表格块，防止把表格切了
    if _is_table_block(text):
        return [text]
    # 大块切分
    units = _clause_units(text)
    # 切分之后进行
    return rebalance(merge_units(units, target_size=size))


def _is_table_block(text: str) -> bool:
    """判断"基本就是一张 Markdown 表格"（允许前面挂一行标题）。

    判据：有表头分隔行，且几乎每行都是表格行。用来避免把表格切碎。
    """
    lines = [ln for ln in text.split("\n") if ln.strip()]
    rows = [ln for ln in lines if ln.lstrip().startswith("|")]
    if len(rows) < 3 or len(rows) < len(lines) - 1:
        return False
    return any(is_table_separator(ln) for ln in rows[:3])


# ── 语义边界：从句单元 + 合并成块 ───────────────────────────


def _strength_of(delims: str) -> int:
    """分隔符串 → 边界强度：句末标点比从句标点更该切。"""
    return (
        _STRENGTH_SENTENCE
        if any(ch in delims for ch in "。！？；")
        else _STRENGTH_CLAUSE
    )


def _clause_units(text: str) -> list[tuple[str, int]]:
    """父块 → [(单元文本, 该单元之后的边界强度)]。

    单元 = 一段文字 + 结束它的分隔符。**分隔符留在左边**，所以切出来的块不会
    以标点开头（与 split_parents 的 keep_separator="end" 一致）。

    三条不可分的硬规则：
    1. **表格行整行一个单元**：行内的逗号/冒号不算边界 —— 一行表格数据被从句读
       切开，续块就没有列对齐，也更没法跟表头对上；
    2. **标题行整行一个单元**，且其后的边界强度为 0：标题留在上一块末尾等于
       这一块没有标题，不如把这道缝让给下一块；
    3. 空行并进上一个单元，并把那道缝抬到章节级（空行本身就是小节分界）。
    """
    # 给每一个标点加上强度，方便后续进行合并
    # 比如说按空行切，给units最后一个元素 的文本内容加上一个数字
    units: list[tuple[str, int]] = []
    # 按行切
    for raw in text.splitlines(keepends=True):
        body = raw.rstrip("\n")
        newline = raw[len(body) :]

        if not body.strip():  # 空行 = 章节缝，并进上一个单元
            if units:
                prev_text, prev_strength = units[-1]
                units[-1] = (prev_text + raw, max(prev_strength, _STRENGTH_SECTION))
            continue

        if body.lstrip().startswith("|"):  # 表格行：整行不可分
            units.append((raw, _STRENGTH_SENTENCE if newline else _STRENGTH_NONE))
            continue

        if heading_of(body) is not None:  # 标题行：不可分，且之后不切
            if units:
                prev_text, prev_strength = units[-1]
                units[-1] = (prev_text, max(prev_strength, _STRENGTH_SECTION))
            units.append((raw, _STRENGTH_NONE))
            continue

        start = 0
        for m in _CLAUSE_DELIM_RE.finditer(body):
            units.append((body[start : m.end()], _strength_of(m.group())))
            start = m.end()
        tail = body[start:]
        if tail:
            units.append(
                (tail + newline, _STRENGTH_SENTENCE if newline else _STRENGTH_NONE)
            )
        elif newline and units:
            # 整行正好以标点收尾：换行并进最后一个单元，缝抬到句子级
            prev_text, prev_strength = units[-1]
            units[-1] = (prev_text + newline, max(prev_strength, _STRENGTH_SENTENCE))

    return units


def merge_units(
    units: list[tuple[str, int]],
    target_size: int = 300,
    min_strength: int = _STRENGTH_SENTENCE,  # 只在高强度处断
) -> list[str]:
    """把细单元按目标大小合并成子块，只在高强度缝处断开。

    **表格内部不断开**：切在表格的两行之间，右边那半张表就没有表头了
    （见 _inside_table_boundary），所以这种缝要跳过，等表格结束再断 ——
    代价是这个块会超过 target_size，但表格完整性优先。
    """
    chunks: list[str] = []
    buf = ""

    for index, (text, strength) in enumerate(units):
        buf += text
        # 攒够了 + 当前缝够强 → 在这里断开
        if len(buf) < target_size or strength < min_strength:
            continue
        next_text = units[index + 1][0] if index + 1 < len(units) else ""
        if next_text and _inside_table_boundary(text, next_text):
            continue  # 表格内部：等这张表结束再断
        chunks.append(buf)
        buf = ""

    if buf:
        chunks.append(buf)
    return chunks


def rebalance(
    chunks: list[str], min_chars: int = 80, max_chars: int = 500
) -> list[str]:
    """过短的并入前一块，过长的硬切。

    硬切是兜底（merge_units 已经按语义缝断过一遍），所以只在真的超长时才发生，
    而且尽量不破坏结构：
    1. 优先退到 max_chars 之前的**换行**处下刀，而不是按字符数硬砍；
    2. 这一刀若落在表格内部，就往右挪到这张表结束 —— 同样宁可超长，也不切出
       没有表头的续块。
    """
    # 先处理过短
    merged: list[str] = []
    for c in chunks:
        if merged and len(c) < min_chars:
            merged[-1] += c
        else:
            merged.append(c)

    # 再处理过长
    final: list[str] = []
    for c in merged:
        if len(c) <= max_chars:
            final.append(c)
            continue
        start = 0
        while len(c) - start > max_chars:
            cut = c.rfind("\n", start, start + max_chars)
            if cut <= start:
                cut = start + max_chars  # 整段没有换行可退 → 只能按字符切
            # 表格内部不下刀：往右挪到这张表结束
            while cut < len(c) and _inside_table_boundary(c[start:cut], c[cut:]):
                nxt = c.find("\n", cut)
                if nxt < 0:
                    cut = len(c)
                    break
                cut = nxt + 1
            piece = c[start:cut].strip()
            if piece:
                final.append(piece)
            start = cut
        tail = c[start:].strip()
        if tail:
            final.append(tail)
    return final


def _inside_table_boundary(left: str, right: str) -> bool:
    """这个切点是不是落在表格内部（切了右边就没表头了）。

    两种情况算"表格内部"：
    1. 左边那行是表格行，右边紧接着也是表格行 → 切在表格的两行之间
    2. 左边那行是表格行，且这行还没结束 → 切在表格单元格中间（句号在单元格里）
    """
    left_tail = left.rstrip("\n").split("\n")[-1]
    if "|" not in left_tail:
        return False
    right_head = right.lstrip("\n").split("\n")[0]
    if right_head.lstrip().startswith("|"):
        return True
    return not left.endswith("\n") and "|" in right_head
