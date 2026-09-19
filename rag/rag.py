"""
RAG 检索模块 —— 只负责文档加载、向量化、检索。
生成回答由 Agent 的 LLM 负责，本模块不创建 LLM 实例。
"""

import asyncio
import io
import os
import re
import threading
from pathlib import Path

import jieba

# import pymupdf4llm
import pytesseract
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document

# from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image
from rank_bm25 import BM25Okapi

# 父块存储（MySQL）。注意 db 包在 import 时不连库、不 import 驱动，
# 所以 MySQL 挂着也不会影响本模块启动
from db import (
    DocumentStore,
    MySQLUnavailable,
    parent_store,
)
from rag import structure as st
from rag.local_reranker import LocalReranker
from rag.structure import (
    CHUNK_SCHEMA_VER,
    Section,
)
from rag.vision_ocr import hybrid_image_text

# 清楚 chromadb 的缓存，避免报错
# import chromadb.api.shared_system_client as shared
# shared.SharedSystemClient._identifier_to_system.clear()
# import pymupdf4llm
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

load_dotenv(encoding="utf-8-sig")  # utf-8-sig:兼容带 BOM 的 .env

reranker = LocalReranker()


# ── 知识库异常 ─────────────────────────────────────────────
class KnowledgeBaseError(Exception):
    """知识库不可用（连接失败 / 未初始化 / 为空）。由调用方捕获后转成提示语。"""


# 知识库参数
_vectorstore = None
_retriever = None

# BM25 关键词检索参数（内存索引，重启后懒加载）
_bm25 = None  # BM25Okapi 索引
_bm25_corpus: list[str] = []  # 分词后的语料（与 _chunk_docs 对齐）
_chunk_docs: list[Document] = []  # 全量 chunk（含 metadata，用于映射回来源）

# 文件内容加载参数
_is_initialized: bool = False

# 初始化/重建的互斥锁（防止并发首检双重初始化）
_init_lock = threading.Lock()
_bm25_lock = threading.Lock()

# BM25 检索前统一清洗的标点（全角+空白），保证 query 与语料分词对齐
_PUNCT_RE = re.compile(r"[，。！？、；：‘’“”（）《》【】\s]+")

# ── 可调参数 ─────────────────────────────────────────────
# 切分参数不在这里了，在 rag/structure.py：
#   split_parents   target=1200  父块（生成粒度，大块读得懂）
#   split_children  size=300     子块（检索粒度，小块打分准）

# RRF 融合前每路的候选数（实际取 fetch_k = max(k*2, 6)）。
# 子块从 1000 字降到 450 字后，同样条数的候选覆盖的原文变少了，
# 所以从 20 提到 30 补偿召回。多取候选很便宜（打分的是小块），
# 真正贵的是重排，最终收敛交给 TOP_N。
# 旧注释（仍成立）：topk 增大只是增加候选切片数量，反而给重排增加工作量
TOP_K = 30

# 重排后送给生成侧的条数。
# 旧注释（仍成立）：调大能让检索内容更齐全，但更耗生成模型的 token，
# 也更容易引入不相关内容导致准确性下降。
# 语义变化：现在这个值作用在**子块**上（重排前是子块），聚合后变成父块
TOP_N = 10

# 父块聚合后最多送几个。必须 <= TOP_N，为预算留余量：
# 父块远大于子块，8 × 1200 已接近 10k 字符
PARENT_TOP_N = 8

# 送进上下文的父块总字符预算：装不下的跳过，继续试后面的（不是遇到就停）
PARENT_MAX_CHARS = 6000

# PDF 页眉页脚去噪开关。长 PDF 每页都印着公司名/页码，
# 几百页就是几百份重复文本，同时污染 BM25 和向量两路检索
PDF_DROP_REPEATED_LINES = True

# ── 扫描件（整页图片、无文本层）─────────────────────────────
# 判定：页内文本少于 SCANNED_PAGE_MIN_CHARS 且页内有图片对象 → 当扫描页处理。
# 为什么必须单独处理：pymupdf4llm 对整页图**不会**导出图片、也不会写 markdown
# 图片引用（实测：page_chunks 的 text 为空、image_path 目录为空），所以走
# "markdown 图片引用 → OCR" 那条路的话，扫描件会静默地 0 块入库 —— 而企业资料里
# 盖章合同/纸质手册扫描件占比很高。这里的做法是自己用 PyMuPDF 渲染整页再 OCR。
SCANNED_PAGE_MIN_CHARS = 20
# OCR 渲染精度：150 对中文小字偏糊，300 明显变慢；200 实测关键实体（型号/金额）可读
SCANNED_PAGE_DPI = 200
# tesseract 语言包（本机已装 chi_sim + eng）
OCR_LANG = "chi_sim+eng"

# 知识库默认目录：项目根目录下的 knowledge_base/
KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"
# 持久化目录
PERSIST_DIR = Path(__file__).resolve().parent.parent / "chroma_db"

IMAGES_DIR = Path(__file__).resolve().parent.parent / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)  # 确保目录存在，否则 pymupdf4llm 会报错


# 提取图片路径
def find_image_references(content: str) -> list[str]:
    """
    从文本内容中提取所有 Markdown 图片引用的文件名。
    例如：![alt text](images/image1.png) -> 提取 image1.png
    """
    import re

    pattern = r"!\[.*?\]\((.*?)\)"
    matches = re.findall(pattern, content)
    print(f"提取到的图片引用: {matches}")
    # 只保留文件名部分
    return [os.path.basename(match) for match in matches]


# ==================== 文档加载 ============================
# 所有 _load_* 的返回类型统一改成 list[Section]（rag/structure.py 的结构协议）：
#   Section(text, section_path, order, page_start, page_end, atomic, kind)
# 解析层只负责"把结构信息挖出来"，切几级、怎么切由 structure.py 决定。
# 这样文件类型与切分策略解耦，长文档和短文档走同一条管线。
def _load_pdf(file_path: str) -> list[Section]:
    """PDF → 结构单元。

    结构感知点：
    1. pymupdf4llm 输出的是 Markdown，所以页内的 # 就是章节标题。逐页解析时
       把标题路径和 order **跨页延续** —— 章节本来就是跨页的，逐页重置等于
       把长 PDF 的结构全部丢掉。
    2. 页眉页脚去噪：长 PDF 每页都印着公司名/文档标题/页码，几百页就是几百份
       重复文本，同时污染 BM25（高频无意义词拉偏 idf）和向量（页眉相似的块
       在向量空间里挤成一团）。
    3. 图片文字作为当前章节下的 atomic 单元，而不是堆到页尾/文末 ——
       否则"这张图属于哪一节"就丢了（OCR 是图片类文档的主要检索内容）。
       图片有两个来源，都要覆盖：
         a) 文字页里的插图：pymupdf4llm 会导出成 png 并写成 markdown 图片引用；
         b) **整页扫描件**：页内没有文本层、只有一张整页图。pymupdf4llm 对这种情况
            既不给引用也不导出图片（实测 text 为空、image_path 目录为空），
            所以必须自己用 PyMuPDF 判定并渲染整页 —— 见下面 _is_scanned_page
            与 _ocr_image_section，否则扫描件会静默 0 块入库。
    4. 页码修正：pymupdf4llm 的 chunk metadata 里键名是 **page_number**（原代码取
       .get("page", 0) 取不到，于是所有 PDF 块页码都是 0）。这里优先用
       page_number，取不到再退回页序（1 基）。
    """
    # 提取 Markdown（图片保存到绝对路径 IMAGES_DIR）
    import pymupdf
    import pymupdf4llm

    pages = pymupdf4llm.to_markdown(
        file_path,
        page_chunks=True,
        write_images=True,
        image_path=str(IMAGES_DIR),  # ① 写入：绝对路径，与 CWD 无关
        image_format="png",
        dpi=150,
    )

    # ② 先收集每页文本、图片引用、页码，再做跨页去噪（去噪需要跨页频次才能判断）
    page_texts: list[str] = []
    page_images: list[list[str]] = []
    page_numbers: list[int] = []
    for index, chunk in enumerate(pages):
        content = chunk.get("text", "")
        # 顺手把 Markdown 里的图片引用归一化成纯文件名，
        # 否则绝对路径会被存进 Chroma 的内容里，污染检索结果
        content = re.sub(
            r"!\[([^\]]*)\]\(([^)]+)\)",
            lambda m: f"![{m.group(1)}]({os.path.basename(m.group(2))})",
            content,
        )
        page_texts.append(content)
        page_images.append(find_image_references(content))
        page_numbers.append(int(chunk.get("metadata", {}).get("page_number") or index + 1))

    if PDF_DROP_REPEATED_LINES:
        page_texts, dropped = st.drop_repeated_lines(page_texts)
        if dropped:
            print(f"[PDF] 去掉 {len(dropped)} 种重复行（页眉/页脚），如: {dropped[:5]}")

    sections: list[Section] = []
    level_map: dict[int, str] = {}  # 跨页延续的"层级 → 标题"映射
    order = 0  # 跨页延续的文档内序号

    # 只为"整页扫描件"单独开一次 PDF：渲染是逐页懒做的（一页 200KB 级），
    # 不预渲染全部页面，避免几百页扫描件把内存吃光
    pdf = pymupdf.open(file_path)
    try:
        for index, content in enumerate(page_texts):
            page_no = page_numbers[index]

            page_sections, level_map, order = st.sections_from_text(
                content,
                markdown=True,
                page=page_no,
                start_levels=level_map,
                start_order=order,
            )
            sections.extend(page_sections)

            # ③a 文字页里的插图（pymupdf4llm 导出的 png）
            for img_ref in page_images[index]:
                img_path = IMAGES_DIR / img_ref  # 读取：绝对路径，与 CWD 无关
                if not img_path.exists():
                    continue
                section = _ocr_image_section(
                    img_path.read_bytes(),
                    label=f"图片 {img_ref}",
                    source=f"{file_path}#{img_ref}",
                    level_map=level_map,
                    order=order,
                    page_no=page_no,
                )
                if section is not None:
                    sections.append(section)
                    order += 1

            # ③b 整页扫描件：页内无文本层但整页是图 → 渲染整页再 OCR
            if page_no <= pdf.page_count and _is_scanned_page(pdf[page_no - 1]):
                pix = pdf[page_no - 1].get_pixmap(dpi=SCANNED_PAGE_DPI)
                section = _ocr_image_section(
                    pix.tobytes("png"),
                    label=f"扫描页 第{page_no}页",
                    source=f"{file_path}#scan-p{page_no}",
                    level_map=level_map,
                    order=order,
                    page_no=page_no,
                )
                if section is not None:
                    sections.append(section)
                    order += 1
    finally:
        pdf.close()

    return sections


def _is_scanned_page(page) -> bool:
    """判断是不是"整页扫描件"：页内几乎没有文本层，但有整页图片。

    阈值取 SCANNED_PAGE_MIN_CHARS：真扫描件页内文本长度是 0；而文字页即使只有
    一行标题也会超过 20 字符，所以不会误判。扫描页 + 少量文本（例如扫描件上盖了
    一层电子页码）也算扫描页，两份内容都会被保留。
    """
    text = page.get_text("text").strip()
    return len(text) < SCANNED_PAGE_MIN_CHARS and bool(page.get_images())


def _ocr_image_section(
    img_bytes: bytes,
    *,
    label: str,
    source: str,
    level_map: dict,
    order: int,
    page_no: int,
) -> Section | None:
    """图片字节 → 本地 OCR →（必要时）视觉模型增强 → atomic Section。

    抽成函数是因为有两条来源（文字页插图 / 整页扫描件），行为必须一致。
    返回 None 表示没识别出任何文字（该图对检索没有贡献，不入库）。
    视觉模型只在 VISION_OCR_MODE=auto 且本地 OCR 为空时才会被调用，见 vision_ocr.py。
    """
    import pytesseract
    from PIL import Image

    try:
        # with 是为了**关闭文件句柄**：原来直接 Image.open(路径) 交给 tesseract，
        # 句柄一直不释放，图多的 PDF 会累积 fd
        with Image.open(io.BytesIO(img_bytes)) as im:
            ocr_text = pytesseract.image_to_string(im, lang=OCR_LANG)
    except Exception as e:  # OCR 挂掉不应该毁掉整个文档，交给视觉模型兜底
        print(f"[OCR] {label} 本地识别失败，改由视觉模型兜底: {e!r}")
        ocr_text = ""

    final_text = hybrid_image_text(img_bytes, ocr_text, source=source)
    if not final_text.strip():
        print(f"[OCR] {label} 未识别出文字，跳过")
        return None

    print(f"[OCR] {label} → {len(final_text)} 字")
    return Section(
        text=f"【{label}】\n{final_text.strip()}",
        section_path=st.path_of(level_map),
        order=order,
        page_start=page_no,
        page_end=page_no,
        atomic=True,
        kind="image",
    )



def _load_txt(file_path: str) -> list[Section]:
    """TXT / MD → 结构单元。

    结构识别分两条路：
    - Markdown（后缀是 .md，或内容里出现了 # 标题行）：按 # 的层级切
    - 纯文本：按中文编号切（第X章/第X节、一、、2.3）

    识别不到任何标题时，整篇就塌缩成一个 section —— 这就是"短文档 / 无结构
    文本自动退化"的入口。**不需要任何字数阈值**：判据是"有没有结构"，
    而不是"有多少字"。
    """
    text = Path(file_path).read_text(encoding="utf-8")
    if not text.strip():
        return []

    markdown = file_path.lower().endswith(".md") or bool(
        re.search(r"^#{1,6}\s+\S", text, flags=re.MULTILINE)
    )
    sections, _, _ = st.sections_from_text(text, markdown=markdown)
    return sections


def _docx_heading_level(para) -> int | None:
    """从段落样式（Heading 1 / 标题 1）或 outlineLvl 取标题层级。

    取不到返回 None。样式名不规范的中文文档很多，所以加 outlineLvl 兜底
    （它是 Word 大纲级别的底层记录，0 基）。
    """
    style_name = ""
    try:
        if para.style is not None:
            style_name = para.style.name or ""
    except Exception:
        style_name = ""  # 样式表损坏时不影响正文提取

    m = re.match(r"^\s*(?:Heading|标题)\s*(\d+)", style_name, re.IGNORECASE)
    if m:
        return min(int(m.group(1)), 6)

    from docx.oxml.ns import qn

    p_pr = para._element.find(qn("w:pPr"))
    if p_pr is not None:
        lvl = p_pr.find(qn("w:outlineLvl"))
        if lvl is not None:
            val = lvl.get(qn("w:val"))
            if val is not None and val.isdigit():
                return min(int(val) + 1, 6)
    return None


def _ocr_docx_image(
    blip, doc, file_path: Path, images_dir: Path, seq: int, cache: dict
) -> str:
    """对 docx 里的一张图（a:blip 节点）落盘 + OCR，返回文本。

    cache 以 rId 为键：同一张图（比如每页都出现的 logo）只 OCR 一次，
    避免重复调用本地 OCR 和视觉模型。
    """
    from docx.oxml.ns import qn

    r_id = blip.get(qn("r:embed"))
    if not r_id:
        return ""
    if r_id in cache:
        return cache[r_id]

    try:
        part = doc.part.related_parts[r_id]
        blob = part.blob
    except KeyError:
        cache[r_id] = ""  # 引用失效的图片直接跳过
        return ""

    ext = _guess_image_ext(getattr(part, "content_type", ""), blob)
    images_dir.mkdir(parents=True, exist_ok=True)
    img_path = images_dir / f"{file_path.stem}_img_{seq}{ext}"
    img_path.write_bytes(blob)

    try:
        text = pytesseract.image_to_string(Image.open(img_path), lang="chi_sim+eng")
    except Exception as e:
        print(f"OCR 识别图片 {img_path.name} 失败: {e}")
        text = ""  # 本地 OCR 失败不跳过：留给视觉模型补救
    # 视觉模型增强（混合方案）:OCR 为空时补 Qwen3-VL,见 rag/vision_ocr.py
    text = hybrid_image_text(blob, text, source=str(img_path))
    if text.strip():
        print(f"OCR 识别图片 {img_path.name} 的文字: {text.strip()}")

    cache[r_id] = text.strip()
    return cache[r_id]


def _docx_table_windows(table) -> list[str]:
    """docx 表格 → Markdown 窗口，**每个窗口都重复表头**。

    原来 docx 表格是直接塞进正文、长度不设限的，会被切分器拦腰截断，
    切出没有表头的碎片 —— 同一个坑 Excel 用 XLSX_MAX_CHARS + 重复表头躲开了，
    Word 这边一直没躲。这里复用 structure.window_table_lines 的同一套逻辑。
    """
    rows: list[str] = []
    for row in table.rows:
        cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
        line = "| " + " | ".join(cells) + " |"
        if line.strip("| ").strip():  # 丢掉全空行
            rows.append(line)

    if not rows:
        return []

    col_count = len(table.rows[0].cells)
    if col_count == 0:
        return []

    head_md = rows[0] + "\n" + "|" + " --- |" * col_count
    return [piece for piece, _, _ in st.window_table_lines(head_md, rows[1:])]


def _load_docx(file_path: str, outputimages: str = "docx_images") -> list[Section]:
    """Word → 结构单元（单趟遍历，保住段落↔表格↔图片的原文顺序）。

    原实现有三个会伤到长文档的问题：
    1. **分三趟读**：先收所有段落、再收所有表格、最后把图片 OCR 追加到文末。
       结果 50 张表和所有图片文字全挤在文档末尾，跟所属章节隔了几万字 ——
       检索到表格时根本不知道它属于哪一节。
    2. **丢标题层级**：para.style.name（Heading 1/2/3）从来没被读过，
       长文档的目录结构在入库前就没了，后面再怎么切都补不回来。
    3. **表格不设防**：表被当普通正文交给切分器，长表会被切碎且丢表头。

    改成单趟遍历 doc.element.body，按 XML 里的真实顺序处理 w:p / w:tbl，
    读段落样式拿标题层级，表格走"加窗 + 重复表头"，图片 OCR 插回原段落位置。
    """
    from docx import Document as DocxDocument
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    path = Path(file_path)
    doc = DocxDocument(path)

    images_dir = IMAGES_DIR / outputimages
    sections: list[Section] = []
    heading_path: list[str] = []
    buf: list[str] = []
    order = 0
    img_cache: dict[str, str] = {}
    img_seq = 0

    def flush_text() -> None:
        nonlocal buf, order
        body = "\n".join(buf).strip()
        buf = []
        if body:
            sections.append(
                Section(text=body, section_path=list(heading_path), order=order)
            )
            order += 1

    for child in doc.element.body.iterchildren():
        # ── 段落 ──
        if child.tag == qn("w:p"):
            para = Paragraph(child, doc)
            text = para.text.strip()
            level = _docx_heading_level(para)

            if level is not None and text:
                # 标题：收掉上一节，再更新路径（path[:level-1] 实现层级回退）
                flush_text()
                heading_path[:] = heading_path[: level - 1] + [text]
                continue

            if text:
                buf.append(text)

            blips = child.findall(".//" + qn("a:blip"))
            if blips:
                # 先把累积的正文收掉，让图片紧跟在它所属的那段文字之后，
                # 而不是被堆到文末
                flush_text()
                for blip in blips:
                    img_seq += 1
                    ocr = _ocr_docx_image(
                        blip, doc, path, images_dir, img_seq, img_cache
                    )
                    if ocr:
                        sections.append(
                            Section(
                                text=f"【图片 {img_seq}】\n{ocr}",
                                section_path=list(heading_path),
                                order=order,
                                atomic=True,
                                kind="image",
                            )
                        )
                        order += 1
            continue

        # ── 表格：独立成 atomic 单元，与它所属章节保持位置关系 ──
        if child.tag == qn("w:tbl"):
            flush_text()
            table = Table(child, doc)
            for piece in _docx_table_windows(table):
                sections.append(
                    Section(
                        text=piece,
                        section_path=list(heading_path),
                        order=order,
                        atomic=True,
                        kind="table",
                    )
                )
                order += 1
            continue

    flush_text()
    return sections


def _guess_image_ext(content_type: str, blob: bytes) -> str:
    """根据 content_type 推断图片后缀，推断不了再用文件头魔数兜底"""
    mapping = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "image/tiff": ".tiff",
        "image/bmp": ".bmp",
    }
    ext = mapping.get((content_type or "").strip().lower())
    if ext:
        return ext
    # 文件头魔数兜底
    if blob[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if blob[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if blob[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    if blob[:4] == b"RIFF" and blob[8:12] == b"WEBP":
        return ".webp"
    if blob[:4] in (b"II*\x00", b"MM\x00*"):
        return ".tiff"
    if blob[:2] == b"BM":  # BMP 魔数只有 2 字节(修:原写法 blob[:4] == b"BM" 永远为假)
        return ".bmp"
    return ".png"  # 无法识别时默认 png


# ==================== Excel 加载 ============================
# 单块字符预算。Excel 窗口现在是 atomic 单元、不会被二次切分，
# 所以这个值不再受旧 CHUNK_SIZE=1000 的约束；保留 600 是因为
# 表格行短而密，窗口再大反而稀释单块的语义
XLSX_MAX_CHARS = 600
# 单块最多数据行（防止极窄表一个窗口塞进几十行、语义被稀释）
XLSX_MAX_ROWS = 50
# 单文件最多入库数据行（上传接口没有大小限制，防止一个十万行的表把索引打爆）
XLSX_MAX_TOTAL_ROWS = 5000


def _cell_text(value) -> str:
    """单元格值 → 检索友好的文本。

    - 2999.0 → "2999"：Excel 里数字都是浮点，留着小尾巴会让"2999 元"这种查询匹配不上
    - 日期/时间按中文习惯格式化，而不是 datetime(2026, 8, 1, 0, 0)
    - 单元格里的换行折叠成空格，避免破坏 Markdown 表格的行结构
    """
    from datetime import date, datetime, time, timedelta

    if value is None:
        return ""
    # bool 必须放在 date/数值之前判断（bool 是 int 的子类）
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, datetime):
        fmt = (
            "%Y-%m-%d %H:%M:%S"
            if (value.hour or value.minute or value.second)
            else "%Y-%m-%d"
        )
        return value.strftime(fmt)
    if isinstance(value, date):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, time):
        return value.strftime("%H:%M:%S")
    if isinstance(value, timedelta):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else f"{value:g}"
    return " ".join(str(value).split())


def _sheet_to_windows(sheet) -> list[tuple[str, int, int]]:
    """把一张工作表转成若干「表头 + 数据行」的 Markdown 窗口。

    返回 [(markdown 文本, 起始行, 结束行)]，行号用 Excel 里的真实行号（表头是第 1 行）。

    两个关键设计：
    1. **每个窗口都重复表头**——块是逐块入库、逐块被检索出来的，数据行脱离表头就
       没法理解（"2999 | 24个月" 到底是价格还是库存？）
    2. **按字符预算切窗**——保证每个窗口都是一个 atomic 单元，后续不会被二次切分，
       表结构也就不会在中途断掉

    切窗逻辑与 docx 表格共用 structure.window_table_lines，避免两份实现各自漂移。
    """
    rows: list[tuple[int, list[str]]] = []
    for row_no, raw in enumerate(sheet.iter_rows(values_only=True), 1):
        cells = [_cell_text(v) for v in raw]
        while cells and cells[-1] == "":  # 去掉行尾空列
            cells.pop()
        if not cells or all(c == "" for c in cells):  # 整行空 → 跳过
            continue
        # 记住真实行号：空行会被跳过，用完索引推行号会报错（试过，会少算）
        rows.append((row_no, cells))
        if len(rows) > XLSX_MAX_TOTAL_ROWS:
            print(
                f"[Excel] 工作表 {sheet.title} 超过 {XLSX_MAX_TOTAL_ROWS} 行，超出部分不入库"
            )
            break

    if not rows:
        return []

    _, header = rows[0]
    data = rows[1:]
    width = len(header)
    if width == 0 or not data:
        return []
    head_md = "| " + " | ".join(header) + " |\n" + "|" + " --- |" * width

    row_lines: list[str] = []
    row_numbers: list[int] = []
    for row_no, row in data:
        row_lines.append("| " + " | ".join((row + [""] * width)[:width]) + " |")
        row_numbers.append(row_no)

    # 共享切窗返回的下标是相对 row_lines 的，映射回真实 Excel 行号
    return [
        (text, row_numbers[start], row_numbers[end])
        for text, start, end in st.window_table_lines(
            head_md, row_lines, max_chars=XLSX_MAX_CHARS, max_rows=XLSX_MAX_ROWS
        )
        if 0 <= start < len(row_numbers) and 0 <= end < len(row_numbers)
    ]


def _load_xlsx(file_path: str) -> list[Section]:
    """Excel（.xlsx / .xlsm）→ 结构单元：每张工作表转 Markdown 表格后分窗。

    每个窗口是一个 atomic 单元 —— 父块 = 子块 = 那张带表头的完整表，
    永远不参与二次切分。

    section_path 用工作表名：既当面包屑（"某表.xlsx > Sheet1"），
    也让"只查某张表"的元数据过滤有据可依。

    图片、图表、批注不解析（Excel 里的图像内容需要另走 OCR/视觉模型，当前不支持）。
    """
    from openpyxl import load_workbook

    path = Path(file_path)
    # data_only=True：取公式的计算结果值，而不是 "=A1*B1" 这类公式串
    workbook = load_workbook(path, data_only=True, read_only=True)
    sections: list[Section] = []
    order = 0
    try:
        for sheet in workbook.worksheets:
            windows = _sheet_to_windows(sheet)
            print(f"[Excel] 工作表 {sheet.title} → {len(windows)} 个块")
            for text, start, end in windows:
                sections.append(
                    Section(
                        text=(
                            f"表格：{path.stem}"
                            f"（工作表：{sheet.title}，第 {start}-{end} 行）\n{text}"
                        ),
                        section_path=[str(sheet.title)],
                        order=order,
                        # Excel 没有页码概念，page_start/page_end 保持 None
                        atomic=True,
                        kind="table",
                        # 工作表名与真实行号区间透传进子块 metadata，
                        # 为后续"只查某张表"的元数据过滤留字段
                        extra={
                            "sheet": str(sheet.title),
                            "rows": f"{start}-{end}",
                        },
                    )
                )
                order += 1
    finally:
        workbook.close()  # read_only 模式必须显式关闭，否则文件句柄不释放
    return sections


def _load_file(file_path: str) -> list[Section]:
    """根据文件后缀加载单个文档，统一返回结构单元列表。"""
    file_path_obj = Path(file_path)
    suffix = file_path_obj.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(str(file_path))
    if suffix == ".txt" or suffix == ".md":
        return _load_txt(str(file_path))
    if suffix == ".docx":
        return _load_docx(str(file_path))
    if suffix == ".xlsx" or suffix == ".xlsm":
        return _load_xlsx(str(file_path))
    if suffix == ".xls":
        # openpyxl 读不了旧版二进制 .xls（要 xlrd），明确提示而不是静默返回空
        print(f"旧版 .xls 不支持，请另存为 .xlsx 后重新上传: {file_path}")
        return []
    return []


# def _load_documents(directory: Path) -> List[Document]:
#     """加载目录下所有支持的文档"""
#     all_docs: List[Document] = []
#     if not directory.exists():
#         return all_docs
#     # sorted 保证读取文件顺序一致
#     for file_path in sorted(directory.iterdir()):
#         # extend 将列表中的元素逐一添加到 all_docs 中，而不是将整个列表作为一个元素添加
#         all_docs.extend(_load_file(file_path))
#     return all_docs

# def get_chunk_id(chunk: Document) -> str:
#     """基于文档内容生成唯一 ID（SHA-256）"""
#     content = chunk.page_content
#     # 可以加入元数据如文件名来避免跨文件相同内容被误判，但为了彻底去重，仅内容足矣
#     return hashlib.sha256(content.encode('utf-8')).hexdigest()


def _save_parents(source: str, parents: list[dict]) -> int:
    """写父块到 MySQL：按文件整体替换（先删旧再插新）。

    parents 是 _split_document 产出的 dict 列表，顺带把 breadcrumb / page_start /
    page_end 一起写进去 —— parents 表这三列早就存在，只是一直没人写。

    **MySQL 不可用时只告警、不抛错** —— 子块照常入库，检索侧自动降级为
    纯子块。父块库的可用性绝不能决定"能不能回答问题"。
    """
    rows = [
        (p["pid"], p["text"], p["breadcrumb"], p["page_start"], p["page_end"])
        for p in parents
    ]
    try:
        count = parent_store.replace(source, st.doc_id_of(source), rows)
        print(f"[父块] {source} → MySQL {count} 条")
        return count
    except MySQLUnavailable as e:
        print(f"[父块] MySQL 不可用，跳过父块写入（检索将降级为纯子块）: {e}")
        return 0


def _record_document(
    source: str,
    chunk_count: int,
    parent_count: int,
    error: str | None = None,
) -> None:
    """写 documents 表。

    这是记录性数据（用于重建一致性校验和切分参数版本管理），
    不是回答问题的必需数据，所以失败**不能中断入库流程**。
    """
    doc_id = st.doc_id_of(source)
    filename = Path(source).name
    try:
        store = DocumentStore()
        if error:
            store.mark_failed(doc_id, filename, source, error, CHUNK_SCHEMA_VER)
        else:
            store.mark_ok(
                doc_id, filename, source, chunk_count, parent_count, CHUNK_SCHEMA_VER
            )
    except MySQLUnavailable as e:
        print(f"[文档元数据] 写入失败（不影响检索）: {e}")


def _split_document(
    source: str, sections: list[Section]
) -> tuple[list[dict], list[tuple[str, str, dict]]]:
    """结构单元 → (父块列表, 子块三元组)。

    纯函数：不碰 MySQL、不碰 Chroma，所以能脱离服务单测。

    父块是 dict：{pid, text, breadcrumb, page_start, page_end, section_path, atomic}
    子块三元组是 (child_id, 子块正文, Chroma metadata)。

    ## 两条关键规则（v3 改造时曾丢掉，实测老 chunk 有、新 chunk 没有）

    1. **atomic section（表格/图片/扫描页）自己就是一个父块**，不再拼成整篇文本后
       二次切分。原实现把 sections 拍平成一段文本再切，结果是：① atomic 标记作废，
       一张超过 TABLE_WINDOW_CHARS 的表会被切开、子块失去表头；② 3 个工作表的
       xlsx 被合并成同一个父块，跨表内容混在一起。
    2. **父块带上 breadcrumb / page_start / atomic 并传给子块**：生成侧要靠它把
       [docN] 标注回"哪个文件、哪一节、第几页"，否则引用无法核验。
    """
    doc_id = st.doc_id_of(source)
    parents: list[dict] = []

    for sec in sections:
        # atomic 单元整块成父块；普通文本才按 target 切
        texts = [sec.text] if sec.atomic else st.split_parents(sec.text)
        for text in texts:
            if not text.strip():
                continue
            order_idx = len(parents)
            path = [p for p in (sec.section_path or []) if p]
            parents.append(
                {
                    "pid": st.parent_id_of(source, order_idx),
                    "text": text,
                    "order_idx": order_idx,
                    # 面包屑已经含文件名（"手册.pdf > 第3章 > 3.2"），比裸路径可读
                    "breadcrumb": " > ".join(path) or None,
                    "page_start": sec.page_start,
                    "page_end": sec.page_end,
                    "section_path": path,
                    "atomic": bool(sec.atomic),
                }
            )

    children: list[tuple[str, str, dict]] = []
    for parent in parents:
        pid = parent["pid"]
        for cidx, ctext in enumerate(st.split_children(parent["text"])):
            cid = st.child_id_of(pid, cidx)
            meta = {
                # source 是给生成侧标注引用来源用的
                "source": source,
                "parent_id": pid,  # ← 回溯父块的关键
                "child_id": cid,
                "doc_id": doc_id,
                "order_idx": parent["order_idx"],
                # ↓ 新增：引用可核验 + 让检索侧知道这块是表格/图片
                "atomic": parent["atomic"],
                "chunk_schema_ver": CHUNK_SCHEMA_VER,
            }
            # Chroma 拒绝 None：数值/文本类字段取不到就**不写这个键**，别写 0 假装有值
            if parent["breadcrumb"]:
                meta["breadcrumb"] = parent["breadcrumb"]
            if parent["section_path"]:
                meta["section_path"] = " > ".join(parent["section_path"])
            if parent["page_start"] is not None:
                meta["page_start"] = int(parent["page_start"])
            children.append((cid, ctext, meta))

    return parents, children


def init_rag(file_path: str) -> int:
    """单个文档入库的**对外入口**：任何一步失败都留痕，然后把异常上抛。

    为什么要包这一层：GBK 编码的 txt、损坏的 PDF 会在 `_load_file` 就抛异常，
    那时 documents 表还一个字都没写过。实测结果是——接口返回 500，但库里查不到
    "这个文件曾经来过"（38 行里只有空文件那条 failed）。失败必须留痕，否则运维
    只能看到一次 500，查不出是哪个文件没进库。
    """
    try:
        return _init_rag(file_path)
    except Exception as e:  # noqa: BLE001
        print(f"[入库] {file_path} 失败: {e!r}")
        _record_document(file_path, 0, 0, error=f"入库失败: {type(e).__name__}: {e}")
        raise


def _init_rag(file_path: str) -> int:
    """（正文）解析 → 两级切分 → 父块写 MySQL、子块写 Chroma。

    ## 写入顺序

    遵循"**父块的存活区间必须覆盖子块的存活区间**"：
      ① 先写父块（MySQL 单事务，原子）
      ② 再写子块（Chroma）：**先 add 新块，再删本次没写进去的旧块**
    中途失败只会留下孤儿父块（无害），不会出现"子块指向不存在的父块"。
    反过来的话，检索就会静默降级 —— 不报错，只是答案变差。

    ② 内部为什么不能"先删后加"：add 失败（embedding 额度/网络抖动，正是外面
    那层重试要防的场景）时旧块已经删光，而 _record_document 只在成功路径执行，
    这份文档就变成"检索不到、也没有失败记录"的孤儿。先加后清最坏只是多留旧块。

    返回：入库的子块数。
    """
    sections = _load_file(file_path)  # 只加载新上传的文件
    if not sections:
        print(f"文件 {file_path} 未加载到任何文档")
        _record_document(file_path, 0, 0, error="未解析出任何内容")
        return 0
    # 切分 大块和小块
    parents, children = _split_document(file_path, sections)
    print(f"[切分] {file_path}: {len(parents)} 个父块 / {len(children)} 个子块")

    # ① 父块 → MySQL（先父后子）
    _save_parents(file_path, parents)

    # ② 子块 → Chroma
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v2", dashscope_api_key=os.getenv("QIANWEN_API_KEY")
    )

    store = Chroma(
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR),
        collection_name="knowledge_base",
    )

    # 按文件来源去重：同一个文件重新上传时要覆盖旧块。
    # 顺序是"**先加后清**"，不能反过来（原来是先 delete(where=source) 再 add）：
    #   add_texts 失败时旧块已经被删、新父块已经写进 MySQL，而 _record_document
    #   只在成功路径执行 → 文档静默变成检索不到的孤儿（有 500，但库里的记录还是 ok）。
    # ids 用的是确定性 child_id，所以重复 add 天然幂等，先加是安全的。
    new_ids = [c[0] for c in children]
    try:
        # 向量化 + 持久化入库
        if children:
            store.add_texts(
                texts=[c[1] for c in children],
                metadatas=[c[2] for c in children],
                ids=new_ids,
            )
            print(f"文件 {file_path} 入库 {len(children)} 个子块")
        else:
            print(f"文件 {file_path} 未切分出任何文档块")

        # 清掉这个文件残留的旧块（重建后块数变少时才会真的有东西要删）。
        # children 为空时不动旧数据：宁可留着旧版本，也不要把文档清空。
        if children:
            existing = set(store.get(where={"source": file_path}).get("ids") or [])
            stale = existing - set(new_ids)
            if stale:
                store.delete(ids=sorted(stale))
                print(f"文件 {file_path} 清理旧块 {len(stale)} 个")
    except Exception as e:
        # 留痕交给外层 init_rag 统一做（解析阶段失败也要留痕，不能只有这里记）
        print(f"子块入库失败: {e!r}")
        raise

    # ③ 文档元数据（记录性数据，失败不影响检索）
    _record_document(file_path, len(children), len(parents))

    return len(children)


# ==================== 多路召回（dense + BM25 → RRF 融合）====================


def _clean_query(text: str) -> str:
    """去掉中文标点和空白，保证 BM25 分词对齐（query 与语料统一清洗）"""
    return _PUNCT_RE.sub("", text or "")


def _ensure_ready() -> None:
    """懒连接持久化的 Chroma 知识库（不调用 init_rag，纯读取侧初始化）。

    - 幂等 + 线程安全：并发首次调用只会初始化一次
    - 上传走 init_rag 写库，这里每次检索都从持久化目录读最新状态
    """
    global _vectorstore, _retriever, _is_initialized
    if _vectorstore is not None:
        return
    # 获取锁
    with _init_lock:
        if _vectorstore is not None:
            return
        try:
            embeddings = DashScopeEmbeddings(
                model="text-embedding-v2",
                dashscope_api_key=os.getenv("QIANWEN_API_KEY"),
            )
            store = Chroma(
                embedding_function=embeddings,
                persist_directory=str(PERSIST_DIR),
                collection_name="knowledge_base",
            )
            _vectorstore = store
            # 数据库转为检索器
            _retriever = store.as_retriever(search_kwargs={"k": TOP_K})
            _is_initialized = True
            print("知识库连接成功（读取侧懒初始化）")
        except Exception as e:
            print(f"知识库连接失败: {e}")
            _vectorstore = None
            _retriever = None
            _is_initialized = False


def _build_bm25_index() -> None:
    """从 Chroma 全量拉取 chunk，构建 BM25 内存索引（幂等，线程安全）"""
    global _bm25, _bm25_corpus, _chunk_docs
    if _vectorstore is None:
        return
    with _bm25_lock:
        if _vectorstore is None:
            return
        try:
            # 拿到向量数据库中的文本和元数据（页码，来源）等
            data = _vectorstore._collection.get(include=["documents", "metadatas"])
            texts = data.get("documents") or []
            metas = data.get("metadatas") or []
            # print(f"texts:",texts)
            # print(f"metas:",metas)
            if not texts:
                _bm25, _bm25_corpus, _chunk_docs = None, [], []
                return
            _chunk_docs = [
                # Document将“文本”和“元数据”打包成一个对象。
                Document(page_content=t, metadata=m or {})
                for t, m in zip(texts, metas)
            ]
            # print(f"chunk_docs:::::",_chunk_docs)
            # jieba.lcut 将中文分词然后变成列表
            # 这里建立索引实际上是对之前存入到向量库中的切片chunk进行的
            _bm25_corpus = [jieba.lcut(_clean_query(t)) for t in texts]

            # print(f"_bm25_corpus：：",_bm25_corpus)
            _bm25 = BM25Okapi(_bm25_corpus)
            print("bm25222", _bm25)
            print(f"BM25 索引构建完成，共 {len(texts)} 个 chunk")
        except Exception as e:
            print(f"BM25 索引构建失败，降级为纯向量检索: {e}")
            _bm25, _bm25_corpus, _chunk_docs = None, [], []


def _sparse_search(query: str, k: int) -> list[Document]:
    """BM25 关键词路：jieba 分词 → 取分数最高的 k 个 chunk（0 分视为未命中）。

    BM25 是内存索引，上传新文档后通过 chunk 数量变化自动触发重建。
    """
    global _bm25, _chunk_docs
    if _vectorstore is None:
        return []
    try:
        # _chunk_docs   分词语料，用来和用户的问题分词之后进行对比
        # 文档有更新或者第一次拿到锁的用户才会触发建立索引
        print("bm25111", _bm25)
        if _bm25 is None or len(_chunk_docs) != _vectorstore._collection.count():
            _build_bm25_index()
            print("bm25333", _bm25)

    except Exception as e:
        print(f"BM25 索引一致性检查失败: {e}")
    if _bm25 is None or not _chunk_docs:
        return []

    # 将用户问题分词然后对每个文档进行打分，这里清洗用户问题要和建立索引时一致
    scores = _bm25.get_scores(jieba.lcut(_clean_query(query)))

    print("分数列表：scores", scores)

    # sorted从小到大排序，这里取负从大到小并取k个
    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]

    print("取前k个切片对应的分数从大到小：", ranked)
    # 根据切片分筛选   拿到命中的切片
    return [_chunk_docs[i] for i in ranked if scores[i] > 0]


def _doc_key(doc: Document) -> str:
    """给一块算一个稳定身份。

    优先用入库时生成的确定性 child_id，**而不是 page_content**：
    两个不同块的正文完全可能一模一样（Excel/表格窗口刻意重复表头，块间天然
    有大量重复文本），拿正文当 key 会让它们的 metadata 互相覆盖。
    没有 child_id 的是旧数据，退回正文（保持兼容）。
    """
    child_id = doc.metadata.get("child_id")
    return str(child_id) if child_id else doc.page_content


# RRF 的 k —— **从 60 调到 10**，依据是实测消融（eval/retrieval_recall.py）。
#
# k 越小，名次差异保留得越多：
#   k=60：rank1 = 1/61 = 0.0164，rank40 = 1/100 = 0.0100 → 只差 1.6 倍；
#         而"两路都命中"直接把分数翻倍 → 20 个"两路都出现但都不对"的块，
#         能把某一路的**第 1 名**挤出 top_n。
#   k=10：rank1 = 0.0909，rank40 = 0.0200 → 差 4.5 倍，单路第 1 不会被浅候选池淹没。
#
# 实测证据（488 块的语料库，5 条未命中里有 3 条是这个原因）：
#   一次解决率是什么意思？   dense=None  bm25=1  → 融合后 rrf=None
#   装锁师傅多久能到我家？   dense=11    bm25=None → 融合后 rrf=None
#   锁体安装孔距是多少毫米？ dense=None  bm25=11 → 融合后 rrf=None
RRF_K = 10


def _rrf_fuse(
    dense_docs: list[Document],
    sparse_docs: list[Document],
    k: int = RRF_K,
    top_n: int = TOP_K,
) -> list[Document]:
    """RRF 融合：score(d) = Σ 1/(k + rank)。按块身份去重，天然处理两路命中同一块。"""
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}
    for docs in (dense_docs, sparse_docs):
        for rank, doc in enumerate(docs, 1):
            key = _doc_key(doc)
            doc_map.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_n]
    return [doc_map[key] for key, _ in ranked]


def _format_docs(docs: list[Document]) -> str:
    """把检索结果格式化为生成侧可读的上下文（带来源注脚）

    注脚升级：带上章节面包屑和页码。原来只有 source 路径，
    生成侧给不出可核验的出处，用户也没法回查原文位置。
    """
    parts = []
    for i, doc in enumerate(docs, 1):
        # 面包屑已经含文件名（"手册.pdf > 3.2 退换货政策"），比裸路径更可读
        location = str(
            doc.metadata.get("breadcrumb") or doc.metadata.get("source", "未知来源")
        )
        page = doc.metadata.get("page_start")
        if page:
            location = f"{location} (p.{page})"
        parts.append(f"[文档片段 {i} — 来源: {location}]\n{doc.page_content}")
    return "\n\n".join(parts)


# 检索
async def retrieve(query: str, k: int = TOP_K) -> list[Document]:
    """
    多路召回：dense（向量）+ sparse（BM25）两路异步并行检索，RRF 融合后返回。

    - 两路各自失败时降级为另一路（gather return_exceptions）
    - 不调用 init_rag（上传/入库由 init_rag 负责，这里只读持久化库）
    - 调用方必须是 async 环境；非 async 场景用 retrieve_sync

    参数:
        query: 查询文本（通常就是用户的问题）
        k: 返回的文档片段数

    返回:
        RRF 融合后的原始 Document 列表（未命中为空列表）。

    异常:
        KnowledgeBaseError: 知识库连接失败 / 未初始化 / 为空。
    """
    global _vectorstore

    if _vectorstore is None:
        # 初始化数据库并转为检索器
        _ensure_ready()
    if _vectorstore is None:
        raise KnowledgeBaseError("知识库连接失败，请检查后重试。")
    if _vectorstore._collection.count() == 0:
        raise KnowledgeBaseError("知识库为空，请先上传文档到知识库。")

    # 融合前每路多取一些，融合后再砍到 k
    # 粗筛，先筛选相近语义数量较多 然后在tpk
    fetch_k = max(k * 2, 6)

    # 两路召回：异步并行执行
    # 语义相似度搜索
    # 任务对象（Task） 下面这两条都没有运行只是创建对象
    dense_task = asyncio.to_thread(_vectorstore.similarity_search, query, fetch_k)
    # bm25检索
    # 返回命中的原文档数据_chunk_docs
    sparse_task = asyncio.to_thread(_sparse_search, query, fetch_k)
    # 开启两个线程运行
    # 因为dense_task，sparse_task 这两个在上面是创建任务然后等下面await gather两个一起运行完
    # 并发执行器gather
    # gather  如果传进去的是协程对象 则会调用create_task包装成协程任务  如果传进去的是任务 就相当于await 则直接返回该任务
    # gather  和 create_task会创建任务（并将协程对象放入到事件循环中等待await执行）
    # 如果不考虑并发以及一些情况，gather 和await差不多
    # 协程对象就是你调用一个 async def 函数时，返回的那个东西
    # create_task 只能接收协程对象（接收任务会报错，gather是两个都可以）创建task 放入事件循环 等待执行
    dense_docs, sparse_docs = await asyncio.gather(
        dense_task, sparse_task, return_exceptions=True
    )
    # print(f"dddd",dense_docs)
    # print(f"sp：：：",sparse_docs)

    # isinstance判断 这个数据是否是抛出的异常数据
    dense_ok = not isinstance(dense_docs, BaseException)
    sparse_ok = not isinstance(sparse_docs, BaseException)
    if not dense_ok:
        print(f"向量检索失败，降级为仅关键词检索: {dense_docs}")
        dense_docs = []
    if not sparse_ok:
        print(f"BM25 检索失败，降级为仅向量检索: {sparse_docs}")
        sparse_docs = []

    docs = _rrf_fuse(dense_docs, sparse_docs, top_n=k)
    if not docs:
        return []
    print("RRF融合成功：", docs)
    return docs


# 开启事件循环让里面的任务可以调度
def retrieve_sync(query: str, k: int = TOP_K) -> list[Document]:
    """同步版 retrieve：供 LangChain 工具等无事件循环的线程调用（agent 工具场景）。"""
    return asyncio.run(retrieve(query, k))


def _consistency_report(chroma_count: int) -> dict:
    """对比两个存储，找出"不一致"和"切分参数过期"。

    Chroma（子块）与 MySQL（父块）是两次独立写入、没有跨库事务，
    所以需要一个地方能**发现**不一致，而不是让它静默劣化 ——
    这正是 documents 表存在的首要理由。
    """
    report: dict[str, object] = {
        "parent_count": None,
        # ⚠️ 不能叫 document_count —— 那个键在本模块的语义是"文档块数量"(chunk 数)，
        # 复用它会把 chunk 数覆盖成文件数，静默改掉 get_status 调用方的行为
        "file_count": None,
        "stale_documents": [],
        "mysql_error": None,
    }
    try:
        report["parent_count"] = parent_store.count()
        docs = DocumentStore().all()
        report["file_count"] = len(docs)
        # 切分参数版本对不上的文档 → 改了 CHUNK_SCHEMA_VER 后按这个列表重灌即可
        report["stale_documents"] = [
            d.filename for d in docs if d.chunk_schema_ver != CHUNK_SCHEMA_VER
        ]
    except MySQLUnavailable as e:
        report["mysql_error"] = str(e)
    return report


def get_status() -> dict:
    """
    返回知识库当前状态：是否已初始化、文档块数量、存储目录。

    供 Agent 工具（get_knowledge_base_status）和意图门控（status_query）使用。

    新增字段（只增不改，调用方不受影响）：
      parent_count       MySQL 里的父块数
      file_count         documents 表里登记的已入库文件数
      stale_documents    切分参数版本过期的文件（需要重新解析）
      mysql_error        MySQL 不可用时的原因（None 表示正常）
    """
    try:
        store = Chroma(
            embedding_function=DashScopeEmbeddings(
                model="text-embedding-v2",
                dashscope_api_key=os.getenv("QIANWEN_API_KEY"),
            ),
            persist_directory=str(PERSIST_DIR),
            collection_name="knowledge_base",
        )
        print("知识库状态embedding模型")
        count = store._collection.count()
    except Exception as e:
        print(f"获取知识库状态失败: {e}")
        return {
            "initialized": False,
            "document_count": 0,
            "knowledge_base_dir": str(KNOWLEDGE_BASE_DIR),
            "error": str(e),
            "parent_count": None,
            "file_count": None,
            "stale_documents": [],
            "mysql_error": None,
        }
    return {
        "initialized": True,
        "document_count": count,
        "knowledge_base_dir": str(KNOWLEDGE_BASE_DIR),
        # 父块库不可用不影响检索（会自动降级），所以状态里只如实标注，不算失败
        **_consistency_report(count),
    }


# 重排序模型（直接调 DashScope text-rerank API，对齐官方 curl，绕开旧集成类硬编码模型/吞错误的问题）
class _DashScopeReranker:
    """薄封装：把官方 rerank curl 包成 compress_documents 接口，reordering() 调用处不变。
    - 按 index 回原列表取文档，保留 source/page 元数据
    """

    def __init__(self, model: str, top_n: int, api_key: str | None):
        self.model = model
        self.top_n = top_n
        self.api_key = api_key

    def compress_documents(
        self, documents: list[Document], query: str
    ) -> list[Document]:
        import dashscope

        resp = dashscope.TextReRank.call(
            model=self.model,
            query=query,
            documents=[d.page_content for d in documents],
            top_n=self.top_n,
            return_documents=True,  # 对齐官方 curl
            api_key=self.api_key,
        )
        # 官方 API：status_code==200 成功；失败时真实原因在 code/message 里
        if resp.status_code != 200:
            print(f"重排 API 失败: code={resp.code}, message={resp.message}")
            raise RuntimeError(f"DashScope rerank 失败: {resp.code} {resp.message}")

        # results 已按相关性从高到低排序，含 index；按 index 回原列表取文档
        return [documents[r["index"]] for r in resp.output.results]


# def get_reordering():
#     """返回重排器实例（模型默认 qwen3-rerank，可用 RERANK_MODEL 覆盖）。"""
#     reranker = _DashScopeReranker(
#         model="gte-rerank-v2", top_n=TOP_N, api_key=os.getenv("QWEN_RERANK_API_KEY")
#     )
#     print("重排模型准备：", reranker)
#     return reranker


# # 重排序
# def reordering(query: str, docs: list[Document]) -> list[Document]:
#     if not docs:
#         return docs
#     # 按相关性从高到低排好序的、新的文档列表  compress_documents文档压缩器 并返回topn条文档
#     try:
#         reranker = get_reordering().compress_documents(documents=docs, query=query)
#         print("重排序com之后：", reranker)

#         # 回填元数据（页码 注脚等）
#         orig_map = {d.page_content: d for d in docs}
#         return [orig_map.get(d.page_content, d) for d in reranker][:TOP_N]
#     except Exception as e:
#         print(f"重排失败，降级为 RRF 默认顺序: {e}")
#         return docs


def _expand_to_parents(children: list[Document]) -> list[Document]:
    """把重排后的子块展开成父块 —— small-to-big 的最后一步。

    **方向只能是"小子块 → 大父块"**：打分用小块（相关文字占比高，向量重心不被
    稀释），送给 LLM 用大块（有章节标题、有表头，读得懂）。父块不参与任何相似度
    计算，它只是一本字典：拿子块的 parent_id 按主键取全文。

    流程：按 parent_id 去重（同一父块被 5 个子块命中只留 1 个）→ 查 MySQL →
    按预算截断。

    两条降级路径（都不会让检索返回空）：
    1. MySQL 不可用 → 原样返回子块
    2. 某个 parent_id 查不到 → 用那个命中的子块顶替
    """
    if not children:
        return children

    # 1) 按 parent_id 去重。children 已经是相关性顺序（RRF / 重排的输出），
    #    所以每个 parent 第一次出现的那个子块就是它分数最高的子块。
    best: dict[str, Document] = {}
    for doc in children:
        parent_id = doc.metadata.get("parent_id")
        if parent_id:
            # 按 parent_id 去重，保留第一个出现的子块（这里是重排序已经排好了分数）
            # setdefault 如果这个 parent_id 还没出现过 → 把当前 doc 存进去
            # 如果这个 parent_id 已经出现过 → 什么都不做，保留先存进去的那个
            best.setdefault(str(parent_id), doc)

    if not best:
        return children

    pids = list(best)[:PARENT_TOP_N]

    # 2) 一次 IN 查询取回父块（连带 breadcrumb/page_start，供引用标注用）
    try:
        rows = parent_store.get(pids)
    except MySQLUnavailable as e:
        print(f"[父块] 取父块失败，降级为纯子块检索: {e}")
        return children

    # 3) 按预算截断。装不下的**跳过**，继续试后面的小父块；遇到就停会白丢内容
    expanded: list[Document] = []
    total = 0
    for pid in pids:
        row = rows.get(pid)
        child = best[pid]
        if row is None:
            # 父块缺失（重建中途失败过 / 被人手工删过）时用命中的子块顶上，
            # 绝不能让这一块从结果里静默消失
            print(f"[父块] {pid} 未找到，降级用命中的子块顶替")
            text = child.page_content
            meta = dict(child.metadata)
        else:
            text = row["text"]
            # 父块表里的出处更准（它带的是整节的面包屑），但要先把子块的元数据拷一份，
            # 避免下游改动污染原始 metadata 对象
            meta = dict(child.metadata)
            if row.get("breadcrumb"):
                meta["breadcrumb"] = row["breadcrumb"]
            if row.get("page_start") is not None:
                meta["page_start"] = row["page_start"]
        if total + len(text) > PARENT_MAX_CHARS:
            continue
        expanded.append(Document(page_content=text, metadata=meta))
        total += len(text)

    print(
        f"[父块] {len(children)} 个子块 → {len(pids)} 个父块候选 → "
        f"送出 {len(expanded)} 块（{total} 字）"
    )
    return expanded


# 调用本地重排序模型
def reordering(query: str, docs: list[Document]) -> list[Document]:
    """重排 + 父块聚合（small-to-big 的入口）。

    **顺序很关键：先重排子块，再展开成父块。** 反过来的话（先聚合成父块再重排）
    打分的对象就变成大块，回到"语义被稀释"的老问题，重排精度会明显下降。

    返回类型仍是 list[Document]，所以 agent/graph.py 和
    generatellm.generate_answer_within_budget 一行都不用改 —— 只是每个 Document
    从"1000 字的碎片"变成"1600 字的完整小节"。
    """
    if not docs:
        return docs

    if len(docs) == 1:
        # 只有一块时重排没有意义，但仍然要做父块展开：
        # 命中一个 450 字的碎片 ≠ 只能给模型 450 字，它所属的小节才是有用的上下文
        return _expand_to_parents(docs)

    try:
        # 懒加载：reranker 内部首次调用时才加载模型，后续直接复用
        result = reranker.rerank(query, docs, TOP_N)

        # 回填元数据（页码/面包屑注脚等）
        # 注意：要遍历 rerank 的返回值 result（重排后的文档列表），
        # 而不是 reranker 实例本身——后者不可迭代，会抛 TypeError 走降级分支
        # local_reranker 目前返回的就是原 Document 对象、metadata 天然保留，
        # 这里按 child_id 回填是为了防它将来改成重建对象
        orig_map = {_doc_key(d): d for d in docs}
        reranked = [orig_map.get(_doc_key(d), d) for d in result][:TOP_N]
    except Exception as e:
        # 打印完整堆栈，避免静默吞错后重排悄悄失效
        import traceback

        traceback.print_exc()
        print(f"重排失败，降级为 RRF 默认顺序: {e}")
        reranked = docs[:TOP_N]  # RRF 顺序本身就是按相关性排的

        # _expand_to_parents 按praent_id去重，查MySQL取父块全文，按预算截断
    return _expand_to_parents(reranked)
