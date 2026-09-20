"""企业级 RAG 测试语料生成器（可复现、自带 ground truth）。

用法：
    venv\\Scripts\\python.exe eval\\gen\\gen_corpus.py --profile S --seed 42

为什么用"生成"而不是"收集真实文档"：
  1. 数量可扩（S 档 20 个 → M/L 档 60/150 个），不需要人工找文档；
  2. **ground truth 零成本**：事实本身是常量，文档渲染和 golden 标注引用同一个
     字符串，所以 anchor 一定是库里原文的子串 —— 不会出现"锚点打错字 → 召回率
     假性归零"这种把标注问题误判成检索问题的情况；
  3. 可固定 seed、可 git diff：语料本身不入库（eval/corpus/ 已 gitignore），
     生成器 + golden 标注入库。

三条来自实测的硬约束（都验证过，别凭直觉改）：
  1. **PDF 必须逐行 insert_text**：用 TextWriter 整块写会让 pymupdf4llm 输出粘连
     文本（标题识别失效、段落黏成一行）。
  2. **PDF 里不要手写 `#`**：pymupdf4llm 自己会按字号推断并加 `#`，手写会得到 `# # 标题`。
  3. **表格样本要 >1200 字符**（structure.TABLE_WINDOW_CHARS）：否则整表被当成一个
     父块，测不出"atomic 表格被二次切分"的问题。

输出的三类文件：
  eval/corpus/          语料本体（gitignore）
  eval/corpus/manifest.json   每个文件：格式、用途、预期入库结果（哨兵用例靠它断言）
  eval/golden/queries.jsonl   golden（入 schema 与 eval/queries.jsonl 兼容，可被
                              retrieval_recall.py 直接读）
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # 脚本方式跑（sys.path[0] = eval/gen）与 -m 方式跑都要能用
    import filler as F
except ImportError:  # pragma: no cover
    from eval.gen import filler as F  # type: ignore[no-redef]

CJK_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simsun.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
)
PDF_FONT = "china-s"  # PyMuPDF 内置 CJK 字体
PAGE_W, PAGE_H = 595, 842  # A4 pt


# ════════════════════════════════════════════════════════════
# 一、事实（唯一真相源）
#    文档渲染和 golden 标注都引用这些常量 → anchor 天然正确。
# ════════════════════════════════════════════════════════════
F_L1_WARRANTY = "岚盾 L1 整机保修期为 24 个月，自签收之日起算。"
F_L1PRO_WARRANTY = "岚盾 L1 Pro 整机保修期为 36 个月，含免费上门服务。"
F_L2_WARRANTY = "岚盾 L2 整机保修期为 12 个月。"
F_L2PRO_WARRANTY = "岚盾 L2 Pro 整机保修期为 12 个月。"
F_L1_PRICE = "岚盾 L1 市场零售价为 1299 元。"
F_L1PRO_PRICE = "岚盾 L1 Pro 市场零售价为 1799 元。"
F_L2_PRICE = "岚盾 L2 市场零售价为 999 元。"
F_L2PRO_PRICE = "岚盾 L2 Pro 市场零售价为 1499 元，含一次免费上门安装。"

F_L1_BATTERY = "岚盾 L1 电池容量为 5000mAh。"
F_L1PRO_BATTERY = "岚盾 L1 Pro 电池容量为 8000mAh，支持 Type-C 应急供电。"
F_L2PRO_BATTERY = "岚盾 L2 Pro 电池容量为 10000mAh。"
F_TIANDIGOU = "支持天地钩的型号为 L1 Pro 与 L2 Pro；L1 与 L2 不支持天地钩。"
F_MOUNT_HOLE = "天地钩锁体安装孔距为 60mm。"
F_EMERGENCY_POWER = "门锁电量耗尽时，可用 Type-C 接口外接 5V/1A 电源应急供电。"

F_RETURN_V1 = "无理由退货期限为 7 天，自签收之日起算。"
F_RETURN_V2 = "无理由退货期限调整为 15 天，自 2025 年 3 月 1 日起生效。"
F_FREIGHT_V1 = "退货产生的运费由买家承担。"
F_FREIGHT_V2 = "因质量问题退货的，来回运费均由卖方承担。"
F_EXCHANGE_V2 = "质量问题换货期限为 30 天，需提供检测报告。"

F_INSTALL_SLA = "市区范围内 24 小时内上门响应，郊区 48 小时内响应。"
F_INSTALL_FEE = "上门安装服务费为 199 元；L1 Pro 与 L2 Pro 免上门费。"
F_REPAIR_L1 = "保外维修价目：L1 为 199 元，L1 Pro 为 299 元，L2 为 159 元，L2 Pro 为 259 元。"
F_EXTEND_PRICE = "可加购 12 个月延保服务，价格为 299 元。"
F_EXTEND_RULE = "延保服务需在整机保修期内购买，购买后总保修期在原基础上顺延 12 个月。"
F_SPARE_BATTERY = "备件价格：L1 电池 89 元，L1 Pro 电池 129 元。"
F_SPARE_LOCKBODY = "备件价格：标准锁体 199 元，天地钩锁体 259 元。"

F_ALIAS = "岚盾 L1 尊享版的正式型号为 L1 Pro，物料编码为 LD-L1P。"
F_GATEWAY_CODE = "网关配对码默认为 8888，可在 App 内修改。"
F_RESOLUTION = "一次解决率（resolution_rate）指客服首次接触即解决问题的比例。"
F_S3_WARRANTY = "云枢 S3 智能音箱整机保修期为 24 个月。"
F_S3_WEIGHT = "云枢 S3 智能音箱整机重量为 0.6kg。"
F_S3_FIRMWARE = "云枢 S3 固件升级免费，通过 App 推送完成。"


# ════════════════════════════════════════════════════════════
# 一·补、文件名常量
#    golden 与语料渲染共用同一批名字：预检脚本第一次跑就抓到过一次
#    "golden 里写的文件名和实际落盘名不一致"（锚点校验直接失败）。
# ════════════════════════════════════════════════════════════
DOC_PRODUCT_MANUAL = "岚盾L1系列_产品手册_v2.1.pdf"
DOC_INSTALL_GUIDE = "岚盾L1Pro_安装指南.pdf"
DOC_MANUAL_EN = "岚盾L2Pro_User_Manual_EN.pdf"
DOC_PART_COMPAT = "岚盾配件兼容表.pdf"
DOC_POLICY_V1 = "售后服务政策_v1_2024.docx"
DOC_POLICY_V2 = "售后服务政策_v2_2025.docx"
DOC_RETURN_RULES = "退换货操作细则.docx"
DOC_INSTALL_SERVICE = "上门安装服务规范.docx"
DOC_WARRANTY_EXT = "质保与延保说明.docx"
DOC_PARAM_TABLE = "岚盾产品参数对照表.xlsx"
DOC_SERVICE_STATS = "2025年售后服务统计.xlsx"
DOC_SPARE_STOCK = "备件库存与价格.xlsx"
DOC_TERMS = "产品术语与别名表.md"
DOC_FAQ = "常见问题FAQ.md"
DOC_SCRIPT = "客服话术模板.txt"
DOC_CHANGELOG = "产品变更记录.md"
DOC_S3_MANUAL = "云枢S3音箱_产品手册.pdf"
DOC_S3_PARAMS = "云枢S3音箱_参数表.xlsx"
DOC_GBK = "术语表_GBK.txt"
DOC_BROKEN = "损坏文件_截断.pdf"
DOC_EMPTY = "空文件.txt"
DOC_BADEXT = "不支持的格式.png"


# ════════════════════════════════════════════════════════════
# 二、渲染器
# ════════════════════════════════════════════════════════════
def _cjk_font_path() -> str | None:
    for p in CJK_FONT_CANDIDATES:
        if Path(p).exists():
            return p
    return None


def _wrap(text: str, font, size: float, width: float) -> list[str]:
    """按像素宽度折行 —— insert_text 不会自动换行，而一行太长会溢出页面。"""
    lines, cur = [], ""
    for ch in text:
        if ch == "\n":
            lines.append(cur)
            cur = ""
            continue
        if font.text_length(cur + ch, size) > width and cur:
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        lines.append(cur)
    return lines


class _PdfBuilder:
    """PDF 渲染：逐行 insert_text（约束 1），不手写 # （约束 2）。"""

    def __init__(self, header: str | None = None, footer: str | None = None):
        import pymupdf

        self.pymupdf = pymupdf
        self.doc = pymupdf.open()
        self.font = pymupdf.Font(PDF_FONT)
        self.header = header
        self.footer = footer

    def _decorate(self, page, page_no: int) -> None:
        # 每页都印同样的页眉/页码：真实企业文档就是这样，也正是"页眉去噪"要处理的对象
        if self.header:
            page.insert_text((50, 34), self.header, fontsize=8, fontname=PDF_FONT)
        if self.footer:
            page.insert_text(
                (50, PAGE_H - 30), f"{self.footer}  ·  第 {page_no} 页",
                fontsize=8, fontname=PDF_FONT,
            )

    def add_text_page(self, blocks: list[tuple[str, str]]) -> None:
        """blocks: [(style, text)]，style ∈ h1/h2/body/table"""
        page = self.doc.new_page(width=PAGE_W, height=PAGE_H)
        self._decorate(page, self.doc.page_count)
        y = 70.0
        for style, text in blocks:
            size = {"h1": 17.0, "h2": 13.5, "body": 10.5, "table": 9.5}[style]
            for line in _wrap(text, self.font, size, PAGE_W - 110):
                if y > PAGE_H - 70:
                    page = self.doc.new_page(width=PAGE_W, height=PAGE_H)
                    self._decorate(page, self.doc.page_count)
                    y = 70.0
                page.insert_text((55, y), line, fontsize=size, fontname=PDF_FONT)
                y += size * 1.75
            y += 6  # 段间距：pymupdf4llm 据此分段
        return None

    def add_scanned_page(self, text: str, title: str | None = None) -> None:
        """整页扫描件：只有像素、没有文本层（模拟纸质件扫描）。"""
        from PIL import Image, ImageDraw, ImageFont

        img = Image.new("RGB", (1240, 1754), "white")
        draw = ImageDraw.Draw(img)
        body = ImageFont.truetype(_cjk_font_path(), 30)
        big = ImageFont.truetype(_cjk_font_path(), 40)
        y = 120
        for i, line in enumerate(text.split("\n")):
            draw.text((80, y), line, fill="black", font=big if (i == 0 and title) else body)
            y += 68
        buf = io.BytesIO()
        img.save(buf, "PNG")
        page = self.doc.new_page(width=PAGE_W, height=PAGE_H)
        page.insert_image(page.rect, stream=buf.getvalue())
        self._decorate(page, self.doc.page_count)

    def save(self, path: Path) -> None:
        self.doc.save(path, garbage=3, deflate=True)
        self.doc.close()


def build_pdf(path: Path, pages: list[list[tuple[str, str]]], **kw) -> None:
    b = _PdfBuilder(**kw)
    for blocks in pages:
        b.add_text_page(blocks)
    b.save(path)


def build_pdf_with_scan(
    path: Path, pages: list[list[tuple[str, str]]], scan_text: str, **kw
) -> None:
    b = _PdfBuilder(**kw)
    for blocks in pages:
        b.add_text_page(blocks)
    b.add_scanned_page(scan_text, title="扫描件")
    b.save(path)


def md_table(rows: list[list[str]]) -> str:
    head = "| " + " | ".join(rows[0]) + " |"
    sep = "|" + " --- |" * len(rows[0])
    body = ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join([head, sep, *body])


def build_docx(
    path: Path,
    blocks: list[tuple],
    *,
    header: str | None = None,
) -> None:
    """blocks: ("h1"|"h2"|"p", text) | ("table", rows)"""
    import docx

    d = docx.Document()
    for block in blocks:
        kind, payload = block[0], block[1]
        if kind == "h1":
            d.add_heading(payload, level=1)
        elif kind == "h2":
            d.add_heading(payload, level=2)
        elif kind == "h3":
            d.add_heading(payload, level=3)
        elif kind == "p":
            d.add_paragraph(payload)
        elif kind == "table":
            rows = payload
            t = d.add_table(rows=len(rows), cols=len(rows[0]))
            t.style = "Table Grid"
            for i, row in enumerate(rows):
                for j, cell in enumerate(row):
                    t.cell(i, j).text = cell
        else:
            raise ValueError(kind)
    if header:
        d.sections[0].header.paragraphs[0].text = header
    d.save(path)


def build_xlsx(path: Path, sheets: list[dict]) -> None:
    """sheets: [{"name":.., "rows":[[..]], "merge":["A8:C8"], "width":[...]}]"""
    from openpyxl import Workbook

    wb = Workbook()
    wb.remove(wb.active)
    for spec in sheets:
        ws = wb.create_sheet(spec["name"])
        for row in spec["rows"]:
            ws.append(row)
        for rng in spec.get("merge", []):
            ws.merge_cells(rng)
        for i, w in enumerate(spec.get("width", []), 1):
            ws.column_dimensions[chr(64 + i)].width = w
    wb.save(path)


# ════════════════════════════════════════════════════════════
# 三、语料清单（S 档：17 个正式文档 + 3 个哨兵）
# ════════════════════════════════════════════════════════════
def build_corpus(
    out: Path,
    rng: random.Random,
    profile: str = "S",
    extra_golden: list[dict] | None = None,
) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    made: list[dict] = []

    def rec(path: Path, **kw) -> None:
        made.append(
            {
                "file": path.name,
                "format": path.suffix.lstrip("."),
                "size": path.stat().st_size,
                "sha1": hashlib.sha1(path.read_bytes()).hexdigest()[:12],
                **kw,
            }
        )

    # ---- 1. 产品手册（PDF，多页 + 跨页表格 + 页眉页脚）----
    p = out / "岚盾L1系列_产品手册_v2.1.pdf"
    build_pdf(
        p,
        [
            [
                ("h1", "第1章 产品概览"),
                ("body", "岚盾 L1 系列包含 L1 与 L1 Pro 两个型号，均支持指纹与密码开锁。"),
                ("h2", "1.1 型号与售价"),
                ("body", F_L1_PRICE),
                ("body", F_L1PRO_PRICE),
            ],
            [
                ("h1", "第2章 硬件参数"),
                ("body", F_L1_BATTERY),
                ("body", F_L1PRO_BATTERY),
                ("table", md_table([
                    ["型号", "电池容量", "锁体材质", "支持天地钩"],
                    ["L1", "5000mAh", "锌合金", "否"],
                    ["L1 Pro", "8000mAh", "锌合金", "是"],
                ])),
            ],
            [
                ("h1", "第3章 保修政策"),
                ("body", F_L1_WARRANTY),
                ("body", F_L1PRO_WARRANTY),
                ("table", md_table([
                    ["型号", "整机保修", "备注"],
                    ["L1", "24 个月", "标准保修"],
                    ["L1 Pro", "36 个月", "含免费上门服务"],
                ])),
            ],
            [
                ("body", F_TIANDIGOU),
                ("body", F_MOUNT_HOLE),
                ("body", "备注：保修期以发票日期为准，延保另计。"),
            ],
        ],
        header="岚盾智造 · 产品手册（内部资料，请勿外传）",
        footer="岚盾L1系列_产品手册_v2.1",
    )
    rec(p, purpose="多页 PDF：多级标题 + 跨页表格 + 每页页眉页脚（页眉去噪的靶子）", expect="ok")

    # ---- 2. 安装指南（PDF：文字页 + 整页扫描件）----
    p = out / "岚盾L1Pro_安装指南.pdf"
    build_pdf_with_scan(
        p,
        [
            [
                ("h1", "第1章 安装准备"),
                ("body", "安装前请确认门体厚度在 40mm 至 120mm 之间。"),
                ("body", F_MOUNT_HOLE),
                ("body", "L1 Pro 支持天地钩锁体，安装时需使用配套的加长螺丝。"),
            ],
            [
                ("h1", "第2章 应急处理"),
                ("body", "若门锁无法开锁，请先确认是否处于低电量状态。"),
                ("body", "详见随附的纸质《应急供电说明》（扫描件）。"),
            ],
        ],
        scan_text="应急供电说明\n" + F_EMERGENCY_POWER + "\n请勿使用超过 5V/1A 的电源适配器。",
        header="岚盾智造 · 安装指南",
    )
    rec(
        p,
        purpose="文字页 + 整页扫描件混合：扫描件的唯一事实只在这里（Type-C 应急供电）",
        expect="ok",
    )

    # ---- 3. 英文手册（PDF）----
    p = out / "岚盾L2Pro_User_Manual_EN.pdf"
    build_pdf(
        p,
        [
            [
                ("h1", "Chapter 1 Overview"),
                ("body", "The Landun L2 Pro is a smart door lock with fingerprint and face recognition."),
                ("body", "The warranty period of the L2 Pro is 12 months from the delivery date."),
            ],
            [
                ("h1", "Chapter 2 Specifications"),
                ("body", "Battery capacity: 10000mAh. Weight: 4.15kg. Support: Tiandigou lock body."),
                ("body", "Retail price: 1499 CNY, including one free on-site installation."),
            ],
        ],
        header="Landun Smart Lock · User Manual (EN)",
    )
    rec(p, purpose="英文 PDF：测中英混排与英文 query", expect="ok")

    # ---- 4. 配件兼容表（PDF，表格密集）----
    p = out / "岚盾配件兼容表.pdf"
    rows = [["配件", "适配型号", "价格", "备注"]]
    for i, (part, model, price, note) in enumerate(
        [
            ("标准锁体", "L1/L2", "199", "不含安装"),
            ("天地钩锁体", "L1 Pro/L2 Pro", "259", "含加长螺丝"),
            ("L1 电池", "L1", "89", "5000mAh"),
            ("L1 Pro 电池", "L1 Pro", "129", "8000mAh，支持 Type-C 应急"),
            ("指纹模块", "全系列", "349", "需返厂更换"),
            ("网关模块", "全系列", "299", "配对码默认 8888"),
        ],
        1,
    ):
        rows.append([part, model, price, note])
    build_pdf(
        p,
        [
            [("h1", "配件兼容与价格表"), ("table", md_table(rows))],
            [("h2", "说明"), ("body", F_SPARE_BATTERY), ("body", F_SPARE_LOCKBODY)],
        ],
        header="岚盾智造 · 配件表",
    )
    rec(p, purpose="表格密集 PDF：整表作为 atomic 单元", expect="ok")

    # ---- 5/6. 售后政策 v1 / v2（DOCX，版本冲突）----
    p = out / "售后服务政策_v1_2024.docx"
    build_docx(
        p,
        [
            ("h1", "售后服务政策（v1，2024 版）"),
            ("p", "本版本自 2024 年 1 月 1 日起执行，已被 v2 版本替代。"),
            ("h2", "1.1 无理由退货"),
            ("p", F_RETURN_V1),
            ("h2", "1.2 退货运费"),
            ("p", F_FREIGHT_V1),
        ],
        header="岚盾智造 · 售后政策（已作废版本）",
    )
    rec(p, purpose="旧版政策：与 v2 构成版本冲突，测'现行版本'能否被优先召回", expect="ok")

    p = out / "售后服务政策_v2_2025.docx"
    build_docx(
        p,
        [
            ("h1", "售后服务政策（v2，2025 版）"),
            ("p", "本版本替代 v1（2024 版），自 2025 年 3 月 1 日起执行。"),
            ("h2", "2.1 无理由退货"),
            ("p", F_RETURN_V2),
            ("h2", "2.2 退货运费"),
            ("p", F_FREIGHT_V2),
            ("h2", "2.3 质量问题换货"),
            ("p", F_EXCHANGE_V2),
        ],
        header="岚盾智造 · 售后政策（现行版本）",
    )
    rec(p, purpose="现行版政策：跨文档/版本冲突的正确答案所在", expect="ok")

    # ---- 7. 退换货操作细则（DOCX，超长表格 >1200 字符 → 触发 atomic 被切坏）----
    p = out / "退换货操作细则.docx"
    big_rows = [["场景", "判定条件", "所需材料", "时限", "责任方"]]
    for i in range(1, 13):
        big_rows.append([
            f"场景 {i}",
            f"客户在签收后第 {i} 天提出，商品存在外观或功能异常，需在系统中登记工单并核对序列号",
            "发票、检测报告、开箱视频",
            f"{i + 5} 个工作日",
            "售后中心",
        ])
    build_docx(
        p,
        [
            ("h1", "第1章 退换货操作细则"),
            ("p", "本细则配合《售后服务政策（v2）》使用，冲突时以政策为准。"),
            ("h2", "1.1 场景判定表"),
            ("table", big_rows),
            ("h2", "1.2 运费处理"),
            ("p", F_FREIGHT_V2),
        ],
        header="岚盾智造 · 售后操作细则",
    )
    rec(
        p,
        purpose="超长表格（>1200 字符）：测 Section.atomic 是否真的阻止二次切分（已知缺陷靶子）",
        expect="ok",
    )

    # ---- 8. 上门安装服务规范（DOCX）----
    p = out / "上门安装服务规范.docx"
    build_docx(
        p,
        [
            ("h1", "第1章 上门安装服务规范"),
            ("h2", "1.1 响应时效"),
            ("p", F_INSTALL_SLA),
            ("h2", "1.2 服务费用"),
            ("p", F_INSTALL_FEE),
            ("h2", "1.3 安装标准"),
            ("p", F_MOUNT_HOLE),
        ],
        header="岚盾智造 · 服务规范",
    )
    rec(p, purpose="服务时效与费用：多跳 query 的来源之一", expect="ok")

    # ---- 9. 质保与延保说明（DOCX）----
    p = out / "质保与延保说明.docx"
    build_docx(
        p,
        [
            ("h1", "第1章 质保与延保说明"),
            ("h2", "1.1 延保价格"),
            ("p", F_EXTEND_PRICE),
            ("h2", "1.2 延保规则"),
            ("p", F_EXTEND_RULE),
            ("h2", "1.3 保外维修"),
            ("p", F_REPAIR_L1),
        ],
        header="岚盾智造 · 质保说明",
    )
    rec(p, purpose="延保与保外维修：多跳 + 表格行级取值", expect="ok")

    # ---- 10. 产品参数对照表（XLSX，3 sheet，宽表）----
    p = out / "岚盾产品参数对照表.xlsx"
    build_xlsx(
        p,
        [
            {
                "name": "整机参数",
                "rows": [
                    ["型号", "物料编码", "售价(元)", "整机保修(月)", "电池容量(mAh)",
                     "重量(kg)", "锁体材质", "支持天地钩", "开锁方式", "联网方式",
                     "工作温度", "防护等级", "应急供电", "适配门厚(mm)"],
                    ["L1", "LD-L1", 1299, 24, 5000, 3.2, "锌合金", "否", "指纹+密码", "WiFi",
                     "-10~55℃", "IP52", "Type-C", "40-120"],
                    ["L1 Pro", "LD-L1P", 1799, 36, 8000, 3.4, "锌合金", "是", "指纹+密码+NFC",
                     "WiFi+蓝牙", "-20~60℃", "IP54", "Type-C", "40-120"],
                    ["L2", "LD-L2", 999, 12, 4100, 4.1, "不锈钢", "否", "指纹", "WiFi",
                     "-10~55℃", "IP52", "无", "40-110"],
                    ["L2 Pro", "LD-L2P", 1499, 12, 10000, 4.15, "不锈钢", "是", "指纹+人脸",
                     "WiFi+4G", "-20~60℃", "IP55", "Type-C", "40-120"],
                ],
            },
            {
                "name": "维修价目",
                "rows": [
                    ["型号", "保外维修价(元)", "备注"],
                    ["L1", 199, "含工时"],
                    ["L1 Pro", 299, "含工时与上门"],
                    ["L2", 159, "含工时"],
                    ["L2 Pro", 259, "含工时与上门"],
                ],
            },
            {
                "name": "备件价格",
                "rows": [
                    ["备件", "单价(元)", "库存"],
                    ["L1 电池", 89, 120],
                    ["L1 Pro 电池", 129, 64],
                    ["标准锁体", 199, 30],
                    ["天地钩锁体", 259, 12],
                ],
            },
        ],
    )
    rec(
        p,
        purpose="3 sheet 宽表（14 列）：近义型号 + 行级取值；也是 atomic 边界被拍平的靶子",
        expect="ok",
    )

    # ---- 11. 2025 年售后服务统计（XLSX，含公式）----
    p = out / "2025年售后服务统计.xlsx"
    build_xlsx(
        p,
        [
            {
                "name": "月度工单",
                "rows": [
                    ["月份", "工单量", "一次解决率(%)", "平均处理时长(小时)", "满意度"],
                    *[[f"2025-{m:02d}", 100 + m * 7, round(78 + m * 0.6, 2), 30 - m * 0.5, 4.5]
                      for m in range(1, 7)],
                ],
            },
            {
                "name": "故障类型",
                "rows": [
                    ["故障类型", "占比(%)", "典型原因"],
                    ["指纹识别失败", 32.5, "手指潮湿或模块脏污"],
                    ["低电量告警", 21.0, "电池老化"],
                    ["联网失败", 18.5, "网关配对码错误"],
                    ["机械卡滞", 12.0, "锁体安装孔距不符"],
                ],
            },
            {
                "name": "汇总计算",
                "rows": [
                    ["指标", "值"],
                    ["上半年工单合计", "=SUM(月度工单!B2:B7)"],
                    ["平均一次解决率", "=AVERAGE(月度工单!C2:C7)"],
                ],
            },
        ],
    )
    rec(p, purpose="多 sheet 数据表 + 公式（data_only 取不到值，真实文档常见坑）", expect="ok")

    # ---- 12. 备件库存与价格（XLSX，合并单元格 + 空行）----
    p = out / "备件库存与价格.xlsx"
    build_xlsx(
        p,
        [
            {
                "name": "库存",
                "rows": [
                    ["备件", "单价(元)", "库存"],
                    ["L1 电池", 89, 120],
                    [],
                    ["L1 Pro 电池", 129, 64],
                    [],
                    ["标准锁体", 199, 30],
                    ["天地钩锁体", 259, 12],
                    ["备注：库存低于 20 需触发补货流程", None, None],
                ],
                "merge": ["A9:C9"],
            }
        ],
    )
    rec(p, purpose="合并单元格 + 空行：测表格窗口与行号映射", expect="ok")

    # ---- 13. 术语与别名表（MD）----
    p = out / "产品术语与别名表.md"
    p.write_text(
        "\n".join(
            [
                "# 产品术语与别名表",
                "",
                "## 型号别名",
                "",
                F_ALIAS,
                "",
                "## 术语解释",
                "",
                F_RESOLUTION,
                "",
                "- 天地钩（Tiandigou）：门锁锁体的一种加固结构，需门体预留天地钩孔位。",
                "- 保外维修：超出整机保修期后的付费维修。",
                "- 网关配对码：门锁与网关建立连接时使用的默认校验码。",
                "",
                "## 常见误写",
                "",
                "- 「岚盾 L1 豪华版」为 L1 的非正式叫法，正式型号为 L1 Pro。",
            ]
        ),
        encoding="utf-8",
    )
    rec(p, purpose="别名映射：别名 query 的答案来源（L1 尊享版 = L1 Pro）", expect="ok")

    # ---- 14. 常见问题 FAQ（MD，目录 + 附录 + 假表格）----
    p = out / "常见问题FAQ.md"
    p.write_text(
        "\n".join(
            [
                "# 常见问题 FAQ",
                "",
                "## 目录",
                "",
                "- 安装与适配",
                "- 电池与供电",
                "- 保修与售后",
                "",
                "## 安装与适配",
                "",
                F_TIANDIGOU,
                "",
                F_MOUNT_HOLE,
                "",
                "## 电池与供电",
                "",
                F_L1_BATTERY,
                "",
                F_L1PRO_BATTERY,
                "",
                "## 保修与售后",
                "",
                F_L1_WARRANTY,
                "",
                "## 附录A 快速对照",
                "",
                "| 项目 | 说明 |",
                "| --- | --- |",
                "",
                "（附录表格待补充）",
                "",
                "## 附录B 修订记录",
                "",
                "| 版本 | 日期 |",
                "| v2.1 | 2025-06-01 |",
                "- v2.0：补充 L1 Pro 电池说明",
            ]
        ),
        encoding="utf-8",
    )
    rec(p, purpose="目录页 + 附录 + 只有 2 行/表头不齐的假 Markdown 表格（表格识别负样本）", expect="ok")

    # ---- 15. 客服话术模板（TXT，无标题结构）----
    p = out / "客服话术模板.txt"
    p.write_text(
        "您好，这里是岚盾智造客服中心，工号{工号}为您服务。\n"
        "非常抱歉给您带来不便，我们会立即为您核实处理。\n"
        "关于保修问题，岚盾 L1 保修 24 个月，岚盾 L1 Pro 保修 36 个月，如需延保可在 App 内加购。\n"
        "关于上门安装，市区 24 小时内响应，郊区 48 小时内响应，L1 Pro 免上门费。\n"
        "如需转接人工，请告知您的问题类型，我这边为您转接专员。\n",
        encoding="utf-8",
    )
    rec(p, purpose="纯文本无标题：测'无结构文档塌缩成一个 section'的退化路径", expect="ok")

    # ---- 16. 变更记录（MD，长列表）----
    p = out / "产品变更记录.md"
    lines = ["# 岚盾产品变更记录", ""]
    for i in range(1, 26):
        lines.append(
            f"- 2025-{((i - 1) % 12) + 1:02d}-15：变更单 CR-{1000 + i}，"
            f"调整了第 {i} 项装配工艺参数，涉及 L1 系列锁体公差，影响范围已评估。"
        )
    lines += [
        "",
        "## 重要变更",
        "",
        F_L2PRO_PRICE,
        "",
        F_L2PRO_BATTERY,
    ]
    p.write_text("\n".join(lines), encoding="utf-8")
    rec(p, purpose="长列表文档：测长文档切分与噪声条目对召回的干扰", expect="ok")

    # ---- 17. 干扰域：云枢 S3 音箱（PDF）----
    p = out / "云枢S3音箱_产品手册.pdf"
    build_pdf(
        p,
        [
            [
                ("h1", "第1章 产品概览"),
                ("body", "云枢 S3 是一款桌面智能音箱，支持语音助手与多房间联动。"),
                ("body", F_S3_WARRANTY),
                ("body", F_S3_WEIGHT),
            ],
            [
                ("h1", "第2章 固件与升级"),
                ("body", F_S3_FIRMWARE),
                ("body", "音箱的保修政策与门锁产品线相互独立，请勿混用。"),
            ],
        ],
        header="云枢科技 · 音箱产品手册",
    )
    rec(p, purpose="干扰域：同样有'保修期/固件'词汇，且保修 24 个月与 L1 相同", expect="ok")

    # ---- 18. 干扰域：云枢参数表（XLSX）----
    p = out / "云枢S3音箱_参数表.xlsx"
    build_xlsx(
        p,
        [
            {
                "name": "整机参数",
                "rows": [
                    ["型号", "售价(元)", "整机保修(月)", "重量(kg)", "联网方式"],
                    ["S3", 399, 24, 0.6, "WiFi+蓝牙"],
                    ["S3 Pro", 599, 24, 0.75, "WiFi+蓝牙+Zigbee"],
                ],
            }
        ],
    )
    rec(p, purpose="干扰域参数表：型号/保修列与门锁表结构相似，测跨域混淆", expect="ok")

    # ---- 19. 哨兵：GBK 编码的 txt ----
    p = out / "术语表_GBK.txt"
    p.write_bytes(
        (
            "岚盾智造 内部术语表（GBK 编码，模拟历史系统导出的文件）\n"
            f"{F_GATEWAY_CODE}\n"
            "低电量告警阈值：电量低于 20% 时 App 推送提醒。\n"
        ).encode("gbk")
    )
    rec(
        p,
        purpose="GBK 编码 txt：当前 _load_txt 用 utf-8 硬解会 UnicodeDecodeError（哨兵，预期失败）",
        expect="fail",
        expect_note="预期入库失败；用来验证失败是否被正确记录（documents.parsed_status=failed）",
    )

    # ---- 20. 哨兵：截断的 PDF ----
    p = out / "损坏文件_截断.pdf"
    tmp = out / "_tmp_ok.pdf"
    build_pdf(tmp, [[("h1", "临时文档"), ("body", "用于制造截断文件。")]])
    data = tmp.read_bytes()
    p.write_bytes(data[: max(200, len(data) // 3)])
    tmp.unlink()
    rec(
        p,
        purpose="截断 PDF：解析应失败且不能拖垮流程（哨兵，预期失败）",
        expect="fail",
        expect_note="预期解析失败；检查是否返回 500 且 documents 表标记 failed",
    )

    # ---- 21. 哨兵：空文件 ----
    p = out / "空文件.txt"
    p.write_bytes(b"")
    rec(p, purpose="0 字节文件：预期 0 块", expect="empty", expect_note="预期 document_count=0")

    # ---- 22. 哨兵：不支持的格式 ----
    p = out / "不支持的格式.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    rec(p, purpose="后缀不在白名单：预期被接口拒绝", expect="reject")

    # ════════════════════════════════════════════════════════
    # 大宗文档族：手写文档负责"事实精确"，这一族负责"体量与干扰"。
    # 没有它，语料只有 28 个子块（比原有 53 块的库还小），
    # recall@40 这种指标就是在十几块里挑，测不出任何东西。
    # ════════════════════════════════════════════════════════
    import filler as F  # eval/gen 同目录；按 -m 方式运行时走下面那条回退

    # ---- 故障码与排查手册（MD：300 行表格）----
    p = out / "故障码与排查手册.md"
    rows = F.troubleshooting(rng, 300)
    p.write_text(
        "# 故障码与排查手册\n\n## 说明\n\n本手册覆盖门锁全系列的故障码，"
        "按现象检索时请以故障码为准。\n\n## 故障码总表\n\n"
        + md_table(rows)
        + "\n\n## 排查原则\n\n"
        "1. 先断电重启，再逐项排查硬件；\n"
        "2. 优先用故障码定位，避免凭现象猜测；\n"
        "3. 涉及锁体拆装的操作必须由认证工程师执行；\n"
        "4. 排查过程需在工单系统中留痕。\n",
        encoding="utf-8",
    )
    rec(p, purpose="大宗：300 行故障码表（长表格 + 长文档切分压力）", expect="ok")

    # ---- 安装作业指导书（MD：120 条步骤）----
    p = out / "安装作业指导书.md"
    p.write_text(
        "# 安装作业指导书\n\n## 作业流程\n\n"
        + "\n\n".join(F.install_steps(rng, 120))
        + "\n\n## 保养要求\n\n"
        + "\n\n".join(F.maintenance(rng, 60)),
        encoding="utf-8",
    )
    rec(p, purpose="大宗：安装步骤 + 保养建议，句式与 FAQ 高度相似", expect="ok")

    # ---- 客服培训材料（MD：对话 + 问答）----
    p = out / "客服培训材料.md"
    p.write_text(
        "# 客服培训材料\n\n## 场景对话\n\n"
        + "\n\n".join(F.training_dialogues(rng, 150))
        + "\n\n## 高频问答\n\n"
        + "\n\n".join(f"**问：{q}**\n\n答：{a}" for q, a in F.faq_entries(rng, 80)),
        encoding="utf-8",
    )
    rec(p, purpose="大宗：口语化对话（对 embedding 有干扰价值）", expect="ok")

    # ---- 术语总表（MD：250 条）----
    p = out / "术语总表.md"
    p.write_text(
        "# 术语总表\n\n以下术语按拼音首字母排序，编号唯一。\n\n"
        + "\n".join(F.term_entries(rng, 250)),
        encoding="utf-8",
    )
    rec(p, purpose="大宗：术语列表（与《产品术语与别名表》形成近义干扰）", expect="ok")

    # ---- 产品变更记录_全量（MD：300 条）----
    p = out / "产品变更记录_全量.md"
    p.write_text(
        "# 产品变更记录（全量）\n\n"
        + "\n".join(F.changelog(rng, 300)),
        encoding="utf-8",
    )
    rec(p, purpose="大宗：300 条变更记录（长列表切分 + 噪声条目）", expect="ok")

    # ---- 会议纪要（MD：150 条）----
    p = out / "服务例会纪要_2025.md"
    p.write_text(
        "# 服务例会纪要（2025）\n\n" + "\n\n".join(F.meeting_notes(rng, 150)),
        encoding="utf-8",
    )
    rec(p, purpose="大宗：带日期的决议（含过期信息，真实语料的典型噪声）", expect="ok")

    # ---- 工单样例（XLSX：3 sheet × 200 行）----
    p = out / "工单样例_2025.xlsx"
    head = ["工单号", "型号", "城市", "故障现象", "状态", "工程师", "耗时(小时)", "来源渠道"]
    build_xlsx(
        p,
        [
            {"name": f"工单-{i + 1}", "rows": [head, *F.tickets(rng, 200)]}
            for i in range(3)
        ],
    )
    rec(p, purpose="大宗：600 行工单样例（多 sheet 长表，行号映射压力）", expect="ok")

    # ---- 备件总表（XLSX：400 行）----
    p = out / "备件总表.xlsx"
    build_xlsx(
        p,
        [
            {
                "name": "备件明细",
                "rows": [
                    ["备件编码", "适配型号", "单价(元)", "库存", "状态", "最近入库"],
                    *F.spare_rows(rng, 400),
                ],
            }
        ],
    )
    rec(p, purpose="大宗：400 行备件表（长表格窗口）", expect="ok")

    # ---- 服务统计明细（XLSX：多 sheet）----
    p = out / "服务统计明细_2025.xlsx"
    build_xlsx(
        p,
        [
            {
                "name": "月度明细",
                "rows": [
                    ["月份", "工单量", "一次解决率(%)", "平均处理时长(小时)", "返修率(%)", "满意度"],
                    *[[f"2025-{m:02d}", 400 + m * 13, round(74 + m * 0.7, 2),
                       round(28 - m * 0.4, 1), round(2.5 - m * 0.05, 2), 4.4]
                      for m in range(1, 13)],
                ],
            },
            {
                "name": "城市分布",
                "rows": [["城市", "工单量", "按时完成率(%)"],
                         *[[c, 60 + i * 17, round(88 + i * 0.9, 1)]
                           for i, c in enumerate(F.CITIES)]],
            },
            {
                "name": "故障分布",
                "rows": [["故障现象", "占比(%)", "平均处理时长(小时)"],
                         *[[s, round(rng.uniform(3, 20), 1), round(rng.uniform(4, 60), 1)]
                           for s in F.FAULT_SYMPTOMS]],
            },
        ],
    )
    rec(p, purpose="大宗：多 sheet 统计明细（与 2025年售后服务统计 形成近义干扰）", expect="ok")

    # ---- 干扰域大宗：云枢音箱故障排查（XLSX）----
    p = out / "云枢音箱_故障排查.xlsx"
    build_xlsx(
        p,
        [
            {
                "name": "排查表",
                "rows": [["问题", "可能原因", "处理建议"],
                         *[[p_, _pick_pair(rng), "按指引重试"] for p_ in
                           [x for x in F.distractor_paragraphs(rng, 150)]]],
            }
        ],
    )
    rec(p, purpose="干扰域大宗：云枢音箱排查表（词汇与门锁重叠）", expect="ok")

    # ---- 语料家族：按 profile 把规模推到企业级（S 档为 0，保持原有复现结果）----
    family_golden = build_families(out, rng, profile, rec)
    if extra_golden is not None:
        extra_golden.extend(family_golden)

    return made


def _pick_pair(rng: random.Random) -> str:
    """给云枢排查表用的"可能原因"列（独立成函数避免长表达式里嵌套引号）。"""
    return rng.choice(["设备固件版本不一致", "路由器信道拥塞", "麦克风被遮挡", "网络延迟过高"])


# ════════════════════════════════════════════════════════════
# 三·补、语料家族（按 profile 把规模推到企业级）
#
# 为什么需要：S 档只有 32 份文档 / 400 多块，检索是在几百块里挑，
# recall@1 一条命中就动 3.8 个百分点，统计意义太弱。真实企业库是几千块。
#
# 但**规模不能靠灌水**：把同一份内容复制 100 遍只会让指标虚假变好。
# 这里用"同族近义干扰"——同族文件词汇高度重叠、取值各不相同：
#   同一个备件编码在 5 个区域表里单价不同；
#   同一个故障码在 4 本型号手册里含义不同；
#   12 个月的工单样例结构完全一样、只有编号与数值不同。
# 这才是"大海捞针 + 近似项区分"的真实难度，而不是灌水。
#
# 顺带产出标注：编号/取值在生成时就确定，所以锚点天然精确。
# ════════════════════════════════════════════════════════════
PROFILES = {
    "S": {"regions": 0, "months": 0, "manuals": 0, "quarters": 0, "lines": 0,
          "domains": 0, "batches": 0, "areas": 0, "versions": 0, "s3models": 0,
          "scans": 0, "notices": 0, "cases": 0, "exams": 0, "outlets": 0},
    # M：32 + 76 ≈ 108 份文件，目标 2000+ 子块
    "M": {"regions": 5, "months": 12, "manuals": 4, "quarters": 4, "lines": 4,
          "domains": 3, "batches": 5, "areas": 4, "versions": 3, "s3models": 4,
          "scans": 6, "notices": 5, "cases": 6, "exams": 4, "outlets": 3},
    # L：≈180 份文件，用于规模极限与召回退化曲线
    "L": {"regions": 9, "months": 12, "manuals": 8, "quarters": 8, "lines": 8,
          "domains": 6, "batches": 10, "areas": 8, "versions": 6, "s3models": 8,
          "scans": 12, "notices": 10, "cases": 12, "exams": 8, "outlets": 6},
}

REGIONS = ["华东", "华南", "华北", "西南", "华中", "东北", "西北", "东南", "中原"]
AREAS = ["杭州", "南京", "苏州", "宁波", "合肥", "无锡", "常州", "绍兴", "温州"]
DOMAINS = ["门锁硬件", "联网与网关", "售后流程", "安装施工", "数据指标", "备件物流"]
S3_MODELS = ["S3", "S3 Pro", "S3 Max", "S3 Lite", "S3 mini", "S3 Plus", "S3 SE", "S3 Ultra"]
QUARTERS = ["Q1", "Q2", "Q3", "Q4", "Q1下半年", "Q2下半年", "Q3下半年", "Q4下半年"]
FAULT_MODELS = ["L1", "L1 Pro", "L2", "L2 Pro", "L3", "L3 Pro", "X1", "X1 Pro"]
SPEC_CODES = [
    ("SL-1001", "标准锁体"), ("SL-1002", "天地钩锁体"), ("BT-2001", "L1 电池"),
    ("BT-2002", "L1 Pro 电池"), ("FP-3001", "指纹模块"), ("GW-4001", "网关模块"),
    ("PN-5001", "前面板"), ("CB-6001", "排线"), ("CY-7001", "锁芯"), ("SN-8001", "门磁"),
]
SYMPTOMS = ["指纹识别失败", "联网中断", "低电量告警", "机械卡滞", "门磁误报", "蓝牙配对失败"]
CAUSES_S = ["手指潮湿", "网关配对码错误", "电池老化", "锁舌受阻"]
ACTIONS_S = ["清洁感应区", "重新配对网关", "更换电池", "调整锁体"]


def build_families(out: Path, rng: random.Random, profile: str, rec) -> list[dict]:
    """按 profile 生成同族文档，返回由"植入事实"派生的标注行。"""
    spec = PROFILES.get(profile, PROFILES["S"])
    golden: list[dict] = []

    def plant(query, file, anchor, kind, bucket="编号检索", **kw):
        row = {"query": query, "bucket": bucket, "file": file.name, "anchor": anchor,
               "kind": kind}
        row.update(kw)
        golden.append(row)

    # ---- 1) 区域备件表：同一编码、不同区域单价不同（近似项区分）----
    for i in range(spec["regions"]):
        region = REGIONS[i]
        path = out / f"备件总表_{region}仓.xlsx"
        rows = [["备件编码", "名称", "单价(元)", "库存"]]
        planted = []
        for code, name in SPEC_CODES:
            price = 110 + (i + 1) * 7 + int(code[-2:]) * 3
            stock = 10 + (i * 13 + int(code[-3:-1])) % 180
            rows.append([code, name, price, stock])
            planted.append((code, name, price))
        build_xlsx(path, [{"name": f"{region}仓库存", "rows": rows}])
        rec(path, purpose=f"同族：{region}仓备件表（与其它区域同编码不同单价）", expect="ok")
        code, name, price = planted[i % len(planted)]
        plant(f"{region}仓的{name}（{code}）单价是多少元？", path,
              f"| {code} | {name} | {price} |", "same_code_cross_region")

    # ---- 2) 月度工单样例：结构相同、编号唯一 ----
    head = ["工单号", "型号", "城市", "故障现象", "状态", "工程师", "耗时(小时)", "来源渠道"]
    for m in range(1, spec["months"] + 1):
        path = out / f"工单样例_2025-{m:02d}.xlsx"
        rows = [head]
        ids = []
        for k in range(1, 121):
            wo = f"WO-25{m:02d}-{k:04d}"
            ids.append(wo)
            rows.append([wo, rng.choice([c[1] for c in SPEC_CODES]), rng.choice(AREAS),
                         rng.choice(SYMPTOMS), rng.choice(["已关闭", "已完成", "待客户确认"]),
                         rng.choice(["王工", "李工", "张工", "陈工"]),
                         rng.randint(1, 72), rng.choice(["400 电话", "App", "门店报修"])])
        build_xlsx(path, [{"name": f"工单-{m:02d}", "rows": rows}])
        rec(path, purpose=f"同族：2025-{m:02d} 月工单样例（120 行）", expect="ok")
        wo = ids[m % len(ids)]
        plant(f"工单 {wo} 的处理时长是多少小时？", path, wo, "unique_id_lookup")

    # ---- 3) 型号故障码手册：同故障码跨手册含义不同 ----
    for i in range(spec["manuals"]):
        model = FAULT_MODELS[i % len(FAULT_MODELS)]
        path = out / f"故障码手册_{model}.md"
        rows = [["故障码", "现象", "可能原因", "处理建议"]]
        codes = []
        for k in range(1, 61):
            code = f"E-{model.replace(' ', '')}-{1000 + k}"
            codes.append(code)
            rows.append([code, rng.choice(SYMPTOMS), rng.choice(CAUSES_S), rng.choice(ACTIONS_S)])
        path.write_text(
            f"# {model} 故障码手册\n\n本手册仅适用于 {model} 型号，"
            f"其它型号的同名故障码含义不同，请勿混用。\n\n" + md_table(rows),
            encoding="utf-8",
        )
        rec(path, purpose=f"同族：{model} 故障码手册（同名故障码跨手册含义不同）", expect="ok")
        code = codes[(i * 7) % len(codes)]
        plant(f"{model} 型号的故障码 {code} 是什么现象？", path, code, "same_code_cross_model")

    # ---- 4) 季度例会纪要：每份植入一条唯一决议 ----
    for i in range(spec["quarters"]):
        q = QUARTERS[i]
        path = out / f"服务例会纪要_{q}.md"
        fact = (f"{q} 例会决议：{REGIONS[i % len(REGIONS)]}仓备件安全库存下限"
                f"调整为 {120 + i * 15} 件。")
        path.write_text(f"# 服务例会纪要（{q}）\n\n{fact}\n\n"
                        + "\n\n".join(F.meeting_notes(rng, 60)), encoding="utf-8")
        rec(path, purpose=f"同族：{q} 例会纪要（含唯一决议句）", expect="ok")
        plant(f"{q} 例会决定把备件安全库存下限调到多少件？", path, fact, "planted_fact")

    # ---- 5) 产品线变更记录：CR 编号唯一 ----
    for i in range(spec["lines"]):
        line = ["门锁", "音箱", "摄像头", "门铃", "网关", "传感器", "配件", "中控"][i % 8]
        path = out / f"产品变更记录_{line}线.md"
        entries = []
        for k in range(1, 81):
            cr = f"CR-{line}-{2000 + k}"
            entries.append(f"- {cr}：{rng.choice(['调整装配公差', '修改固件默认值', '更换供应商', '更新包装清单'])}，"
                           f"影响范围已评估。")
        path.write_text(f"# {line}产品线变更记录\n\n" + "\n".join(entries), encoding="utf-8")
        rec(path, purpose=f"同族：{line}线变更记录（80 条唯一 CR 编号）", expect="ok")
        cr = f"CR-{line}-{2000 + (i * 11) % 80 + 1}"
        plant(f"变更单 {cr} 改了什么？", path, cr, "unique_id_lookup")

    # ---- 6) 领域术语总表：术语编号唯一 ----
    for i in range(spec["domains"]):
        dom = DOMAINS[i]
        path = out / f"术语总表_{dom}.md"
        entries = []
        for k in range(1, 101):
            tid = f"T-{dom}-{k:03d}"
            entries.append(f"- {tid}｜{rng.choice(['天地钩', '虚位密码', '门磁', '活体检测', '网关', '锁芯等级'])}："
                           f"指{rng.choice(['门锁结构中的加固部件', '安全等级的衡量指标', '与网关通信的子设备'])}。")
        path.write_text(f"# {dom}领域术语总表\n\n" + "\n".join(entries), encoding="utf-8")
        rec(path, purpose=f"同族：{dom}术语表（100 条唯一编号）", expect="ok")
        tid = f"T-{dom}-{(i * 23) % 100 + 1:03d}"
        plant(f"术语编号 {tid} 指的是什么？", path, tid, "unique_id_lookup")

    # ---- 7) 培训材料分批：批次题号唯一 ----
    for i in range(spec["batches"]):
        path = out / f"客服培训材料_第{i + 1}批.md"
        blocks = []
        for k in range(1, 61):
            qid = f"BATCH-{i + 1}-{k:03d}"
            blocks.append(
                f"**{qid} 问：**{rng.choice(['门锁离线怎么办？', '如何添加指纹？', '低电量能用多久？'])}\n\n"
                f"**答：**先在 App 内{rng.choice(['重新配网', '固件升级', '权限重置'])}，仍不行再提交工单。"
            )
        path.write_text(f"# 客服培训材料（第 {i + 1} 批）\n\n" + "\n\n".join(blocks),
                        encoding="utf-8")
        rec(path, purpose=f"同族：培训材料第 {i + 1} 批（60 条唯一题号）", expect="ok")
        qid = f"BATCH-{i + 1}-{(i * 17) % 60 + 1:03d}"
        plant(f"培训题 {qid} 的标准答案是什么？", path, qid, "unique_id_lookup")

    # ---- 8) 区域安装指导书：植入区域费用差异 ----
    for i in range(spec["areas"]):
        area = AREAS[i]
        path = out / f"安装作业指导书_{area}.md"
        fact = f"{area}区域上门安装服务费为 {199 + i * 20} 元，超出 20 公里的部分按每公里 3 元另计。"
        path.write_text(f"# {area}区域安装作业指导书\n\n{fact}\n\n"
                        + "\n\n".join(F.install_steps(rng, 40)), encoding="utf-8")
        rec(path, purpose=f"同族：{area}区域安装指导书（区域费用差异）", expect="ok")
        plant(f"{area}区域上门安装服务费是多少钱？", path, fact, "same_fact_cross_region")

    # ---- 9) 延保版本说明：同条款不同版本取值不同 ----
    for i in range(spec["versions"]):
        ver = ["A", "B", "C", "D", "E", "F"][i % 6]
        path = out / f"延保服务说明_版本{ver}.docx"
        fact = f"版本 {ver} 提供的延保时长为 {12 * (i + 1)} 个月，价格为 {299 + i * 100} 元。"
        build_docx(path, [("h1", f"延保服务说明（版本 {ver}）"), ("p", fact),
                          ("p", "本版本生效后，此前版本的延保条款同时废止。")],
                   header=f"岚盾智造 · 延保说明 {ver}")
        rec(path, purpose=f"同族：延保说明版本 {ver}（同条款不同版本取值不同）", expect="ok")
        plant(f"版本 {ver} 的延保是多长时间、多少钱？", path, fact, "same_fact_cross_version")

    # ---- 10/11) 云枢系列手册与参数表（干扰域扩量）----
    for i in range(spec["s3models"]):
        model = S3_MODELS[i]
        warranty = 12 + i * 6
        price = 499 + i * 200
        path = out / f"云枢{model}_产品手册.pdf"
        fact = f"云枢 {model} 智能音箱整机保修期为 {warranty} 个月。"
        build_pdf(path, [[("h1", "第1章 产品概览"),
                          ("body", f"云枢 {model} 是桌面智能音箱，支持语音助手与多房间联动。"),
                          ("body", fact)],
                         [("h1", "第2章 参数"),
                          ("body", f"{model} 整机重量为 {round(0.5 + i * 0.1, 2)}kg，支持 WiFi 与蓝牙。")]],
                  header=f"云枢科技 · {model} 产品手册")
        rec(path, purpose=f"干扰域同族：云枢 {model} 手册（保修期各不相同）", expect="ok")
        plant(f"云枢 {model} 音箱的保修期是多久？", path, fact, "distractor_family")

        path = out / f"云枢{model}_参数表.xlsx"
        build_xlsx(path, [{"name": "整机参数",
                           "rows": [["型号", "售价(元)", "整机保修(月)", "重量(kg)"],
                                    [model, price, warranty, round(0.5 + i * 0.1, 2)]]}])
        rec(path, purpose=f"干扰域同族：云枢 {model} 参数表", expect="ok")
        plant(f"云枢 {model} 的售价是多少元？", path, f"| {model} | {price} |", "distractor_family")

    # ---- 12) 扫描件 PDF：唯一编号只出现在扫描页里 ----
    scan_topics = ["应急供电", "安装孔位", "电池更换", "网关配对", "固件升级", "防撬报警",
                   "临时密码", "门磁校准", "指纹录入", "远程开锁", "报警静音", "低电提醒"]
    for i in range(spec["scans"]):
        topic = scan_topics[i % len(scan_topics)]
        sn = f"SN-{3300 + i}"
        scan_fact = f"本说明编号 {sn}：{topic}请参照图示操作。"
        path = out / f"扫描件说明_{topic}.pdf"
        build_pdf_with_scan(
            path,
            [[("h1", f"{topic}说明"),
              ("body", f"本页为电子版正文，完整内容见随附扫描页（编号 {sn}）。")]],
            scan_text=f"{topic}说明\n{scan_fact}\n如有疑问请联系当地服务网点。",
            header="岚盾智造 · 扫描件",
        )
        rec(path, purpose=f"扫描件同族：{topic}（唯一编号只在扫描页里）", expect="ok")
        plant(f"编号 {sn} 的说明是关于什么的？", path, sn, "ocr_sentinel")

    # ---- 13) 政策补充说明 ----
    # ⚠️ 主题列表长度必须 >= profile 的 notices 数：文件名由主题名派生，
    # 数量超了就会撞名 → 后面的覆盖前面的 → 前面的植入事实与锚点一起消失
    # （L 档 10 份 / 5 个主题时踩过，被 parse_check 的锚点校验逮住）。
    notice_topics = ["运费", "赠品", "发票", "上门改约", "二次维修",
                     "安装返工", "备件补发", "运费险", "延保转移", "上门检测"]
    assert len(notice_topics) >= spec["notices"], "政策补充说明主题数不足，会导致文件名撞车"
    for i in range(spec["notices"]):
        topic = notice_topics[i]
        path = out / f"政策补充说明_{topic}.docx"
        fact = f"补充说明 {i + 1}：涉及{topic}的争议，以签收后 {7 + i * 2} 天内提出的申请为准。"
        build_docx(path, [("h1", f"政策补充说明（{topic}）"), ("p", fact)],
                   header="岚盾智造 · 政策补充")
        rec(path, purpose=f"同族：政策补充说明（{topic}）", expect="ok")
        plant(f"{topic}相关的争议要在签收后多少天内提出？", path, fact, "same_fact_cross_doc")

    # ---- 14) 投诉案例集 ----
    for i in range(spec["cases"]):
        path = out / f"投诉案例集_第{i + 1}辑.md"
        blocks = []
        for k in range(1, 41):
            cid = f"CASE-{i + 1}-{k:03d}"
            blocks.append(f"### {cid}\n客户反馈：{rng.choice(['门锁无法联网', '指纹识别率低', '安装后门体变形'])}\n"
                          f"处理结论：{rng.choice(['换新', '上门复检', '补偿优惠券'])}")
        path.write_text(f"# 投诉案例集（第 {i + 1} 辑）\n\n" + "\n\n".join(blocks), encoding="utf-8")
        rec(path, purpose=f"同族：投诉案例集第 {i + 1} 辑（40 个唯一案例号）", expect="ok")
        cid = f"CASE-{i + 1}-{(i * 13) % 40 + 1:03d}"
        plant(f"投诉案例 {cid} 的处理结论是什么？", path, cid, "unique_id_lookup")

    # ---- 15) 培训试题 ----
    for i in range(spec["exams"]):
        path = out / f"培训试题_第{i + 1}套.md"
        blocks = []
        for k in range(1, 31):
            eid = f"EXAM-{i + 1}-{k:02d}"
            blocks.append(f"- {eid}. {rng.choice(['门锁离线首选排查步骤？', '延保如何生效？', '备件如何申请？'])}"
                          f"（答案见《客服培训材料》）")
        path.write_text(f"# 培训试题（第 {i + 1} 套）\n\n" + "\n".join(blocks), encoding="utf-8")
        rec(path, purpose=f"同族：培训试题第 {i + 1} 套", expect="ok")
        eid = f"EXAM-{i + 1}-{(i * 7) % 30 + 1:02d}"
        plant(f"试题 {eid} 考的是什么内容？", path, eid, "unique_id_lookup")

    # ---- 16) 服务网点清单 ----
    for i in range(spec["outlets"]):
        path = out / f"服务网点清单_第{i + 1}版.xlsx"
        rows = [["城市", "网点名称", "电话", "覆盖半径(km)"]]
        for j, city in enumerate(AREAS):
            rows.append([city, f"{city}服务中心", f"400-{1000 + i * 10 + j}", 20 + j])
        build_xlsx(path, [{"name": "网点", "rows": rows}])
        rec(path, purpose=f"同族：服务网点清单第 {i + 1} 版", expect="ok")
        city = AREAS[(i * 3) % len(AREAS)]
        plant(f"第 {i + 1} 版网点清单里{city}服务中心的覆盖半径是多少公里？", path,
              f"| {city} | {city}服务中心 |", "cross_doc_lookup")

    # ---- 程序化负样本：编号/型号不存在 ----
    if spec["months"]:
        for wo in ("WO-2599-0001", "WO-2501-9999"):
            golden.append({"query": f"工单 {wo} 的处理时长是多少小时？", "bucket": "负样本",
                           "file": None, "anchor": None, "kind": "refusal",
                           "expect": "编号不存在，应说明资料不足而不是硬召回"})
    if spec["s3models"]:
        golden.append({"query": "云枢 S9 Ultimate 音箱的保修期是多久？", "bucket": "负样本",
                       "file": None, "anchor": None, "kind": "refusal",
                       "expect": "该型号不存在（只有 S3 系列），强干扰负样本"})

    return golden



# ════════════════════════════════════════════════════════════
# 四、golden（标注）：anchor 直接引用事实常量 → 天然是库内原文子串
# ════════════════════════════════════════════════════════════
def build_golden() -> list[dict]:
    def q(query, bucket, file, anchor, kind, **kw):
        row = {"query": query, "bucket": bucket, "file": file, "anchor": anchor, "kind": kind}
        row.update(kw)
        return row

    rows = [
        # ── 单跳：原词 ──────────────────────────────────
        q("岚盾 L1 的整机保修期是多久？", "原词", "岚盾L1系列_产品手册_v2.1.pdf", F_L1_WARRANTY, "single_hop"),
        q("岚盾 L1 Pro 保修多长时间？", "原词", "岚盾L1系列_产品手册_v2.1.pdf", F_L1PRO_WARRANTY, "single_hop"),
        q("岚盾 L1 Pro 卖多少钱？", "原词", "岚盾L1系列_产品手册_v2.1.pdf", F_L1PRO_PRICE, "single_hop"),
        q("上门安装是多久响应？", "原词", "上门安装服务规范.docx", F_INSTALL_SLA, "single_hop"),
        q("延保 12 个月要多少钱？", "原词", "质保与延保说明.docx", F_EXTEND_PRICE, "single_hop"),
        q("保外维修 L1 Pro 多少钱？", "原词", "质保与延保说明.docx", F_REPAIR_L1, "table_lookup"),
        q("无理由退货的政策在 v2 里改成了几天？", "原词", "售后服务政策_v2_2025.docx", F_RETURN_V2, "version_conflict"),
        q("L1 Pro 电池的备件价格是多少？", "原词", "岚盾配件兼容表.pdf", F_SPARE_BATTERY, "table_lookup"),
        q("网关配对码默认是多少？", "原词", "术语表_GBK.txt", F_GATEWAY_CODE, "encoding_sentinel",
          expect="该文件是 GBK 编码，当前会解析失败 → 本条预期召回失败，用于暴露编码问题"),
        # ── 单跳：口语改写 ──────────────────────────────
        q("门锁没电了怎么应急供电？", "口语改写", "岚盾L1Pro_安装指南.pdf", F_EMERGENCY_POWER, "ocr_page",
          expect="事实只出现在扫描页，必须走 OCR 才能召回"),
        q("装锁师傅多久能到我家？", "口语改写", "上门安装服务规范.docx", F_INSTALL_SLA, "paraphrase"),
        q("质量问题退回去的运费谁掏？", "口语改写", "售后服务政策_v2_2025.docx", F_FREIGHT_V2, "version_conflict"),
        q("想要多保一年得花多少钱？", "口语改写", "质保与延保说明.docx", F_EXTEND_PRICE, "paraphrase"),
        q("哪些型号能用天地钩锁体？", "口语改写", "岚盾L1系列_产品手册_v2.1.pdf", F_TIANDIGOU, "paraphrase"),
        q("一次解决率是什么意思？", "口语改写", "产品术语与别名表.md", F_RESOLUTION, "term"),
        q("岚盾 L1 尊享版是哪个型号？", "口语改写", "产品术语与别名表.md", F_ALIAS, "alias"),
        q("LD-L1P 是哪个产品？", "口语改写", "产品术语与别名表.md", F_ALIAS, "alias"),
        # ── 跨文档 / 多跳 ──────────────────────────────
        q("L1 Pro 上门安装要收费吗？整机保修多久？", "跨文档", "上门安装服务规范.docx", F_INSTALL_FEE, "multi_hop",
          also_files=["岚盾L1系列_产品手册_v2.1.pdf"]),
        q("L2 和 L2 Pro 的保修期一样吗？", "跨文档", "岚盾产品参数对照表.xlsx", "| L2 Pro | LD-L2P | 1499 | 12 |", "multi_hop"),
        q("买了延保之后总共能保修多久？", "跨文档", "质保与延保说明.docx", F_EXTEND_RULE, "multi_hop"),
        q("锁体安装孔距是多少毫米？", "跨文档", "岚盾L1系列_产品手册_v2.1.pdf", F_MOUNT_HOLE, "table_lookup"),
        q("L2 Pro 的价格和电池容量分别是多少？", "跨文档", DOC_CHANGELOG, F_L2PRO_PRICE, "multi_hop",
          also_files=[DOC_PARAM_TABLE]),
        # ── 干扰域：跨域区分 ────────────────────────────
        q("云枢 S3 音箱的保修期是多久？", "跨文档", "云枢S3音箱_产品手册.pdf", F_S3_WARRANTY, "distractor"),
        q("云枢 S3 有多重？", "跨文档", "云枢S3音箱_参数表.xlsx", "| S3 | 399 | 24 | 0.6 |", "distractor"),
        q("云枢 S3 固件升级要收费吗？", "跨文档", "云枢S3音箱_产品手册.pdf", F_S3_FIRMWARE, "distractor"),
        # ── 英文文档 ────────────────────────────────────
        q("What is the warranty period of the L2 Pro?", "跨文档", "岚盾L2Pro_User_Manual_EN.pdf",
          "warranty period of the L2 Pro is 12 months", "english"),
        # ── 负样本：库里没有答案 ────────────────────────
        q("岚盾 L3 的售价是多少？", "负样本", None, None, "refusal",
          expect="不存在该型号，正确行为是不作答/说明资料不足"),
        q("岚盾门锁是哪一年上市的？", "负样本", None, None, "refusal",
          expect="语料未记载上市时间"),
        q("岚盾和云枢哪个更值得买？", "负样本", None, None, "refusal", expect="主观比较，语料无法回答"),
        q("帮我写一份门锁采购合同模板", "负样本", None, None, "refusal", expect="生成类请求，非知识库问题"),
        q("今天北京机动车限行尾号是多少？", "负样本", None, None, "refusal", expect="域外实时信息"),
        q("岚盾 L1 的固件升级要多少钱？", "负样本", None, None, "refusal",
          expect="固件升级是云枢产品线的话题，门锁语料里没有 → 强干扰负样本"),
    ]
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="S", choices=["S", "M", "L"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "eval" / "corpus"))
    ap.add_argument("--golden", default=str(ROOT / "eval" / "golden" / "queries.jsonl"))
    args = ap.parse_args()

    out = Path(args.out)
    rng = random.Random(args.seed)
    if _cjk_font_path() is None:
        print("缺少系统 CJK 字体，无法生成 PDF/扫描件语料", file=sys.stderr)
        return 2

    family_golden: list[dict] = []
    made = build_corpus(out, rng, args.profile, family_golden)
    golden = build_golden() + family_golden

    manifest = {
        "profile": args.profile,
        "seed": args.seed,
        "files": made,
        "golden_count": len(golden),
        "golden_kinds": {},
    }
    for row in golden:
        manifest["golden_kinds"][row["kind"]] = manifest["golden_kinds"].get(row["kind"], 0) + 1

    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    gpath = Path(args.golden)
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in golden) + "\n",
        encoding="utf-8",
    )

    total = sum(f["size"] for f in made)
    print(f"语料目录: {out}")
    print(f"生成文件: {len(made)} 个，合计 {total / 1024 / 1024:.1f} MB")
    for f in made:
        print(f"  {f['file']:<38} {f['format']:<5} {f['size'] / 1024:>8.1f} KB  预期={f['expect']}")
    print(f"\ngolden: {len(golden)} 条 → {gpath}")
    for kind, n in sorted(manifest["golden_kinds"].items()):
        print(f"  {kind:<20} {n}")
    print(f"\nmanifest: {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
