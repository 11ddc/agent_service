"""扫描件 PDF 的 OCR 回归测试 —— 零外部服务（不连 Redis/MySQL/Chroma/网络）。

背景（实测得到，不是推测）：
    pymupdf4llm 对"整页是图、没有文本层"的 PDF **既不导出图片、也不写 markdown
    图片引用**（page_chunks 的 text 为空、image_path 目录为空）。于是原实现
    `_load_pdf` 对扫描件返回 0 个 section，`init_rag` 返回 0 个块，而
    `/api/upload` 仍然回 {"success": true, "document_count": 0} —— 扫描件静默丢失，
    专门为图片写的 OCR + Qwen3-VL 混合链路永远不会被触发。

这里把修复钉住：扫描页必须被渲染并 OCR 成 atomic 单元入库，且与文字页共存。

依赖本机 tesseract 与系统 CJK 字体；缺任一则整体 skip（宁可跳过，不能静默通过）。
"""

from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

import rag.rag as rag  # 注意：import 它才会设置 pytesseract.tesseract_cmd

CJK_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simsun.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
)


def _cjk_font_path() -> str | None:
    for p in CJK_FONT_CANDIDATES:
        if Path(p).exists():
            return p
    return None


def _tesseract_ready() -> bool:
    import pytesseract

    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(
    _cjk_font_path() is None or not _tesseract_ready(),
    reason="需要本机 tesseract 与系统 CJK 字体",
)


def _page_image(text: str) -> Image.Image:
    """把一段文字渲染成整页图片（模拟扫描件：只有像素，没有文本层）。"""
    font_path = _cjk_font_path()
    img = Image.new("RGB", (1240, 1754), "white")
    draw = ImageDraw.Draw(img)
    body = ImageFont.truetype(font_path, 30)
    title = ImageFont.truetype(font_path, 40)
    y = 120
    for i, line in enumerate(text.split("\n")):
        draw.text((90, y), line, fill="black", font=title if i == 0 else body)
        y += 70
    return img


def _save_scan_pdf(path: Path, pages: list[str]) -> None:
    images = [_page_image(t) for t in pages]
    first, rest = images[0], images[1:]
    first.save(path, "PDF", resolution=100, save_all=True, append_images=rest)


@pytest.fixture
def offline_ocr(monkeypatch):
    """只用本地 tesseract：既不联网，也不触发付费视觉模型。"""
    monkeypatch.setenv("VISION_OCR_MODE", "off")


def test_scanned_pdf_is_ocr_indexed(tmp_path, offline_ocr):
    """整页扫描件必须被 OCR 成 atomic 单元 —— 修复前这里是 0 个 section。"""
    pdf = tmp_path / "扫描件_安装须知.pdf"
    _save_scan_pdf(
        pdf,
        [
            "扫描件安装须知",
            "A100 Pro 支持天地钩，安装孔距 60mm。\n保修凭证以发票日期为准。",
            "保外维修价目：A100 为 199 元，A100 Pro 为 299 元。",
        ],
    )

    sections = rag._load_pdf(str(pdf))

    assert sections, "扫描件不能解析出 0 个 section（这正是修复前的行为）"
    text = "\n".join(s.text for s in sections)
    assert "天地钩" in text, f"OCR 没读出关键实体，实际内容: {text[:200]!r}"
    assert "A100 Pro" in text
    assert "299" in text

    scans = [s for s in sections if s.kind == "image"]
    assert len(scans) == 3, "每一页扫描页都该产出一个图片型单元"
    assert all(s.atomic for s in scans), "扫描页整体不可再切"
    assert {s.page_start for s in scans} == {1, 2, 3}, "页码应记录真实页序"


def test_mixed_pdf_keeps_text_pages_and_scan_pages(tmp_path, offline_ocr):
    """一页文字 + 一页扫描件：两条路径都要保留，不能互相覆盖。"""
    import pymupdf

    text_pdf = tmp_path / "_text.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 90), "退换货政策", fontsize=18, fontname="china-s")
    page.insert_text(
        (72, 130), "自签收之日起 15 天内，商品未拆封可无理由退货。", fontsize=11, fontname="china-s"
    )
    doc.save(text_pdf)
    doc.close()

    scan_pdf = tmp_path / "_scan.pdf"
    _save_scan_pdf(scan_pdf, ["扫描页 安装补充说明"])

    mixed = tmp_path / "混合_政策与扫描页.pdf"
    out = pymupdf.open(text_pdf)
    with pymupdf.open(scan_pdf) as sc:
        out.insert_pdf(sc)
    out.save(mixed)
    out.close()

    sections = rag._load_pdf(str(mixed))

    text_secs = [s for s in sections if s.kind != "image"]
    image_secs = [s for s in sections if s.kind == "image"]
    assert text_secs, "文字页不该因为新增扫描件处理而丢失"
    assert image_secs, "扫描页仍要入库"

    joined = "\n".join(s.text for s in text_secs)
    assert "15 天" in joined
    assert {s.page_start for s in text_secs} == {1}
    assert {s.page_start for s in image_secs} == {2}


def test_text_only_pdf_is_not_treated_as_scan(tmp_path, monkeypatch):
    """普通文字 PDF 不能被误判成扫描件（否则会多做一次整页 OCR）。"""
    import pymupdf

    called = {"n": 0}
    real_is_scanned = rag._is_scanned_page

    def _spy(page):
        result = real_is_scanned(page)
        if result:
            called["n"] += 1
        return result

    monkeypatch.setattr(rag, "_is_scanned_page", _spy)

    pdf = tmp_path / "文字版.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 90), "保修政策", fontsize=18, fontname="china-s")
    page.insert_text((72, 130), "A100 整机保修 24 个月。", fontsize=11, fontname="china-s")
    doc.save(pdf)
    doc.close()

    sections = rag._load_pdf(str(pdf))

    assert called["n"] == 0
    assert any("24 个月" in s.text for s in sections)
