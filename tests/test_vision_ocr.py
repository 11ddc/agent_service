"""视觉模型 OCR 增强(rag/vision_ocr.py)测试 —— 零服务:视觉 API 全部打桩。

覆盖:
- 魔数嗅探;
- vision_extract_text 的入参拼装与空图短路、缺 key 报错;
- hybrid_image_text 三种模式(auto/always/off)与回落行为。
"""
import pytest

from rag import vision_ocr as vo
from rag.vision_ocr import VisionError


# ── _sniff_mime ──


@pytest.mark.parametrize(
    "blob,expected",
    [
        (b"\x89PNG\r\n\x1a\nrest", "image/png"),
        (b"\xff\xd8\xffrest", "image/jpeg"),
        (b"GIF89a...", "image/gif"),
        (b"RIFF....WEBP", "image/webp"),
        (b"II*\x00rest", "image/tiff"),
        (b"BMrest", "image/bmp"),
        (b"\x00\x01\x02", "image/png"),  # 无法识别 → 默认 png
    ],
)
def test_sniff_mime(blob, expected):
    assert vo._sniff_mime(blob) == expected


# ── vision_extract_text ──


def test_vision_extract_text_empty_bytes_short_circuits(monkeypatch):
    def _forbidden(*a, **k):
        raise AssertionError("空图片不应触发模型调用")

    monkeypatch.setattr(vo, "_request", _forbidden)

    assert vo.vision_extract_text(b"") == ""


def test_vision_extract_text_forwards_bytes_and_source(monkeypatch):
    captured = {}

    def _fake_request(image_bytes, *, prompt, mime, model):
        captured.update(image_bytes=image_bytes, prompt=prompt, mime=mime, model=model)
        return "识别到的表格内容"  # 契约:真实 _request 内部已 strip

    monkeypatch.setattr(vo, "_request", _fake_request)

    out = vo.vision_extract_text(b"PNGDATA", source="a.pdf#img1.png")

    assert out == "识别到的表格内容"
    assert captured["image_bytes"] == b"PNGDATA"
    assert captured["mime"] == "image/png"
    assert "a.pdf#img1.png" in captured["prompt"]
    assert captured["model"] == vo.DEFAULT_MODEL


def test_vision_extract_text_raises_when_key_missing(monkeypatch):
    monkeypatch.delenv("QIANWEN_API_KEY", raising=False)

    with pytest.raises(VisionError, match="QIANWEN_API_KEY"):
        vo.vision_extract_text(b"\x89PNG\r\n\x1a\nrest")


# ── hybrid_image_text:三种模式与回落 ──


def test_hybrid_off_never_calls_vision(monkeypatch):
    monkeypatch.setenv("VISION_OCR_MODE", "off")

    def _forbidden(*a, **k):
        raise AssertionError("off 模式不应调用视觉模型")

    monkeypatch.setattr(vo, "vision_extract_text", _forbidden)

    assert vo.hybrid_image_text(b"img", "") == ""
    assert vo.hybrid_image_text(b"img", "本地OCR结果") == "本地OCR结果"


def test_hybrid_auto_keeps_nonempty_ocr_without_vision(monkeypatch):
    monkeypatch.setenv("VISION_OCR_MODE", "auto")

    def _forbidden(*a, **k):
        raise AssertionError("OCR 已有结果时不应调用视觉模型")

    monkeypatch.setattr(vo, "vision_extract_text", _forbidden)

    assert vo.hybrid_image_text(b"img", "清晰文字") == "清晰文字"


def test_hybrid_auto_calls_vision_when_ocr_empty(monkeypatch):
    monkeypatch.setenv("VISION_OCR_MODE", "auto")
    monkeypatch.setattr(
        vo, "vision_extract_text", lambda image_bytes, source="": "视觉识别的表格"
    )

    assert vo.hybrid_image_text(b"img", "") == "视觉识别的表格"


def test_hybrid_auto_falls_back_to_empty_when_vision_fails(monkeypatch):
    monkeypatch.setenv("VISION_OCR_MODE", "auto")

    def _boom(*a, **k):
        raise VisionError("模型 500")

    monkeypatch.setattr(vo, "vision_extract_text", _boom)

    # 本地 OCR 为空 + 视觉失败 → 返回空(不抛,上传不中断)
    assert vo.hybrid_image_text(b"img", "") == ""


def test_hybrid_always_merges_ocr_and_vision(monkeypatch):
    monkeypatch.setenv("VISION_OCR_MODE", "always")
    monkeypatch.setattr(
        vo, "vision_extract_text", lambda image_bytes, source="": "视觉补充内容"
    )

    out = vo.hybrid_image_text(b"img", "本地内容")

    assert "本地内容" in out
    assert "[视觉补充]: 视觉补充内容" in out


def test_hybrid_always_keeps_ocr_when_vision_fails(monkeypatch):
    monkeypatch.setenv("VISION_OCR_MODE", "always")

    def _boom(*a, **k):
        raise VisionError("超时")

    monkeypatch.setattr(vo, "vision_extract_text", _boom)

    assert vo.hybrid_image_text(b"img", "本地内容") == "本地内容"
