"""视觉模型 OCR 增强(混合方案)—— 基于千问 Qwen3-VL-Flash(DashScope OpenAI 兼容接口)。

设计:本地 pytesseract 快路径 + 视觉大模型慢路径,按开关分流:
  - VISION_OCR_MODE=auto(默认):本地 OCR 文本为空时才调视觉模型,控制成本;
  - VISION_OCR_MODE=always:每张图片都调视觉模型,结果与 OCR 合并(更全、更贵);
  - VISION_OCR_MODE=off:关闭视觉增强,退化为纯 pytesseract(等于原行为)。

可调环境变量:
  - VISION_MODEL:模型名,默认 qwen3-vl-flash
  - QIANWEN_API_KEY:密钥(与项目里 DashScopeEmbeddings 同源)

设计要点:
  - import 本模块零开销(openai 客户端延迟到调用时创建、按次新建,避免缺 key 时
    import 即炸,也天然线程安全);
  - 任何模型调用失败统一抛 VisionError,调用方(rag.py)负责回落原逻辑,
    保证"视觉模型挂了也不影响上传入库"。
"""

import base64
import os

from dotenv import load_dotenv

load_dotenv(encoding="utf-8-sig")  # utf-8-sig:兼容带 BOM 的 .env(键名不会被 \ufeff 污染)

DEFAULT_MODEL = "qwen3-vl-flash"
_DASH_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 文件头魔数 → mime(轻量嗅探;识别不了就按 image/png,视觉模型能容忍)
_MIME_SNIFF = [
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"RIFF", "image/webp"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
    (b"BM", "image/bmp"),
]


class VisionError(RuntimeError):
    """视觉模型调用失败(缺 key / 网络 / 接口报错 / 无返回)。"""


def _mode() -> str:
    return os.getenv("VISION_OCR_MODE", "auto").strip().lower()


def _sniff_mime(image_bytes: bytes) -> str:
    for magic, mime in _MIME_SNIFF:
        if image_bytes[: len(magic)] == magic:
            return mime
    return "image/png"


def _data_url(image_bytes: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"


_SYSTEM_PROMPT = (
    "你是文档识别助手。用户会发来一张来自文档(PDF/Word)的图片,请:"
    "1) 完整转写图中所有文字,保持原有顺序与换行;"
    "2) 若包含表格,输出 Markdown 格式表格(保留表头与行列;合并单元格无法表达时,"
    "   用合理占位保持行列对齐);"
    "3) 若是图表/流程图/截图,先一句话概括内容,再转写图中文字。"
    "只输出识别内容本身,不要寒暄、不要解释过程。"
)


def _request(image_bytes: bytes, prompt: str, mime: str, model: str) -> str:
    """真正调用视觉模型(openai 客户端按次创建,延迟导入)。"""
    api_key = os.getenv("QIANWEN_API_KEY")
    if not api_key:
        raise VisionError("未配置 QIANWEN_API_KEY,视觉模型不可用")

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=_DASH_BASE_URL, timeout=60)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": _data_url(image_bytes, mime)},
                        },
                    ],
                },
            ],
            temperature=0.1,
        )
    except Exception as e:
        raise VisionError(f"视觉模型调用失败: {e!r}") from e

    try:
        text = resp.choices[0].message.content or ""
    except (AttributeError, IndexError) as e:
        raise VisionError(f"视觉模型返回结构异常: {e!r}") from e
    return text.strip()


def vision_extract_text(image_bytes: bytes, *, source: str = "") -> str:
    """用视觉模型识别一张图片,返回结构化文本(失败抛 VisionError)。"""
    if not image_bytes:
        return ""
    model = os.getenv("VISION_MODEL", DEFAULT_MODEL)
    prompt = "请识别下面这张来自文档的图片"
    if source:
        prompt += f"(来源:{source})"
    prompt += ":"
    return _request(
        image_bytes, prompt=prompt, mime=_sniff_mime(image_bytes), model=model
    )


def hybrid_image_text(image_bytes: bytes, ocr_text: str, *, source: str = "") -> str:
    """混合 OCR:根据 VISION_OCR_MODE 决定是否补一轮视觉模型。

    返回最终应入库的图片文本;任何异常都降级为"只用本地 OCR 结果",
    保证视觉模型故障不影响文档入库。
    """
    ocr_text = ocr_text or ""
    mode = _mode()

    if mode == "off":
        return ocr_text.strip()
    if mode == "always" and ocr_text.strip():
        # 都保留:本地结果 + 视觉补充(带标记,避免两边内容混在一起难以核对)
        try:
            vision_text = vision_extract_text(image_bytes, source=source)
        except VisionError as e:
            print(f"[vision_ocr] 视觉补充失败,仅保留本地 OCR: {e}")
            return ocr_text.strip()
        if vision_text:
            return f"{ocr_text.strip()}\n\n[视觉补充]: {vision_text}"
        return ocr_text.strip()

    # mode == "auto"(或 always 但本地无结果):OCR 为空才补视觉
    if ocr_text.strip():
        return ocr_text.strip()

    try:
        return vision_extract_text(image_bytes, source=source)
    except VisionError as e:
        print(f"[vision_ocr] 本地 OCR 为空且视觉模型不可用,图片文字跳过: {e}")
        return ""
