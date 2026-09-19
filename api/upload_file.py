import asyncio
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from rag.rag import init_rag

router = APIRouter()

# 允许上传的文件类型
ALLOWED_SUFFIXES = {".pdf", ".txt", ".md", ".docx", ".xlsx", ".xlsm"}

# 知识库目录
KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"

# 单个文件大小上限。这是"别把进程内存和磁盘交给一个未鉴权的请求"的兜底，
# 不是产品策略——要放开就调这个常量。
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB

# 分块读取大小：既不把整个文件读进内存，也不会因为块太小而频繁 await
_UPLOAD_CHUNK = 1024 * 1024


def _safe_filename(raw: str | None) -> str:
    """把客户端传来的文件名收敛成一个**纯文件名**；带目录成分的直接拒绝。

    为什么必须做：multipart 里的 filename 是客户端随便写的字符串，Starlette
    只做 charset 解码、不做任何清洗（starlette/formparsers.py 里就是
    `_user_safe_decode(options[b"filename"], ...)`）。而 `save_dir / filename`
    有两个逃逸口：
      - `"../../main.py"`          → 沿路径向上爬到 knowledge_base 之外
      - `"C:/Windows/Temp/x.txt"`  → 绝对路径直接顶掉 save_dir
    后缀白名单挡不住这两种（`Path("../../a.pdf").suffix == ".pdf"`）。

    做法：统一分隔符 → 取最后一段 → 要求"取完与原文完全一致"，
    即只接受不含任何目录成分的名字。反斜杠必须显式换掉：在 POSIX 上
    `\\` 不是分隔符，`"..\\..\\x.pdf"` 会被当成一个普通文件名漏过去。
    """
    name = (raw or "").strip()
    leaf = name.replace("\\", "/").rsplit("/", 1)[-1]
    if leaf != name or leaf in {"", ".", ".."}:
        raise HTTPException(status_code=400, detail=f"非法文件名: {raw!r}")
    return leaf


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    上传文档到知识库，自动触发 RAG 增量索引。

    支持格式：PDF / TXT / Markdown / DOCX / Excel(.xlsx/.xlsm)
    """
    # 1. 文件名与类型校验（文件名先过：路径穿越要在落盘之前挡住）
    filename = _safe_filename(file.filename)
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        return {
            "success": False,
            "error": f"不支持的文件类型 '{suffix}'，仅支持: {', '.join(ALLOWED_SUFFIXES)}",
        }

    # 2. 大小预检：multipart 解析完就知道长度了，不必先把超限文件收进内存
    size = getattr(file, "size", None)
    if size is not None and size > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"文件过大（{size} 字节，上限 {MAX_UPLOAD_BYTES} 字节）",
        )

    save_dir = Path(KNOWLEDGE_BASE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在
    file_path = save_dir / filename

    # 3. 分块落盘到临时文件，最后原子改名：
    #    - 不会把整个文件读进内存（原来 `content = await file.read()` 是全量读）
    #    - 中断/超限时不会留下一个"看起来完整"的半截文档给索引去解析
    #    临时名带 uuid：并发上传同一个文件名时不会互相写花对方的临时文件。
    part_path = save_dir / f".{filename}.{uuid4().hex}.part"
    written = 0
    try:
        with part_path.open("wb") as fh:
            while True:
                chunk = await file.read(_UPLOAD_CHUNK)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    # 兜底：客户端用 chunked 传输、没有 Content-Length 时
                    # size 可能取不到，只能边收边判
                    raise HTTPException(
                        status_code=413,
                        detail=f"文件过大（超过上限 {MAX_UPLOAD_BYTES} 字节）",
                    )
                fh.write(chunk)
        part_path.replace(file_path)  # 原子替换：要么是完整文件，要么什么都没变
    finally:
        part_path.unlink(missing_ok=True)  # 成功时已被 replace 掉，这里是失败清理

    # 4. 索引（解析+embedding+入库是重活，丢线程池，别卡事件循环）
    doc_count = await asyncio.to_thread(init_rag, str(file_path))

    return {
        "success": True,
        "filename": filename,
        # "file_size": written,
        "document_count": doc_count,
    }
