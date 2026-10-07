import asyncio
import os
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from body_limit import MAX_UPLOAD_BYTES
from rag.rag import init_rag

router = APIRouter()

# 允许上传的文件类型
ALLOWED_SUFFIXES = {".pdf", ".txt", ".md", ".docx", ".xlsx", ".xlsm"}

# 知识库目录
KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"

# 单个文件大小上限。这是"别把进程内存和磁盘交给一个未鉴权的请求"的兜底，
# 不是产品策略——要放开就调这个常量。
#
# ⚠️ 真正的兜底在 `body_limit.py`：那一层在 multipart 解析**之前**按整个请求体
#    限流（Starlette 对文件字段没有任何大小上限，见该模块的说明）。
#    这里判的是"单个文件"的精确上限，在 multipart 帧开销之外留了余量。
#    常量从 body_limit 引入，避免两个上限各处维护、日久漂移。

# 分块读取大小：既不把整个文件读进内存，也不会因为块太小而频繁 await
_UPLOAD_CHUNK = 1024 * 1024

# Windows 保留设备名：`CON` / `NUL` / `AUX` / `COM1`… 在**任何扩展名下**都会被当成设备。
# 实测（Windows 11 / CPython 3.12）：
#   · `NUL.txt` → 写盘"成功"，但磁盘上什么都没有（接口却回报成功 = 静默丢数据）
#   · `aux.md` / `COM1.pdf` → 写盘直接抛 FileNotFoundError（接口 500）
# 两种都不该从上传接口漏出去。
_RESERVED_STEMS = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)

# Windows 文件名里的非法字符。这些不会造成越界，但会让 open() 抛 OSError → 接口 500，
# 而正确语义是 400（"你的文件名不合法"）。
_ILLEGAL_NAME_CHARS = frozenset('<>"|?*')

# 文件名长度上限（字节）。ext4/NTFS 单个组件上限 255 字节；超了就是 OSError。
MAX_FILENAME_BYTES = 255


def _is_reserved_device_name(name: str) -> bool:
    """按 Windows 的规则判断：取第一个点之前的词干，去掉尾部空格后比大小写不敏感。"""
    stem = name.split(".", 1)[0].strip().upper()
    return stem in _RESERVED_STEMS


def _safe_filename(raw: str | None) -> str:
    """把客户端传来的文件名收敛成一个**纯文件名**；带目录成分的直接拒绝。

    为什么必须做：multipart 里的 filename 是客户端随便写的字符串，Starlette
    只做 charset 解码、不做任何清洗（starlette/formparsers.py 里就是
    `_user_safe_decode(options[b"filename"], ...)`）。而 `save_dir / filename`
    有两个逃逸口：
      - `"../../main.py"`          → 沿路径向上爬到 knowledge_base 之外
      - `"C:/Windows/Temp/x.txt"`  → 绝对路径直接顶掉 save_dir
      - `"C:evil.pdf"`             → **盘符相对路径**：没有分隔符，上面那套
        "取最后一段再比对"会放它过去，但 `save_dir / "C:evil.pdf"` 并不在
        save_dir 下 —— Windows 上 ntpath.join 遇到"另一个盘符"会**整个丢弃**
        前一段，得到 `Path("C:evil.pdf")`，落盘落在那块盘的当前目录里。
    后缀白名单挡不住这三种（`Path("../../a.pdf").suffix == ".pdf"`）。

    做法：统一分隔符 → 取最后一段 → 要求"取完与原文完全一致"，
    即只接受不含任何目录成分的名字。反斜杠必须显式换掉：在 POSIX 上
    `\\` 不是分隔符，`"..\\..\\x.pdf"` 会被当成一个普通文件名漏过去。
    另外**冒号一律拒绝**：它同时是盘符分隔符和 NTFS 交换数据流（ADS）的分隔符。

    最后一层是"能落盘"的约束 —— 名字合法不代表能写成文件：
      - 控制字符（尤其裸 NUL）会让 `Path.resolve()` 抛 ValueError → 接口 500；
      - `< > " | ? *` 在 Windows 上是非法字符 → open() 抛 OSError → 500；
      - 超过文件名长度上限 → OSError → 500。
    这类请求应当得到 400（"你的文件名不合法"），而不是 500（"服务器挂了"）。
    """
    name = (raw or "").strip()
    leaf = name.replace("\\", "/").rsplit("/", 1)[-1]
    if leaf != name or leaf in {"", ".", ".."}:
        raise HTTPException(status_code=400, detail=f"非法文件名: {raw!r}")
    if ":" in leaf or os.path.splitdrive(leaf)[0]:
        raise HTTPException(status_code=400, detail=f"非法文件名: {raw!r}")
    # Windows 会**静默**去掉结尾的点和空格：不拦的话 `"报表.pdf."` 实际存成
    # `"报表.pdf"`，接口回报的名字却不是它，还可能覆盖掉另一次上传。
    if leaf != leaf.rstrip(". "):
        raise HTTPException(status_code=400, detail=f"非法文件名: {raw!r}")
    if _is_reserved_device_name(leaf):
        raise HTTPException(status_code=400, detail=f"非法文件名: {raw!r}")
    if any(ch in leaf for ch in _ILLEGAL_NAME_CHARS) or any(
        ord(ch) < 32 or ord(ch) == 127 for ch in leaf
    ):
        raise HTTPException(status_code=400, detail=f"非法文件名: {raw!r}")
    if len(leaf.encode("utf-8")) > MAX_FILENAME_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"文件名过长（上限 {MAX_FILENAME_BYTES} 字节）: {leaf[:40]!r}…",
        )
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

    # 兜底不变量：拼完之后，最终路径的父目录必须**就是**知识库目录。
    # 这一条防的不是上面那次校验，而是"将来有人放宽 _safe_filename"的回归 ——
    # 校验可以演化，但这个不变量不能破。resolve() 会把盘符相对路径
    # （"C:evil.pdf"）按那块盘的当前目录展开，从而在这里被挡住。
    #
    # ⚠️ resolve() 本身也可能抛：名字里含裸 NUL 时抛 ValueError（实测会变成 500）。
    #    那属于"文件名不合法"，应当 400。
    try:
        resolved_parent = file_path.resolve().parent
        expected_parent = save_dir.resolve()
    except (OSError, ValueError) as e:
        raise HTTPException(
            status_code=400, detail=f"非法文件名: {file.filename!r}（{type(e).__name__}）"
        ) from e
    if resolved_parent != expected_parent:
        raise HTTPException(status_code=400, detail=f"非法文件名: {file.filename!r}")

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
