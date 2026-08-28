from fastapi import APIRouter, UploadFile, File
from pathlib import Path
import asyncio
from rag.rag import init_rag

router = APIRouter()

# 允许上传的文件类型
ALLOWED_SUFFIXES = {".pdf", ".txt", ".md", ".docx"}

# 知识库目录
KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    上传文档到知识库，自动触发 RAG 增量索引。

    支持格式：PDF / TXT / Markdown / DOCX
    """
    # 1. 校验文件类型
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        return {
            "success": False,
            "error": f"不支持的文件类型 '{suffix}'，仅支持: {', '.join(ALLOWED_SUFFIXES)}"
        }

# 构建保存路径
    save_dir = Path(KNOWLEDGE_BASE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在
    
    file_path = save_dir / file.filename
    content = await file.read()
    file_path.write_bytes(content)  # 保存文件（二进制写入）
    
    # 现在 file_path 是绝对路径，再传给 init_rag
    doc_count = await asyncio.to_thread(init_rag, str(file_path))

    return {
        "success": True,
        "filename": file.filename,
        # "file_size": len(content),
        "document_count": doc_count,
    }
