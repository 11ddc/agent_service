"""知识库管理接口：审核发布 / 下架 / 改可见范围 / 文档清单。

## 为什么"上传即生效"不行

知识库内容会被注入**所有用户**的提示词。一份错误文档、或者一份本该
"只有售后能看"的内部文档直接生效，后果是"所有用户都被喂了错误/越权的答案"。
所以流程是：上传 → `draft`（**检索不到**）→ 审核 → `published` → 才参与回答。

| 方法 | 路径 | 作用 | 角色 |
|---|---|---|---|
| GET | `/api/kb/documents` | 文档清单（可按 status 过滤） | kb_admin / operator |
| POST | `/api/kb/documents/{doc_id}/publish` | 审核通过（参与回答） | kb_admin |
| POST | `/api/kb/documents/{doc_id}/archive` | 下架（立刻不参与回答） | kb_admin |
| POST | `/api/kb/documents/{doc_id}/visibility` | 改可见范围 | kb_admin |

## 两处存储的更新顺序是**按失败方向**选的

检索只认 Chroma 里的 metadata，MySQL 只是"记录 + 控制台展示"，两者没有跨库事务。
所以顺序不是随便定的：

- **发布**：先改 MySQL，再改 Chroma。中途失败 → 库里写着 published、但块还是 draft
  → **文档依然检索不到**（安全方向），接口会明确报错让运维重试（操作幂等）。
- **下架**：先改 Chroma，再改 MySQL。中途失败 → 块已经查不到、库里还写着 published
  → **依然是不可见**（安全方向）。

反过来的话，一次失败就可能让一份内部文档在"下架"之后仍然被检索到。
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from auth import audit
from auth.deps import Principal, require_roles
from db import DocumentStore, MySQLUnavailable
from rag import acl as rag_acl
from rag.rag import KnowledgeBaseError, update_document_acl

logger = logging.getLogger(__name__)

router = APIRouter()

_mutate_roles = require_roles("kb_admin")
_view_roles = require_roles("kb_admin", "operator")


class VisibilityRequest(BaseModel):
    visibility: str = Field(..., max_length=16)


class DocumentOut(BaseModel):
    doc_id: str
    filename: str
    parsed_status: str
    chunk_count: int
    parent_count: int
    tenant_id: str
    owner_id: str | None = None
    visibility: str
    status: str
    uploaded_at: str | None = None
    published_at: str | None = None
    published_by: str | None = None


def _doc_out(row) -> DocumentOut:
    return DocumentOut(
        doc_id=row.doc_id,
        filename=row.filename,
        parsed_status=row.parsed_status,
        chunk_count=row.chunk_count,
        parent_count=row.parent_count,
        tenant_id=row.tenant_id,
        owner_id=row.owner_id,
        visibility=row.visibility,
        status=row.status,
        uploaded_at=row.uploaded_at.isoformat() if row.uploaded_at else None,
        published_at=row.published_at.isoformat() if row.published_at else None,
        published_by=row.published_by,
    )


def _load_document(doc_id: str):
    """取文档行；不存在 → 404；MySQL 不可用 → 503（管理接口没有"降级"一说）。"""
    try:
        row = DocumentStore().get(doc_id)
    except MySQLUnavailable as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="文档存储暂时不可用，请稍后重试",
        ) from e
    if row is None:
        raise HTTPException(status_code=404, detail=f"文档不存在: {doc_id}")
    return row


def _sync_chunks(source: str, **fields) -> int:
    """把 ACL 变更同步到向量库的子块 metadata。失败 → 502（让调用方重试）。"""
    try:
        return update_document_acl(source, **fields)
    except KnowledgeBaseError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"向量库更新失败，变更可能未完全生效，请重试：{e}",
        ) from e


@router.get("/documents")
async def list_documents(
    principal: Principal = Depends(_view_roles),
    status_filter: str | None = None,
    limit: int = 200,
) -> list[DocumentOut]:
    """文档清单。默认只列**本租户**的（跨租户要看别的租户得显式扩展）。"""
    try:
        rows = DocumentStore().list_by_status(
            status=status_filter, tenant_id=principal.tenant_id, limit=limit
        )
    except MySQLUnavailable as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="文档存储暂时不可用，请稍后重试",
        ) from e
    return [_doc_out(r) for r in rows]


@router.post("/documents/{doc_id}/publish")
async def publish_document(
    doc_id: str,
    request: Request,
    principal: Principal = Depends(_mutate_roles),
) -> dict:
    """审核通过：文档从此参与回答。

    顺序：**先 MySQL 再 Chroma** —— 中途失败时文档依然检索不到（安全方向）。
    """
    row = _load_document(doc_id)

    try:
        DocumentStore().set_status(doc_id, rag_acl.STATUS_PUBLISHED, principal.user_id)
    except MySQLUnavailable as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="文档存储暂时不可用，请稍后重试",
        ) from e

    chunks = _sync_chunks(row.source, status=rag_acl.STATUS_PUBLISHED)

    await audit.record(
        "kb.publish", "ok", principal=principal, target=row.filename,
        detail=f"doc_id={doc_id} chunks={chunks}", request=request,
    )
    logger.info("文档已发布: %s（%s 块）by %s", row.filename, chunks, principal.username)

    result = _doc_out(_load_document(doc_id)).model_dump()
    result["chunks_updated"] = chunks
    if chunks == 0:
        # 块数为 0 说明这份文档压根没进向量库（解析失败过）
        result["warning"] = "该文档在向量库里没有块（可能解析失败），发布后仍无法被检索到"
    return result


@router.post("/documents/{doc_id}/archive")
async def archive_document(
    doc_id: str,
    request: Request,
    principal: Principal = Depends(_mutate_roles),
) -> dict:
    """下架：立刻不参与回答。

    顺序：**先 Chroma 再 MySQL** —— 中途失败时文档依然不可见（安全方向）。
    """
    row = _load_document(doc_id)

    chunks = _sync_chunks(row.source, status=rag_acl.STATUS_ARCHIVED)

    try:
        DocumentStore().set_status(doc_id, rag_acl.STATUS_ARCHIVED)
    except MySQLUnavailable as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="文档存储暂时不可用，请稍后重试",
        ) from e

    await audit.record(
        "kb.archive", "ok", principal=principal, target=row.filename,
        detail=f"doc_id={doc_id} chunks={chunks}", request=request,
    )
    logger.info("文档已下架: %s（%s 块）by %s", row.filename, chunks, principal.username)

    result = _doc_out(_load_document(doc_id)).model_dump()
    result["chunks_updated"] = chunks
    return result


@router.post("/documents/{doc_id}/visibility")
async def set_document_visibility(
    doc_id: str,
    payload: VisibilityRequest,
    request: Request,
    principal: Principal = Depends(_mutate_roles),
) -> dict:
    """改可见范围：tenant（本租户）/ private（仅上传者）/ public（所有租户）。"""
    visibility = (payload.visibility or "").strip().lower()
    if visibility not in rag_acl.VISIBILITIES:
        raise HTTPException(
            status_code=422,
            detail=f"未知的可见性 {payload.visibility!r}（可选：{', '.join(rag_acl.VISIBILITIES)}）",
        )

    row = _load_document(doc_id)
    chunks = _sync_chunks(row.source, visibility=visibility)

    try:
        DocumentStore().set_visibility(doc_id, visibility)
    except MySQLUnavailable as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="文档存储暂时不可用，请稍后重试",
        ) from e

    await audit.record(
        "kb.set_visibility", "ok", principal=principal, target=row.filename,
        detail=f"doc_id={doc_id} visibility={visibility} chunks={chunks}", request=request,
    )

    result = _doc_out(_load_document(doc_id)).model_dump()
    result["chunks_updated"] = chunks
    return result
