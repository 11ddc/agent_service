"""文档元数据存储：记录哪些文档入了库、解析是否成功、切了多少块、用的哪版切分参数。

它解决三个实际问题：
1. **重建一致性校验** —— Chroma（子块）与 MySQL（父块）是两次独立写入，
   没有跨库事务。靠 chunk_count / parent_count / chunk_schema_ver 能发现
   "两个存储不一致"或"切分参数已过期"，而不是让它静默劣化。
2. **切分参数升级** —— 改了分隔符或 chunk_size 后，按 chunk_schema_ver 精确
   找出哪些文档需要重新解析，不必全库重灌。
3. **知识库管理** —— 有这张表才谈得上做管理界面。
"""

from dataclasses import dataclass
from datetime import datetime

from db.mysql import mysql_cursor

_COLUMNS = (
    "doc_id, filename, source, uploaded_at, parsed_status, error_msg, "
    "chunk_count, parent_count, chunk_schema_ver, "
    "tenant_id, owner_id, visibility, status, published_at, published_by"
)

STATUS_OK = "ok"
STATUS_FAILED = "failed"


@dataclass(frozen=True)
class DocumentRow:
    doc_id: str
    filename: str
    source: str
    uploaded_at: datetime | None
    parsed_status: str
    error_msg: str | None
    chunk_count: int
    parent_count: int
    chunk_schema_ver: str | None
    # ↓ ACL 与审核状态（A4）。带默认值放在末尾，保持既有按位置构造的调用兼容。
    tenant_id: str = "default"
    owner_id: str | None = None
    visibility: str = "tenant"
    status: str = "draft"
    published_at: datetime | None = None
    published_by: str | None = None


def _row_to_document(row: tuple) -> DocumentRow:
    return DocumentRow(
        doc_id=row[0],
        filename=row[1],
        source=row[2],
        uploaded_at=row[3],
        parsed_status=row[4],
        error_msg=row[5],
        chunk_count=row[6] or 0,
        parent_count=row[7] or 0,
        chunk_schema_ver=row[8],
        tenant_id=row[9] or "default",
        owner_id=row[10],
        visibility=row[11] or "tenant",
        status=row[12] or "draft",
        published_at=row[13],
        published_by=row[14],
    )


class DocumentStore:
    """documents 表的读写。MySQL 不可用时抛 MySQLUnavailable，由调用方降级。

    注意：文档元数据是「记录性」数据，不是回答问题的必需数据，
    所以入库流程里它失败**不应中断**整个上传 —— 由 rag.py 单独 try 包住。
    """

    def _upsert(
        self,
        doc_id: str,
        filename: str,
        source: str,
        parsed_status: str,
        chunk_count: int,
        parent_count: int,
        chunk_schema_ver: str | None,
        error_msg: str | None,
        acl: dict | None = None,
    ) -> None:
        acl = acl or {}
        with mysql_cursor() as cur:
            cur.execute(
                "INSERT INTO documents "
                "(doc_id, filename, source, uploaded_at, parsed_status, error_msg, "
                " chunk_count, parent_count, chunk_schema_ver, "
                " tenant_id, owner_id, visibility, status) "
                "VALUES (%s, %s, %s, NOW(), %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                # 同一个文件重新上传时刷新全部字段：uploaded_at 语义是"最近一次入库时间"。
                #
                # ⚠️ ACL 字段也一起刷新：**重新上传 = 内容变了 = 需要重新审核**。
                #    所以之前已发布的文档重新上传后会回到 draft（审核状态不会被继承）。
                "ON DUPLICATE KEY UPDATE "
                " filename = VALUES(filename), source = VALUES(source), "
                " uploaded_at = VALUES(uploaded_at), "
                " parsed_status = VALUES(parsed_status), "
                " error_msg = VALUES(error_msg), "
                " chunk_count = VALUES(chunk_count), "
                " parent_count = VALUES(parent_count), "
                " chunk_schema_ver = VALUES(chunk_schema_ver), "
                " tenant_id = VALUES(tenant_id), owner_id = VALUES(owner_id), "
                " visibility = VALUES(visibility), status = VALUES(status), "
                " published_at = NULL, published_by = NULL",
                (
                    doc_id,
                    filename,
                    source,
                    parsed_status,
                    error_msg,
                    chunk_count,
                    parent_count,
                    chunk_schema_ver,
                    acl.get("tenant_id", "default"),
                    acl.get("owner_id"),
                    acl.get("visibility", "tenant"),
                    acl.get("status", "draft"),
                ),
            )

    def mark_ok(
        self,
        doc_id: str,
        filename: str,
        source: str,
        chunk_count: int,
        parent_count: int,
        chunk_schema_ver: str,
        acl: dict | None = None,
    ) -> None:
        self._upsert(
            doc_id,
            filename,
            source,
            STATUS_OK,
            chunk_count,
            parent_count,
            chunk_schema_ver,
            None,
            acl,
        )

    def mark_failed(
        self,
        doc_id: str,
        filename: str,
        source: str,
        error_msg: str,
        chunk_schema_ver: str | None = None,
        acl: dict | None = None,
    ) -> None:
        self._upsert(
            doc_id,
            filename,
            source,
            STATUS_FAILED,
            0,
            0,
            chunk_schema_ver,
            error_msg[:2000],
            acl,
        )

    def set_status(self, doc_id: str, status: str, published_by: str | None = None) -> int:
        """改审核状态。只有切到 `published` 才写 published_at/published_by。"""
        with mysql_cursor() as cur:
            cur.execute(
                "UPDATE documents SET status = %s, "
                " published_at = CASE WHEN %s = 'published' THEN NOW() ELSE published_at END, "
                " published_by = CASE WHEN %s = 'published' THEN %s ELSE published_by END "
                "WHERE doc_id = %s",
                (status, status, status, published_by, doc_id),
            )
            return int(cur.rowcount or 0)

    def set_visibility(self, doc_id: str, visibility: str) -> int:
        with mysql_cursor() as cur:
            cur.execute(
                "UPDATE documents SET visibility = %s WHERE doc_id = %s",
                (visibility, doc_id),
            )
            return int(cur.rowcount or 0)

    def backfill_acl(
        self,
        doc_id: str,
        *,
        tenant_id: str,
        owner_id: str | None,
        visibility: str,
        status: str,
    ) -> int:
        """给**存量**文档回填 ACL（供 `python -m rag.acl_backfill` 使用）。

        为什么要有它：ACL 新列的默认值是 `draft` —— 存量文档会被判为"待审核"从而
        检索不到。回填把它们标成"本租户已发布"，最接近上线 ACL 之前的实际行为。
        """
        with mysql_cursor() as cur:
            cur.execute(
                "UPDATE documents SET tenant_id = %s, owner_id = %s, visibility = %s, "
                "status = %s, published_at = CASE WHEN %s = 'published' THEN NOW() "
                "ELSE published_at END, published_by = 'backfill' WHERE doc_id = %s",
                (tenant_id, owner_id, visibility, status, status, doc_id),
            )
            return int(cur.rowcount or 0)

    def list_by_status(
        self, status: str | None = None, tenant_id: str | None = None, limit: int = 200
    ) -> list[DocumentRow]:
        limit = max(1, min(int(limit), 1000))
        sql = f"SELECT {_COLUMNS} FROM documents"
        conds, args = [], []
        if status:
            conds.append("status = %s")
            args.append(status)
        if tenant_id:
            conds.append("tenant_id = %s")
            args.append(tenant_id)
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        sql += f" ORDER BY uploaded_at DESC LIMIT {limit}"
        with mysql_cursor() as cur:
            cur.execute(sql, tuple(args))
            return [_row_to_document(row) for row in cur.fetchall()]

    def get(self, doc_id: str) -> DocumentRow | None:
        with mysql_cursor() as cur:
            cur.execute(
                f"SELECT {_COLUMNS} FROM documents WHERE doc_id = %s", (doc_id,)
            )
            row = cur.fetchone()
        return _row_to_document(row) if row else None

    def all(self) -> list[DocumentRow]:
        with mysql_cursor() as cur:
            cur.execute(f"SELECT {_COLUMNS} FROM documents ORDER BY uploaded_at")
            return [_row_to_document(row) for row in cur.fetchall()]

    def delete(self, doc_id: str) -> int:
        with mysql_cursor() as cur:
            cur.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))
            return cur.rowcount or 0
