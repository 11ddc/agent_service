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
    "chunk_count, parent_count, chunk_schema_ver"
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
    ) -> None:
        with mysql_cursor() as cur:
            cur.execute(
                "INSERT INTO documents "
                "(doc_id, filename, source, uploaded_at, parsed_status, error_msg, "
                " chunk_count, parent_count, chunk_schema_ver) "
                "VALUES (%s, %s, %s, NOW(), %s, %s, %s, %s, %s) "
                # 同一个文件重新上传时刷新全部字段：uploaded_at 语义是"最近一次入库时间"
                "ON DUPLICATE KEY UPDATE "
                " filename = VALUES(filename), source = VALUES(source), "
                " uploaded_at = VALUES(uploaded_at), "
                " parsed_status = VALUES(parsed_status), "
                " error_msg = VALUES(error_msg), "
                " chunk_count = VALUES(chunk_count), "
                " parent_count = VALUES(parent_count), "
                " chunk_schema_ver = VALUES(chunk_schema_ver)",
                (
                    doc_id,
                    filename,
                    source,
                    parsed_status,
                    error_msg,
                    chunk_count,
                    parent_count,
                    chunk_schema_ver,
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
        )

    def mark_failed(
        self,
        doc_id: str,
        filename: str,
        source: str,
        error_msg: str,
        chunk_schema_ver: str | None = None,
    ) -> None:
        self._upsert(
            doc_id, filename, source, STATUS_FAILED, 0, 0, chunk_schema_ver, error_msg[:2000]
        )

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
