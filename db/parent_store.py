"""父块存储：按文件整体替换父块，按 id 批量取回全文。

父块只服务生成侧（大块给 LLM 读），不参与向量/BM25 检索，所以只需要主键查询，
不需要放进 Chroma 白算一遍 embedding。

一致性规则（Chroma 与 MySQL 是两个存储，没有跨库事务）：
    写入：先写父块（replace 是单事务，原子）→ 再写子块（Chroma）
    删除：先删子块（Chroma）→ 再删父块（replace 开头那次 DELETE）
中途失败只留下孤儿父块（无害），不会出现"子块指向不存在的父块"。
"""

import hashlib

from db.mysql import mysql_cursor


def source_hash(source: str) -> str:
    """按文件**路径**算 hash，用于按文件清理。

    不要用正文 hash：文档改一个字 hash 就变了，旧父块永远清理不掉，
    结果是"每次编辑都多留一份旧内容"。
    """
    return hashlib.sha1(source.encode("utf-8")).hexdigest()


def replace(
    source: str,
    doc_id: str,
    parents: list[tuple[str, str, str | None, int | None, int | None]],
) -> int:
    """按文件整体替换父块：同一事务内先删旧、再插新。返回写入条数。

    parents 是 [(parent_id, 父块正文, breadcrumb, page_start, page_end), ...]，
    id 由调用方按确定性规则生成。后三列是给"引用可核验"用的（表结构早就留好了，
    之前一直没写）。
    先删再插让"删掉了但没插上"不可能发生（异常会整体回滚）。
    """
    sh = source_hash(source)
    rows = [
        (pid, doc_id, source, sh, idx, len(text), text, crumb, ps, pe)
        for idx, (pid, text, crumb, ps, pe) in enumerate(parents)
    ]
    with mysql_cursor() as cur:
        cur.execute("DELETE FROM parents WHERE source_hash = %s", (sh,))
        if rows:
            cur.executemany(
                "INSERT INTO parents "
                "(parent_id, doc_id, source, source_hash, order_idx, char_len, text, "
                " breadcrumb, page_start, page_end) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                rows,
            )
    return len(rows)


def get(ids: list[str]) -> dict[str, dict]:
    """批量按 id 取父块，返回 {parent_id: {"text", "breadcrumb", "page_start"}}。

    一次 IN 查询，不是 N 次单查。带上 breadcrumb/page_start 是为了让生成侧能把
    [docN] 标注回章节与页码 —— 只给正文的话，用户永远无法核验出处。
    """
    if not ids:
        return {}
    placeholders = ", ".join(["%s"] * len(ids))
    with mysql_cursor() as cur:
        cur.execute(
            "SELECT parent_id, text, breadcrumb, page_start FROM parents "
            f"WHERE parent_id IN ({placeholders})",
            tuple(ids),
        )
        return {
            row[0]: {"text": row[1], "breadcrumb": row[2], "page_start": row[3]}
            for row in cur.fetchall()
        }


def count() -> int:
    with mysql_cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM parents")
        row = cur.fetchone()
        return int(row[0]) if row else 0
