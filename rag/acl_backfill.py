"""给**存量** chunk 与文档回填 ACL metadata：`python -m rag.acl_backfill`

## 为什么必须显式回填

ACL 打开后，**缺少 ACL 元数据的块一律判为不可见**。这是刻意选的失败方向 ——
不能让"字段缺失"悄悄变成"默认放行"。代价是：存量库在开启 ACL 的那一刻会突然
"检索不到东西"。正确的做法是给一次**显式、可审计**的回填，而不是把默认放宽。

回填内容（两个存储都回填，缺一不可）：

- Chroma 每个子块的 metadata：`status=published`、`visibility=<默认>`、
  `tenant_id=<默认>`、`owner_id=legacy`
- MySQL `documents` 行：同上（否则控制台会一直显示"待审核"）

## 为什么用"Chroma 里缺 status 的块"来界定存量

不按时间戳猜：只有**真正缺 ACL 元数据**的块才是存量。新上传的文档一定带 ACL
字段，所以重复执行是幂等的，也不会把新上传的草稿误发布。

## 用法

    python -m rag.acl_backfill --dry-run              # 只看有多少要回填
    python -m rag.acl_backfill                        # 回填（默认 tenant 可见）
    python -m rag.acl_backfill --visibility public    # 改成所有租户可见
"""

import argparse
import sys

import config
import rag.rag as rag_module  # 必须按模块引用：`from rag.rag import _vectorstore` 拿的是**值拷贝**，
#                              后面 _ensure_ready() 填的是 rag.rag 里的全局变量，拷贝不会更新
from db import DocumentStore
from db.mysql import MySQLUnavailable
from rag import acl as rag_acl

BATCH = 500


def _legacy_sources(col) -> dict[str, int]:
    """扫描全库，返回 {source: 缺 ACL 的块数}。"""
    legacy: dict[str, int] = {}
    offset = 0
    total = col.count()
    while offset < total:
        data = col.get(limit=BATCH, offset=offset, include=["metadatas"])
        for meta in data.get("metadatas") or []:
            meta = meta or {}
            if "status" in meta and "visibility" in meta and "tenant_id" in meta:
                continue
            source = str(meta.get("source") or "")
            if source:
                legacy[source] = legacy.get(source, 0) + 1
        offset += BATCH
    return legacy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="给存量文档回填 ACL")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不修改")
    parser.add_argument(
        "--visibility", default=rag_acl.DEFAULT_VISIBILITY,
        choices=list(rag_acl.VISIBILITIES), help="回填成的可见范围",
    )
    parser.add_argument("--tenant-id", default=config.AUTH_DEFAULT_TENANT)
    parser.add_argument("--owner-id", default="legacy", help="回填成谁上传的")
    args = parser.parse_args(argv)

    if rag_module._vectorstore is None:
        rag_module._ensure_ready()
    if rag_module._vectorstore is None:
        print("知识库连接失败，无法回填", file=sys.stderr)
        return 1

    col = rag_module._vectorstore._collection
    total = col.count()
    legacy = _legacy_sources(col)
    chunks = sum(legacy.values())

    print(f"集合: {config.CHROMA_COLLECTION}  总块数: {total}")
    print(f"缺 ACL 元数据的块: {chunks}（分布在 {len(legacy)} 份文档里）")
    if not chunks:
        print("无需回填（所有块都已带 ACL 元数据）")
        return 0

    if args.dry_run:
        print()
        print("dry-run：未修改任何数据。示例文档：")
        for source, n in list(legacy.items())[:5]:
            print(f"  {n:5} 块  {source}")
        return 0

    acl_fields = {
        "tenant_id": args.tenant_id,
        "owner_id": args.owner_id,
        "visibility": args.visibility,
        "status": rag_acl.STATUS_PUBLISHED,
    }

    updated = 0
    for source in legacy:
        updated += _backfill_source(col, source, acl_fields)

    print(f"已回填 {updated} 个块（visibility={args.visibility} tenant={args.tenant_id}）")

    # MySQL 侧同样回填，否则管理界面会一直显示"待审核"
    store = DocumentStore()
    rows = 0
    for source in legacy:
        doc_id = _doc_id_of(source)
        try:
            rows += store.backfill_acl(
                doc_id,
                tenant_id=args.tenant_id,
                owner_id=args.owner_id,
                visibility=args.visibility,
                status=rag_acl.STATUS_PUBLISHED,
            )
        except MySQLUnavailable as e:
            print(f"MySQL 不可用，文档记录未能回填（Chroma 已回填）: {e}", file=sys.stderr)
            return 1
    print(f"已回填 {rows} 条文档记录")

    # 元数据变了但块数没变 → 必须让 BM25 缓存失效
    rag_module.invalidate_index_cache()
    print("已让 BM25 缓存失效，下次检索会重建")
    return 0


def _doc_id_of(source: str) -> str:
    from rag import structure as st

    return st.doc_id_of(source)


def _backfill_source(col, source: str, fields: dict) -> int:
    """按 source 取全部块 → 合并 metadata → 写回。

    ⚠️ 一次取完，**不要**按 `where={"source": ...}` + offset 翻页：
    回填后这些块**仍然匹配**同一个 where，offset 又不前进，就会反复取到同一批
    而**死循环**。一份文档的块数是有界的（几百块量级），一次取完更简单也更安全。
    """
    data = col.get(where={"source": source}, include=["metadatas"])
    ids = list(data.get("ids") or [])
    if not ids:
        return 0
    metas = [dict(m or {}, **fields) for m in (data.get("metadatas") or [])]
    col.update(ids=ids, metadatas=metas)
    return len(ids)


if __name__ == "__main__":
    raise SystemExit(main())
