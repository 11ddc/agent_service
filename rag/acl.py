"""文档级访问控制（ACL）—— 企业知识库的权限边界。

## 为什么必须有

企业知识库里同时存在"全员可见的产品手册"、"**只有售后能看的内部判责标准**"、
"某客户专属的合同条款"。检索不做权限过滤，等于把内部文档发给所有用户。
而这个项目原来的检索路径**没有任何权限过滤** —— 全仓唯一的 metadata filter 是
`{"source": ...}`（限定文档内检索），dense 与 sparse 两条召回都是全库检索。

## 四个 ACL 字段（写进每个 chunk 的 metadata）

| 字段 | 含义 |
|---|---|
| `tenant_id` | 租户：数据隔离维度 |
| `owner_id` | 上传者（`private` 可见性时唯一能看的人） |
| `visibility` | `tenant`（本租户可见，默认）/ `private`（仅 owner）/ `public`（所有租户） |
| `status` | `draft`（待审核）/ `published`（已发布，**只有它参与回答**）/ `archived`（下架） |

## 强制在哪两处生效（最容易漏的就是第二处）

检索有两条通道，**两条都要过滤**：

1. **dense**：Chroma 的 `similarity_search(..., filter=...)`；
2. **sparse**：进程内的 BM25 对**全量语料**打分 —— 只给 Chroma 加 filter 的话，
   被限制的文档照样会从 BM25 这一路被召回出来。`Acl.allows()` 就是这一路的判据，
   它与 `chroma_filter()` 的语义必须保持一致（有测试专门对齐这两者）。

## ACL 怎么传到检索深处

通过 **ContextVar**（与 `agent/events.py` 的 emitter 同思路）：工具函数的签名由
LLM 的工具 schema 决定，没法加 `acl` 参数；ContextVar 让"一次请求一个 ACL"
自然成立，并且同步/异步、主线程/工作线程都能读到（`asyncio.to_thread` 会拷贝上下文）。

## 失败方向

**默认关闭**：认不出的 `visibility` 一律拒绝；`status` 不是 `published` 一律拒绝。
没有 ACL 元数据的历史块也会被拒（用 `python -m rag.acl_backfill` 回填）。
`ACL_ENABLED=false` 会退回"不过滤"，只应出现在本地演示，启动与 `/health` 都会暴露它。
"""

import logging
import os
from contextvars import ContextVar
from dataclasses import dataclass

import config

logger = logging.getLogger(__name__)

VISIBILITY_TENANT = "tenant"
VISIBILITY_PRIVATE = "private"
VISIBILITY_PUBLIC = "public"
VISIBILITIES = (VISIBILITY_TENANT, VISIBILITY_PRIVATE, VISIBILITY_PUBLIC)
DEFAULT_VISIBILITY = VISIBILITY_TENANT

STATUS_DRAFT = "draft"
STATUS_PUBLISHED = "published"
STATUS_ARCHIVED = "archived"
STATUSES = (STATUS_DRAFT, STATUS_PUBLISHED, STATUS_ARCHIVED)

# 非请求上下文（评测脚本、离线任务、单测）用的系统身份
SYSTEM_USER_ID = "system"


@dataclass(frozen=True)
class Acl:
    """一次检索的访问上下文。

    注意 `role` **不参与可见性判断**：回答链路上对所有人一视同仁地只取已发布文档。
    管理员要看草稿走知识库管理接口（`/api/kb/*`），不把草稿混进答案里。
    """

    tenant_id: str = config.AUTH_DEFAULT_TENANT
    user_id: str = SYSTEM_USER_ID
    role: str = "user"

    @classmethod
    def system(cls) -> "Acl":
        """非请求上下文：默认租户 + 系统身份。"""
        return cls(tenant_id=config.AUTH_DEFAULT_TENANT, user_id=SYSTEM_USER_ID)

    @classmethod
    def of(cls, principal) -> "Acl":
        """从已认证身份构造（principal 来自 auth.deps）。"""
        return cls(
            tenant_id=getattr(principal, "tenant_id", None) or config.AUTH_DEFAULT_TENANT,
            user_id=getattr(principal, "user_id", None) or SYSTEM_USER_ID,
            role=str(getattr(getattr(principal, "role", None), "value", "user")),
        )

    def chroma_filter(self) -> dict | None:
        """dense 通道的 where 过滤；ACL 关闭时返回 None（不过滤）。"""
        if not config.ACL_ENABLED:
            return None
        return {
            "$and": [
                {"status": {"$eq": STATUS_PUBLISHED}},
                {
                    "$or": [
                        {"visibility": {"$eq": VISIBILITY_PUBLIC}},
                        {
                            "$and": [
                                {"visibility": {"$eq": VISIBILITY_TENANT}},
                                {"tenant_id": {"$eq": self.tenant_id}},
                            ]
                        },
                        {
                            "$and": [
                                {"visibility": {"$eq": VISIBILITY_PRIVATE}},
                                {"owner_id": {"$eq": self.user_id}},
                            ]
                        },
                    ]
                },
            ]
        }

    def allows(self, meta: dict | None) -> bool:
        """sparse（BM25）通道的判据，语义必须与 `chroma_filter()` 一致。

        缺字段时按"不可见"处理：没有 ACL 元数据的历史块需要显式回填，
        而不是默认放行。
        """
        if not config.ACL_ENABLED:
            return True
        meta = meta or {}
        if str(meta.get("status") or "") != STATUS_PUBLISHED:
            return False
        visibility = str(meta.get("visibility") or "")
        if visibility == VISIBILITY_PUBLIC:
            return True
        if visibility == VISIBILITY_TENANT:
            return str(meta.get("tenant_id") or "") == self.tenant_id
        if visibility == VISIBILITY_PRIVATE:
            return str(meta.get("owner_id") or "") == self.user_id
        return False  # 认不出的可见性 → 拒绝


_current: ContextVar[Acl | None] = ContextVar("rag_current_acl", default=None)


def current_acl() -> Acl:
    """当前请求的访问上下文；未设置时用系统身份（评测/离线任务）。"""
    acl = _current.get()
    return acl if acl is not None else Acl.system()


def set_acl(acl: Acl | None):
    return _current.set(acl)


def reset_acl(token) -> None:
    _current.reset(token)


def acl_metadata(acl: Acl, *, status: str, visibility: str) -> dict:
    """构造要写进每个 chunk 的 ACL metadata。"""
    if visibility not in VISIBILITIES:
        raise ValueError(
            f"未知的可见性 {visibility!r}（可选：{', '.join(VISIBILITIES)}）"
        )
    if status not in STATUSES:
        raise ValueError(f"未知的状态 {status!r}（可选：{', '.join(STATUSES)}）")
    return {
        "tenant_id": acl.tenant_id or config.AUTH_DEFAULT_TENANT,
        "owner_id": acl.user_id or SYSTEM_USER_ID,
        "visibility": visibility,
        "status": status,
    }


def merge_filters(*filters: dict | None) -> dict | None:
    """把若干 where 条件合并成一个（用于"限定文档内检索"这类已经带条件的场景）。

    ⚠️ 必须**展开一层的 `$and`**：Chroma 的校验器不允许 `$and` 里再套 `$and`
    （`$or` 里套 `$and` 是可以的）。直接拼成 `{"$and": [a, {"$and": [...]}]}` 会抛
    `ValueError: Expected where operator to be one of ...`，而且只有真跑这条路径
    才会发现 —— 实测确认过。展开后是 `{"$and": [字段条件, 字段条件, {"$or": [...]}]}`，
    正好是 Chroma 接受的形状。
    """
    flat: list[dict] = []
    for f in filters:
        if not f:
            continue
        if set(f) == {"$and"} and isinstance(f["$and"], list):
            flat.extend(f["$and"])
        else:
            flat.append(f)
    if not flat:
        return None
    if len(flat) == 1:
        return flat[0]
    return {"$and": flat}
