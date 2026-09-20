"""客服 Agent 的**真实**知识库工具 —— 没有占位实现，全部基于库里已有的数据。

## 为什么需要这一层

工具调用的价值不在于"有工具"，而在于**工具背后真有数据**。这个模块刻意只包含
四类数据源已就绪的能力：

    search_knowledge_base    全库检索          ← Chroma + BM25 + 精排（主链路同款）
    search_in_document       限定文档检索      ← Chroma 的 source 过滤
    list_documents           文档清单          ← MySQL documents 表
    get_document_outline     文档章节结构      ← MySQL parents 表的 breadcrumb

**刻意不做的**：查订单 / 查物流 / 查退款进度。这个项目没有任何订单系统的数据，
在进程内写一个"返回成功"的假工具，只会让模型编出看起来很确定的业务数据 ——
那比没有工具更糟。这类能力正确的形态是走 MCP 边界接真实服务
（见 `mcp_server.py` 与 `.env` 的 `MCP_SERVERS`），或直接短路成"暂不支持"。

## 三条契约（测试里守着）

1. 每个工具的 docstring 必须写清**什么时候该用** —— 模型唯一的选型依据就是它；
2. 底层不可用时返回**可读的降级话术**，绝不抛异常 —— 工具抛错会打断整条
   tool_agent 子图；
3. 返回文本必须带**可核验的出处**（文件名 / 章节 / 页码），否则工具结果无法被
   引用，等于又造了一个不可溯源的信息源。
"""

import logging
import os

from langchain_core.tools import tool

from db import DocumentStore, MySQLUnavailable, parent_store
from rag.rag import KnowledgeBaseError, reordering, retrieve_sync

logger = logging.getLogger(__name__)

# 单个片段回灌给模型的最大字符数：工具结果会和 prompt 一起进上下文，
# 整篇塞进去会把预算吃光（RAG 侧有专门的装箱，工具侧靠这个常量兜）。
_SNIPPET_CHARS = 600
_DEFAULT_TOP_K = 5
_MAX_TOP_K = 10
_MAX_LISTED = 50
_MAX_SECTIONS = 80


# ══════════════════════════════════════════════════════════════
# 依赖入口（全部做成模块级函数，方便测试打桩，也让"MySQL 中途才起来"能生效）
# ══════════════════════════════════════════════════════════════


def _document_store() -> DocumentStore:
    """文档元数据存储。**每次调用时构造**：不缓存连接状态。"""
    return DocumentStore()


def _ready_store():
    """读取侧的 Chroma 实例（懒初始化 + 线程安全），不可用时抛 KnowledgeBaseError。"""
    from rag import rag as _rag

    _rag._ensure_ready()
    if _rag._vectorstore is None:
        raise KnowledgeBaseError("知识库连接失败，请检查后重试。")
    return _rag._vectorstore


def _parent_outline(doc_id: str) -> list[str]:
    """该文档的章节路径列表（父块表里的 breadcrumb，去重后即天然目录）。"""
    return parent_store.outline(doc_id)


# ══════════════════════════════════════════════════════════════
# 渲染与匹配
# ══════════════════════════════════════════════════════════════


def _cite(meta: dict) -> str:
    """拼出处：文件名 ｜ 章节 ｜ 页码。三者缺一，用户就没法回原文核对。"""
    parts = [os.path.basename(meta.get("source") or "") or "未知来源"]
    if meta.get("breadcrumb"):
        parts.append(str(meta["breadcrumb"]))
    if meta.get("page_start") is not None:
        parts.append(f"p.{meta['page_start']}")
    return " ｜ ".join(parts)


def _render(docs: list) -> str:
    blocks = []
    for i, doc in enumerate(docs, 1):
        text = (getattr(doc, "page_content", "") or "").strip()
        if not text:
            continue
        if len(text) > _SNIPPET_CHARS:
            text = text[:_SNIPPET_CHARS] + "…"
        blocks.append(f"[{i}] {_cite(getattr(doc, 'metadata', {}) or {})}\n{text}")
    return "\n\n".join(blocks)


def _clamp_top_k(top_k) -> int:
    try:
        return max(1, min(int(top_k), _MAX_TOP_K))
    except (TypeError, ValueError):
        return _DEFAULT_TOP_K


def _match_documents(rows: list, name: str) -> list:
    """按文件名找文档：先精确（忽略大小写），再退回唯一子串匹配。

    用户说的是"售后服务政策"，文件名却是"售后服务政策_v2_2025.docx"，
    所以子串匹配是必需的；但子串匹配可能命中多份，这时候要如实报歧义，
    不能随便挑一份 —— 挑错了就是答错文档。
    """
    key = (name or "").strip().lower()
    if not key:
        return []
    exact = [r for r in rows if (r.filename or "").lower() == key]
    if exact:
        return exact
    return [r for r in rows if key in (r.filename or "").lower()]


def _load_rows() -> tuple[list | None, str | None]:
    """取全部文档元数据。返回 (rows, 错误话术)；出错时 rows 为 None。"""
    try:
        return _document_store().all(), None
    except MySQLUnavailable as e:
        logger.warning("文档元数据不可用（工具降级）: %s", e)
        return None, "文档元数据暂不可用（MySQL 未就绪），无法按文档名定位。"
    except Exception as e:  # noqa: BLE001
        logger.warning("文档元数据读取异常（工具降级）: %s", e)
        return None, "文档元数据读取失败，请稍后重试。"


def _resolve_document(filename: str):
    """把用户给的文件名解析成唯一一份文档。返回 (row, 错误话术)。"""
    rows, err = _load_rows()
    if rows is None:
        return None, err

    hits = _match_documents(rows, filename)
    if not hits:
        return None, (
            f"知识库里没有名为「{filename}」的文档。"
            "可以先用 list_documents 查看现有文档名。"
        )
    if len(hits) > 1:
        names = "、".join(r.filename for r in hits[:8])
        return None, (
            f"「{filename}」匹配到多份文档，请让用户确认是哪一份：{names}"
        )
    return hits[0], None


# ══════════════════════════════════════════════════════════════
# 工具
# ══════════════════════════════════════════════════════════════


@tool
def search_knowledge_base(query: str, top_k: int = 5) -> str:
    """检索企业知识库，回答产品、售后政策、安装、故障、备件、服务网点等业务问题。

    使用场景：
    - 用户询问产品参数、保修与延保、退换货流程、故障排查、备件价格、服务网点等
    - 答案可能落在哪份文档不确定时，用这个工具做**全库**检索

    不适用场景：
    - 用户点名了具体文档（"《售后服务政策》里怎么写的"）→ 用 search_in_document，
      全库检索容易被其他同族/相似文档干扰
    - 用户问"知识库里有哪些资料"→ 用 list_documents
    - 与业务无关的问题（闲聊、写代码、翻译）不要调用

    Args:
        query: 检索用的用户问题或关键词，尽量保留商品名、型号、单据号等关键信息
        top_k: 返回片段数，默认 5，上限 10

    返回：带出处（文件名 ｜ 章节 ｜ 页码）的资料片段；库不可用或未命中会明确说明。
    """
    k = _clamp_top_k(top_k)
    try:
        docs = retrieve_sync(query, k)
        if docs:
            docs = reordering(query, docs)
    except KnowledgeBaseError as e:
        logger.warning("知识库检索工具降级: %s", e)
        return f"知识库当前不可用（{e}），请稍后重试，或建议用户转人工。"
    except Exception as e:  # noqa: BLE001
        logger.warning("知识库检索工具异常: %s", e)
        return "知识库检索失败，请稍后重试。"

    rendered = _render(docs or [])
    if not rendered:
        return (
            "知识库中未检索到与这个问题相关的内容。"
            "可以换个说法再查，或用 list_documents 确认库里有没有相关资料。"
        )
    return f"检索到以下资料（只依据这些内容回答，并标注来源）：\n\n{rendered}"


@tool
def search_in_document(query: str, filename: str, top_k: int = 5) -> str:
    """在某一份指定文档的内部检索，适合用户已经点名了文档的场景。

    使用场景：
    - 用户说"《售后服务政策》里保修条款怎么写的"、"备件总表里华东仓的编码是多少"
    - 已知答案就在某份文档里，全库检索会被其他相似文档干扰时

    不适用场景：
    - 用户没说限定在哪份文档 → 用 search_knowledge_base
    - 不确定库里有哪份文档 → 先用 list_documents 查文件名

    Args:
        query: 要在这份文档里找什么
        filename: 文档文件名，如 "售后服务政策_v2_2025.docx"。
            只需文件名，不需要路径；记不全时给关键片段即可（如"售后服务政策"）
        top_k: 返回片段数，默认 5，上限 10

    返回：该文档内命中的片段；文件名不存在、匹配到多份、解析失败或未命中都会明确说明。
    """
    k = _clamp_top_k(top_k)
    target, err = _resolve_document(filename)
    if target is None:
        return err

    if target.parsed_status != "ok":
        return (
            f"「{target.filename}」解析失败，内容没有入库，因此无法检索"
            f"（原因：{target.error_msg or '未知'}）。"
        )

    try:
        store = _ready_store()
        docs = store.similarity_search(query, k, filter={"source": target.source})
    except KnowledgeBaseError as e:
        logger.warning("限定文档检索工具降级: %s", e)
        return f"知识库当前不可用（{e}），请稍后重试。"
    except Exception as e:  # noqa: BLE001
        logger.warning("限定文档检索工具异常: %s", e)
        return "检索失败，请稍后重试。"

    rendered = _render(docs or [])
    if not rendered:
        return f"在「{target.filename}」里没有检索到与这个问题相关的内容。"
    return (
        f"在「{target.filename}」里检索到以下内容"
        f"（只依据这些内容回答，并标注来源）：\n\n{rendered}"
    )


@tool
def list_documents(keyword: str = "") -> str:
    """列出知识库里已入库的文档清单（文件名、块数、解析状态）。

    使用场景：
    - 用户问"知识库里有哪些资料"、"有没有关于备件的文档"
    - **检索没结果时先用它**，用来区分"库里根本没这份资料"和"检索没命中"——
      这两种情况该回给用户的话完全不同

    不适用场景：
    - 问具体业务内容 → 用 search_knowledge_base

    Args:
        keyword: 可选，按文件名过滤，例如 "备件"、"政策"；留空则列出全部

    返回：文档清单；解析失败的文档会单独列出原因。
    """
    rows, err = _load_rows()
    if rows is None:
        return f"{err}当前无法枚举知识库文档。"

    kw = (keyword or "").strip().lower()
    if kw:
        rows = [r for r in rows if kw in (r.filename or "").lower()]
        if not rows:
            return f"没有文件名包含「{keyword}」的文档。"

    if not rows:
        return "知识库中还没有已入库的文档，请先上传资料。"

    ok = [r for r in rows if r.parsed_status == "ok"]
    bad = [r for r in rows if r.parsed_status != "ok"]

    lines = [f"知识库共 {len(rows)} 份文档（{len(ok)} 份可用，{len(bad)} 份解析失败）："]
    lines += [f"- {r.filename}（{r.chunk_count} 块）" for r in ok[:_MAX_LISTED]]
    if len(ok) > _MAX_LISTED:
        lines.append(f"- …另有 {len(ok) - _MAX_LISTED} 份未列出")
    if bad:
        lines.append("以下文档解析失败、内容未入库：")
        lines += [
            f"- {r.filename}：{r.error_msg or '原因未知'}" for r in bad[:_MAX_LISTED]
        ]
    return "\n".join(lines)


@tool
def get_document_outline(filename: str) -> str:
    """列出某一份文档的章节结构，用来告诉用户"这份文档讲了什么"。

    使用场景：
    - 用户问"《延保服务说明》里都写了什么"、"这份手册有哪些章节"
    - 先给用户一个目录，再让他挑具体想了解的部分

    不适用场景：
    - 问具体条款内容 → 用 search_in_document
    - 不确定文档名 → 先用 list_documents

    Args:
        filename: 文档文件名，只需文件名，不需要路径；记不全时给关键片段即可

    返回：章节标题列表；文件不存在、无章节结构或父块库不可用时会明确说明。
    """
    target, err = _resolve_document(filename)
    if target is None:
        return err

    try:
        outline = _parent_outline(target.doc_id)
    except MySQLUnavailable as e:
        logger.warning("章节结构工具降级: %s", e)
        return (
            "章节信息暂不可用（父块库 MySQL 未就绪），"
            "请改用 search_in_document 直接在这份文档里检索。"
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("章节结构工具异常: %s", e)
        return "章节信息读取失败，请稍后重试。"

    if not outline:
        return (
            f"「{target.filename}」没有记录到章节结构（可能是纯文本或无标题文档），"
            f"共 {target.chunk_count} 块。建议直接用 search_in_document 检索。"
        )

    lines = [f"「{target.filename}」共 {target.chunk_count} 块，章节结构："]
    lines += [f"- {s}" for s in outline[:_MAX_SECTIONS]]
    if len(outline) > _MAX_SECTIONS:
        lines.append(f"- …另有 {len(outline) - _MAX_SECTIONS} 个章节未列出")
    return "\n".join(lines)


# 供注册处引用：**唯一真源**，避免两处 agent 各写一份工具列表而漂移
KB_TOOLS = [
    search_knowledge_base,
    search_in_document,
    list_documents,
    get_document_outline,
]
