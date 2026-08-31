"""
RAG 检索模块 —— 只负责文档加载、向量化、检索。
生成回答由 Agent 的 LLM 负责，本模块不创建 LLM 实例。
"""

import asyncio
import os
import re
import threading
from pathlib import Path

import jieba

# import pymupdf4llm
import pytesseract
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document

# from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image
from rank_bm25 import BM25Okapi

from rag.local_reranker import LocalReranker

# 清楚 chromadb 的缓存，避免报错
# import chromadb.api.shared_system_client as shared
# shared.SharedSystemClient._identifier_to_system.clear()
# import pymupdf4llm
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

load_dotenv()

reranker = LocalReranker()


# ── 知识库异常 ─────────────────────────────────────────────
class KnowledgeBaseError(Exception):
    """知识库不可用（连接失败 / 未初始化 / 为空）。由调用方捕获后转成提示语。"""


# 知识库参数
_vectorstore = None
_retriever = None

# BM25 关键词检索参数（内存索引，重启后懒加载）
_bm25 = None  # BM25Okapi 索引
_bm25_corpus: list[str] = []  # 分词后的语料（与 _chunk_docs 对齐）
_chunk_docs: list[Document] = []  # 全量 chunk（含 metadata，用于映射回来源）

# 文件内容加载参数
_is_initialized: bool = False

# 初始化/重建的互斥锁（防止并发首检双重初始化）
_init_lock = threading.Lock()
_bm25_lock = threading.Lock()

# BM25 检索前统一清洗的标点（全角+空白），保证 query 与语料分词对齐
_PUNCT_RE = re.compile(r"[，。！？、；：‘’“”（）《》【】\s]+")

# ── 可调参数 ─────────────────────────────────────────────
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
# rrf使用
TOP_K = 20
# 重排使用
TOP_N = 4

# 知识库默认目录：项目根目录下的 knowledge_base/
KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"
# 持久化目录
PERSIST_DIR = Path(__file__).resolve().parent.parent / "chroma_db"

IMAGES_DIR = Path(__file__).resolve().parent.parent / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)  # 确保目录存在，否则 pymupdf4llm 会报错


# 提取图片路径
def find_image_references(content: str) -> list[str]:
    """
    从文本内容中提取所有 Markdown 图片引用的文件名。
    例如：![alt text](images/image1.png) -> 提取 image1.png
    """
    import re

    pattern = r"!\[.*?\]\((.*?)\)"
    matches = re.findall(pattern, content)
    print(f"提取到的图片引用: {matches}")
    # 只保留文件名部分
    return [os.path.basename(match) for match in matches]


# ==================== 文档加载 ============================
def _load_pdf(file_path: str) -> list[Document]:
    # 提取 Markdown（图片保存到绝对路径 IMAGES_DIR）
    import pymupdf4llm
    import pytesseract
    from PIL import Image

    pages = pymupdf4llm.to_markdown(
        file_path,
        page_chunks=True,
        write_images=True,
        image_path=str(IMAGES_DIR),  # ① 写入：绝对路径，与 CWD 无关
        image_format="png",
        dpi=150,
    )

    docs = []
    for chunk in pages:
        content = chunk.get("text", "")

        # ② 顺手把 Markdown 里的图片引用归一化成纯文件名，
        #    否则绝对路径会被存进 Chroma 的内容里，污染检索结果
        content = re.sub(
            r"!\[([^\]]*)\]\(([^)]+)\)",
            lambda m: f"![{m.group(1)}]({os.path.basename(m.group(2))})",
            content,
        )

        for img_ref in find_image_references(content):
            img_path = IMAGES_DIR / img_ref  # ③ 读取：绝对路径，与 CWD 无关
            if img_path.exists():
                ocr_text = pytesseract.image_to_string(
                    Image.open(img_path), lang="chi_sim+eng"
                )
                print(f"OCR 识别图片 {img_ref} 的文字: {ocr_text}")
                if ocr_text.strip():
                    content += f"\n\n[图片内容]: {ocr_text.strip()}"

        docs.append(
            Document(
                page_content=content.strip(),
                metadata={
                    "source": str(file_path),
                    "page": chunk.get("metadata", {}).get("page", 0),
                },
            )
        )
    return docs


def _load_txt(file_path: str) -> list[Document]:
    """读取纯文本文件"""
    text = Path(file_path).read_text(encoding="utf-8")
    if text.strip():
        return [
            Document(page_content=text.strip(), metadata={"source": str(file_path)})
        ]
    return []


def _load_docx(file_path: str, outputimages: str = "docx_images") -> list[Document]:
    """从 Word 文档提取文本、表格，并对文档中的图片做 OCR（参考 _load_pdf 的 OCR 逻辑）"""
    from docx import Document as DocxDocument

    file_path = Path(file_path)
    # 文档的加载对象
    doc = DocxDocument(file_path)

    # 1. 提取段落文本
    content = []
    for para in doc.paragraphs:
        # strip 移除字符串开头和结尾的空白字符
        text = para.text.strip()
        if text:
            content.append(text)

    # 提取表格
    for table in doc.tables:
        if not table.rows:
            continue
        rows = []
        for row in table.rows:
            # 表格数据
            cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
            rows.append("| " + " | ".join(cells) + " |")

        if rows:
            # 计算第一行的数据看有多少列
            col_count = len(table.rows[0].cells)
            # 生成分隔符
            separator = "|" + " --- |" * col_count
            table_md = "\n".join([rows[0], separator] + rows[1:])
            content.append(table_md)

    # 3. 提取图片 + OCR（参考 _load_pdf：落盘到 IMAGES_DIR 后交给 pytesseract）
    ocr_parts = _extract_and_ocr_docx_images(doc, file_path, outputimages)
    if ocr_parts:
        content.append("\n\n".join(ocr_parts))

    if not content:
        return []

    return [
        Document(page_content="\n\n".join(content), metadata={"source": str(file_path)})
    ]


def _collect_docx_image_parts(doc):
    """收集 docx 中的所有图片（内嵌 + 浮动），返回去重后的 image part 列表。

    注：python-docx 的 doc.inline_shapes 只能拿到内嵌图片，
    这里直接遍历文档 XML 中的 a:blip 节点，内嵌/浮动图片都能覆盖。
    """
    from docx.oxml.ns import qn

    seen = set()
    parts = []
    # print(f"文档对象的函数：",doc.element)
    for blip in doc.element.body.iter(qn("a:blip")):
        r_id = blip.get(qn("r:embed"))
        if not r_id or r_id in seen:
            continue
        seen.add(r_id)
        try:
            parts.append(doc.part.related_parts[r_id])
        except KeyError:
            continue  # 引用失效的图片直接跳过
    return parts


def _guess_image_ext(content_type: str, blob: bytes) -> str:
    """根据 content_type 推断图片后缀，推断不了再用文件头魔数兜底"""
    mapping = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "image/tiff": ".tiff",
        "image/bmp": ".bmp",
    }
    ext = mapping.get((content_type or "").strip().lower())
    if ext:
        return ext
    # 文件头魔数兜底
    if blob[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if blob[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if blob[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    if blob[:4] == b"RIFF" and blob[8:12] == b"WEBP":
        return ".webp"
    if blob[:4] in (b"II*\x00", b"MM\x00*"):
        return ".tiff"
    if blob[:4] == b"BM":
        return ".bmp"
    return ".png"  # 无法识别时默认 png


def _extract_and_ocr_docx_images(doc, file_path: Path, outputimages: str) -> list[str]:
    """把 docx 里的图片落盘到 images/<outputimages>/ 并 OCR，返回识别出的文本片段。"""
    # 返回嵌在word文档里的图片对象
    image_parts = _collect_docx_image_parts(doc)
    if not image_parts:
        return []

    # 图片保存目录：基于项目根目录的绝对路径，不依赖启动时的 CWD
    images_dir = IMAGES_DIR / outputimages
    images_dir.mkdir(parents=True, exist_ok=True)

    ocr_parts = []
    for i, part in enumerate(image_parts, 1):
        blob = part.blob
        # 返回图片类型
        ext = _guess_image_ext(getattr(part, "content_type", ""), blob)
        img_path = images_dir / f"{file_path.stem}_img_{i}{ext}"
        # print(f"存储的路径",img_path)
        # 相当于open(img_path, 'wb').write(blob)
        img_path.write_bytes(blob)

        try:
            text = pytesseract.image_to_string(Image.open(img_path), lang="chi_sim+eng")
        except Exception as e:
            print(f"OCR 识别图片 {img_path.name} 失败: {e}")
            continue

        if text.strip():
            print(f"OCR 识别图片 {img_path.name} 的文字: {text.strip()}")
            ocr_parts.append(
                f"![图片 {i}]({img_path.name})\n[图片内容]: {text.strip()}"
            )

    return ocr_parts


def _load_file(file_path: str) -> list[Document]:
    """根据文件后缀加载单个文档"""
    file_path_obj = Path(file_path)
    suffix = file_path_obj.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(str(file_path))
    if suffix == ".txt" or suffix == ".md":
        return _load_txt(str(file_path))
    if suffix == ".docx":
        return _load_docx(str(file_path))
    return []


# def _load_documents(directory: Path) -> List[Document]:
#     """加载目录下所有支持的文档"""
#     all_docs: List[Document] = []
#     if not directory.exists():
#         return all_docs
#     # sorted 保证读取文件顺序一致
#     for file_path in sorted(directory.iterdir()):
#         # extend 将列表中的元素逐一添加到 all_docs 中，而不是将整个列表作为一个元素添加
#         all_docs.extend(_load_file(file_path))
#     return all_docs

# def get_chunk_id(chunk: Document) -> str:
#     """基于文档内容生成唯一 ID（SHA-256）"""
#     content = chunk.page_content
#     # 可以加入元数据如文件名来避免跨文件相同内容被误判，但为了彻底去重，仅内容足矣
#     return hashlib.sha256(content.encode('utf-8')).hexdigest()


def init_rag(file_path: str) -> int:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    raw_docs = _load_file(file_path)  # 只加载新上传的文件
    if not raw_docs:
        print(f"文件 {file_path} 未加载到任何文档")
        return 0

    # 只加载并切分新文件
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(raw_docs)

    embeddings = DashScopeEmbeddings(
        model="text-embedding-v2", dashscope_api_key=os.getenv("QIANWEN_API_KEY")
    )

    store = Chroma(
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR),
        collection_name="knowledge_base",
    )

    # 按文件来源去重：同一个文件重新上传时，先清除它已入库的旧块，再重新写入
    # 这里是根据路径判断，如果路径稍微有点不一样都不会覆盖
    # before = store._collection.count()
    store.delete(where={"source": file_path})
    # after = store._collection.count()
    # print(f"删除 {before - after} 块，剩余 {after} 块")

    # 向量化 + 持久化入库
    if chunks:
        store.add_documents(chunks)
        print(f"文件 {file_path} 入库 {len(chunks)} 个文档块")
    else:
        print(f"文件 {file_path} 未切分出任何文档块")

    return len(chunks)


# ==================== 多路召回（dense + BM25 → RRF 融合）====================


def _clean_query(text: str) -> str:
    """去掉中文标点和空白，保证 BM25 分词对齐（query 与语料统一清洗）"""
    return _PUNCT_RE.sub("", text or "")


def _ensure_ready() -> None:
    """懒连接持久化的 Chroma 知识库（不调用 init_rag，纯读取侧初始化）。

    - 幂等 + 线程安全：并发首次调用只会初始化一次
    - 上传走 init_rag 写库，这里每次检索都从持久化目录读最新状态
    """
    global _vectorstore, _retriever, _is_initialized
    if _vectorstore is not None:
        return
    # 获取锁
    with _init_lock:
        if _vectorstore is not None:
            return
        try:
            embeddings = DashScopeEmbeddings(
                model="text-embedding-v2",
                dashscope_api_key=os.getenv("QIANWEN_API_KEY"),
            )
            store = Chroma(
                embedding_function=embeddings,
                persist_directory=str(PERSIST_DIR),
                collection_name="knowledge_base",
            )
            _vectorstore = store
            # 数据库转为检索器
            _retriever = store.as_retriever(search_kwargs={"k": TOP_K})
            _is_initialized = True
            print("知识库连接成功（读取侧懒初始化）")
        except Exception as e:
            print(f"知识库连接失败: {e}")
            _vectorstore = None
            _retriever = None
            _is_initialized = False


def _build_bm25_index() -> None:
    """从 Chroma 全量拉取 chunk，构建 BM25 内存索引（幂等，线程安全）"""
    global _bm25, _bm25_corpus, _chunk_docs
    if _vectorstore is None:
        return
    with _bm25_lock:
        if _vectorstore is None:
            return
        try:
            # 拿到向量数据库中的文本和元数据（页码，来源）等
            data = _vectorstore._collection.get(include=["documents", "metadatas"])
            texts = data.get("documents") or []
            metas = data.get("metadatas") or []
            # print(f"texts:",texts)
            # print(f"metas:",metas)
            if not texts:
                _bm25, _bm25_corpus, _chunk_docs = None, [], []
                return
            _chunk_docs = [
                # Document将“文本”和“元数据”打包成一个对象。
                Document(page_content=t, metadata=m or {})
                for t, m in zip(texts, metas)
            ]
            # print(f"chunk_docs:::::",_chunk_docs)
            # jieba.lcut 将中文分词然后变成列表
            # 这里建立索引实际上是对之前存入到向量库中的切片chunk进行的
            _bm25_corpus = [jieba.lcut(_clean_query(t)) for t in texts]

            # print(f"_bm25_corpus：：",_bm25_corpus)
            _bm25 = BM25Okapi(_bm25_corpus)
            print("bm25222", _bm25)
            print(f"BM25 索引构建完成，共 {len(texts)} 个 chunk")
        except Exception as e:
            print(f"BM25 索引构建失败，降级为纯向量检索: {e}")
            _bm25, _bm25_corpus, _chunk_docs = None, [], []


def _sparse_search(query: str, k: int) -> list[Document]:
    """BM25 关键词路：jieba 分词 → 取分数最高的 k 个 chunk（0 分视为未命中）。

    BM25 是内存索引，上传新文档后通过 chunk 数量变化自动触发重建。
    """
    global _bm25, _chunk_docs
    if _vectorstore is None:
        return []
    try:
        # _chunk_docs   分词语料，用来和用户的问题分词之后进行对比
        # 文档有更新或者第一次拿到锁的用户才会触发建立索引
        print("bm25111", _bm25)
        if _bm25 is None or len(_chunk_docs) != _vectorstore._collection.count():
            _build_bm25_index()
            print("bm25333", _bm25)

    except Exception as e:
        print(f"BM25 索引一致性检查失败: {e}")
    if _bm25 is None or not _chunk_docs:
        return []

    # 将用户问题分词然后对每个文档进行打分，这里清洗用户问题要和建立索引时一致
    scores = _bm25.get_scores(jieba.lcut(_clean_query(query)))

    print("分数列表：scores", scores)

    # sorted从小到大排序，这里取负从大到小并取k个
    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]

    print("取前k个切片对应的分数从大到小：", ranked)
    # 根据切片分筛选   拿到命中的切片
    return [_chunk_docs[i] for i in ranked if scores[i] > 0]


def _rrf_fuse(
    dense_docs: list[Document],
    sparse_docs: list[Document],
    k: int = 60,
    top_n: int = TOP_K,
) -> list[Document]:
    """RRF 融合：score(d) = Σ 1/(k + rank)。page_content 去重，天然处理两路命中同一块。"""
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}
    for docs in (dense_docs, sparse_docs):
        for rank, doc in enumerate(docs, 1):
            key = doc.page_content
            doc_map.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_n]
    return [doc_map[key] for key, _ in ranked]


def _format_docs(docs: list[Document]) -> str:
    """把检索结果格式化为生成侧可读的上下文（带来源注脚）"""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        parts.append(f"[文档片段 {i} — 来源: {source}]\n{doc.page_content}")
    return "\n\n".join(parts)


# 检索
async def retrieve(query: str, k: int = TOP_K) -> list[Document]:
    """
    多路召回：dense（向量）+ sparse（BM25）两路异步并行检索，RRF 融合后返回。

    - 两路各自失败时降级为另一路（gather return_exceptions）
    - 不调用 init_rag（上传/入库由 init_rag 负责，这里只读持久化库）
    - 调用方必须是 async 环境；非 async 场景用 retrieve_sync

    参数:
        query: 查询文本（通常就是用户的问题）
        k: 返回的文档片段数

    返回:
        RRF 融合后的原始 Document 列表（未命中为空列表）。

    异常:
        KnowledgeBaseError: 知识库连接失败 / 未初始化 / 为空。
    """
    global _vectorstore

    if _vectorstore is None:
        # 初始化数据库并转为检索器
        _ensure_ready()
    if _vectorstore is None:
        raise KnowledgeBaseError("知识库连接失败，请检查后重试。")
    if _vectorstore._collection.count() == 0:
        raise KnowledgeBaseError("知识库为空，请先上传文档到知识库。")

    # 融合前每路多取一些，融合后再砍到 k
    # 粗筛，先筛选相近语义数量较多 然后在tpk
    fetch_k = max(k * 2, 6)

    # 两路召回：异步并行执行
    # 语义相似度搜索
    # 任务对象（Task） 下面这两条都没有运行只是创建对象
    dense_task = asyncio.to_thread(_vectorstore.similarity_search, query, fetch_k)
    # bm25检索
    # 返回命中的原文档数据_chunk_docs
    sparse_task = asyncio.to_thread(_sparse_search, query, fetch_k)
    # 开启两个线程运行
    # 因为dense_task，sparse_task 这两个在上面是创建任务然后等下面await gather两个一起运行完
    # 并发执行器gather
    # gather  如果传进去的是协程对象 则会调用create_task包装成协程任务  如果传进去的是任务 就相当于await 则直接返回该任务
    # gather  和 create_task会创建任务（并将协程对象放入到事件循环中等待await执行）
    # 如果不考虑并发以及一些情况，gather 和await差不多
    # 协程对象就是你调用一个 async def 函数时，返回的那个东西
    # create_task 只能接收协程对象（接收任务会报错，gather是两个都可以）创建task 放入事件循环 等待执行
    dense_docs, sparse_docs = await asyncio.gather(
        dense_task, sparse_task, return_exceptions=True
    )
    # print(f"dddd",dense_docs)
    # print(f"sp：：：",sparse_docs)

    # isinstance判断 这个数据是否是抛出的异常数据
    dense_ok = not isinstance(dense_docs, BaseException)
    sparse_ok = not isinstance(sparse_docs, BaseException)
    if not dense_ok:
        print(f"向量检索失败，降级为仅关键词检索: {dense_docs}")
        dense_docs = []
    if not sparse_ok:
        print(f"BM25 检索失败，降级为仅向量检索: {sparse_docs}")
        sparse_docs = []

    docs = _rrf_fuse(dense_docs, sparse_docs, top_n=k)
    if not docs:
        return []
    print("RRF融合成功：", docs)
    return docs


# 开启事件循环让里面的任务可以调度
def retrieve_sync(query: str, k: int = TOP_K) -> list[Document]:
    """同步版 retrieve：供 LangChain 工具等无事件循环的线程调用（agent 工具场景）。"""
    return asyncio.run(retrieve(query, k))


def get_status() -> dict:
    """
    返回知识库当前状态：是否已初始化、文档块数量、存储目录。

    供 Agent 工具（get_knowledge_base_status）和意图门控（status_query）使用。
    """
    try:
        store = Chroma(
            embedding_function=DashScopeEmbeddings(
                model="text-embedding-v2",
                dashscope_api_key=os.getenv("QIANWEN_API_KEY"),
            ),
            persist_directory=str(PERSIST_DIR),
            collection_name="knowledge_base",
        )
        print("知识库状态embedding模型")
        count = store._collection.count()
    except Exception as e:
        print(f"获取知识库状态失败: {e}")
        return {
            "initialized": False,
            "document_count": 0,
            "knowledge_base_dir": str(KNOWLEDGE_BASE_DIR),
            "error": str(e),
        }
    return {
        "initialized": True,
        "document_count": count,
        "knowledge_base_dir": str(KNOWLEDGE_BASE_DIR),
    }


# 重排序模型（直接调 DashScope text-rerank API，对齐官方 curl，绕开旧集成类硬编码模型/吞错误的问题）
class _DashScopeReranker:
    """薄封装：把官方 rerank curl 包成 compress_documents 接口，reordering() 调用处不变。
    - 按 index 回原列表取文档，保留 source/page 元数据
    """

    def __init__(self, model: str, top_n: int, api_key: str | None):
        self.model = model
        self.top_n = top_n
        self.api_key = api_key

    def compress_documents(
        self, documents: list[Document], query: str
    ) -> list[Document]:
        import dashscope

        resp = dashscope.TextReRank.call(
            model=self.model,
            query=query,
            documents=[d.page_content for d in documents],
            top_n=self.top_n,
            return_documents=True,  # 对齐官方 curl
            api_key=self.api_key,
        )
        # 官方 API：status_code==200 成功；失败时真实原因在 code/message 里
        if resp.status_code != 200:
            print(f"重排 API 失败: code={resp.code}, message={resp.message}")
            raise RuntimeError(f"DashScope rerank 失败: {resp.code} {resp.message}")

        # results 已按相关性从高到低排序，含 index；按 index 回原列表取文档
        return [documents[r["index"]] for r in resp.output.results]


# def get_reordering():
#     """返回重排器实例（模型默认 qwen3-rerank，可用 RERANK_MODEL 覆盖）。"""
#     reranker = _DashScopeReranker(
#         model="gte-rerank-v2", top_n=TOP_N, api_key=os.getenv("QWEN_RERANK_API_KEY")
#     )
#     print("重排模型准备：", reranker)
#     return reranker


# # 重排序
# def reordering(query: str, docs: list[Document]) -> list[Document]:
#     if not docs:
#         return docs
#     # 按相关性从高到低排好序的、新的文档列表  compress_documents文档压缩器 并返回topn条文档
#     try:
#         reranker = get_reordering().compress_documents(documents=docs, query=query)
#         print("重排序com之后：", reranker)

#         # 回填元数据（页码 注脚等）
#         orig_map = {d.page_content: d for d in docs}
#         return [orig_map.get(d.page_content, d) for d in reranker][:TOP_N]
#     except Exception as e:
#         print(f"重排失败，降级为 RRF 默认顺序: {e}")
#         return docs


# 调用本地重排序模型
def reordering(query: str, docs: list[Document]) -> list[Document]:
    if not docs:
        return docs
    if len(docs) == 1:
        return docs

    # 按相关性从高到低排好序的、新的文档列表  compress_documents文档压缩器 并返回topn条文档
    try:
        # 懒加载：reranker 内部首次调用时才加载模型，后续直接复用
        result = reranker.rerank(query, docs, TOP_N)

        # print(f"重排序com之后：",result)

        # 回填元数据（页码 注脚等）
        # 注意：要遍历 rerank 的返回值 result（重排后的文档列表），
        # 而不是 reranker 实例本身——后者不可迭代，会抛 TypeError 走降级分支
        orig_map = {d.page_content: d for d in docs}
        return [orig_map.get(d.page_content, d) for d in result][:TOP_N]
    except Exception as e:
        # 打印完整堆栈，避免静默吞错后重排悄悄失效
        import traceback

        traceback.print_exc()
        print(f"重排失败，降级为 RRF 默认顺序: {e}")
        return docs
