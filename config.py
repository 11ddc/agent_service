"""项目配置集中入口。

以前这里只有一行 REDIS_URL，其余配置散落在各模块的 os.getenv 里（同一个 DashScope key
甚至有三个名字）。新增 embedding / Chroma 配置时把能集中的集中过来。

⚠️ `HF_ENDPOINT` 必须在这里设置：huggingface_hub 的 endpoint 是 **import 时求值一次**的常量，
放到别的模块（例如 rag/local_reranker.py）里设置就太晚了 —— 那里的 import 链
（rag.structure → langchain_text_splitters → sentence_transformers → huggingface_hub）
早就把常量固化成了 huggingface.co（实测确认）。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ⚠️ `load_dotenv()` 必须在这里，且必须早于本文件任何一次 os.getenv —— 本文件在
# **import 时就把** REDIS_URL / EMBEDDING_* / CHROMA_* 读成模块常量了。以前这里没有
# 这行，能读到值纯属 import 顺序的巧合：入口链是 api/chat.py → agent.graph →
# agent/langchina 里的 load_dotenv 先跑了一步，而 redis_client → config 排在它后面。
# 换句话说，换任何入口（uvicorn api.chat:app、写脚本、跑测试）都会静默退回默认值
# （例如 REDIS_URL 变成 localhost:6379），不报错，只是历史与计数悄悄写到别处。
# load_dotenv 默认不覆盖已有环境变量，所以 tests/conftest.py 注入的占位 key 仍然优先。
load_dotenv(encoding="utf-8-sig")  # utf-8-sig:兼容带 BOM 的 .env

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # setdefault：不覆盖你显式设置的值

ROOT = Path(__file__).resolve().parent

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")


def _resolve_model_path(env_name: str, local_dir: Path, repo_id: str) -> str:
    """模型路径解析：**本地有就用本地，没有就退回 HuggingFace 仓库 ID**。

    为什么需要：仓库里不带模型权重（embbding_models/ 与 Reank_models/ 共 2.6GB，
    早已 gitignore）。以前这里的默认值写死成本地目录，克隆下来目录不存在 →
    SentenceTransformer 直接抛错，RAG 检索整条链挂掉。现在退回 HF 仓库 ID 后，
    sentence-transformers 会自己下载（走 config 顶部设置的 HF_ENDPOINT 镜像），
    别人 clone 下来**不改一行配置**就能跑。

    显式设了环境变量就完全听环境变量的（不再做"存在与否"的判断）——否则用户
    指向一个自己还没创建好的目录时会被静默改写成 HF ID，反而更难排查。
    """
    explicit = (os.getenv(env_name) or "").strip()
    if explicit:
        return explicit
    return str(local_dir) if local_dir.exists() else repo_id


# ════════════════════════════════════════════════════════════
# Embedding
#   EMBEDDING_PROVIDER=dashscope  云端 text-embedding-v2（1536 维，按量付费）
#   EMBEDDING_PROVIDER=local      本地 sentence-transformers（离线、免费）
# 换 provider/模型 = 换了向量空间（维度也不同）→ **必须同时换 Chroma 集合名并全量重建**。
# ════════════════════════════════════════════════════════════
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
EMBEDDING_MODEL_PATH = _resolve_model_path(
    "EMBEDDING_MODEL_PATH",
    ROOT / "embbding_models" / "bge-small-zh-v1.5",
    "BAAI/bge-small-zh-v1.5",
)
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
# bge-zh 系要求 query 侧加指令、document 侧不加（加错边不报错，只会悄悄掉召回）
EMBEDDING_QUERY_PREFIX = os.getenv(
    "EMBEDDING_QUERY_PREFIX", "为这个句子生成表示以用于检索相关文章："
)

# ════════════════════════════════════════════════════════════
# Cross-Encoder 精排
# 以前模型路径**硬编码**在 rag/local_reranker.py 里写死成作者的盘符
# （"F:/my-agent-api/Reank_models/..."）—— 别人 clone 到别的目录必然加载失败，
# 而 reordering() 里那层 except 会把它降级成 RRF 顺序，**不报错、只是精排悄悄失效**
# （README 里 R@1 84.8% 的那档就没了）。现在统一走 config + 路径不存在时退回 HF 仓库 ID。
# ════════════════════════════════════════════════════════════
RERANK_MODEL_PATH = _resolve_model_path(
    "RERANK_MODEL_PATH",
    ROOT / "Reank_models" / "bge-reranker-v2-m3",
    "BAAI/bge-reranker-v2-m3",
)
RERANK_DEVICE = os.getenv("RERANK_DEVICE", "cuda")  # 不可用时自动退 cpu，见 local_reranker.py

# ════════════════════════════════════════════════════════════
# Chroma：维度写死在集合里，所以换模型必须换集合名（旧集合留着可一键回滚）
# ════════════════════════════════════════════════════════════
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "knowledge_base_bge512")
CHROMA_SPACE = os.getenv("CHROMA_SPACE", "cosine")  # bge 是归一化余弦模型，别用默认 l2

# ════════════════════════════════════════════════════════════
# OCR（可选）：PDF 扫描页 / 文档内嵌图片走 pytesseract
# 以前路径硬编码成作者机器上的 "C:\Program Files\..."，Linux/macOS 上必然指向一个
# 不存在的文件。现在的解析顺序：TESSERACT_CMD > Windows 默认安装路径（确实存在才用）
# > 留空交给 pytesseract 自己按 PATH 找（Linux/macOS 装完 tesseract 即开箱可用）。
# ════════════════════════════════════════════════════════════
def _resolve_tesseract_cmd() -> str:
    explicit = (os.getenv("TESSERACT_CMD") or "").strip()
    if explicit:
        return explicit
    if os.name == "nt":
        default = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if default.exists():
            return str(default)
    return ""  # 空 = 让 pytesseract 走 PATH


TESSERACT_CMD = _resolve_tesseract_cmd()


def embedding_stamp() -> str:
    """缓存/索引标签：provider + 模型名。换模型后缓存自动失效。

    为什么需要：query 向量缓存原来按 query 文本存，换 embedding 模型后不清缓存就会拿
    **旧模型的向量**去查新索引 —— 要么报错，要么指标全错（静默劣化里最阴的一种）。
    """
    name = (
        Path(EMBEDDING_MODEL_PATH).name
        if EMBEDDING_PROVIDER == "local"
        else "text-embedding-v2"
    )
    return f"{EMBEDDING_PROVIDER}-{name}"
