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

# ════════════════════════════════════════════════════════════
# Embedding
#   EMBEDDING_PROVIDER=dashscope  云端 text-embedding-v2（1536 维，按量付费）
#   EMBEDDING_PROVIDER=local      本地 sentence-transformers（离线、免费）
# 换 provider/模型 = 换了向量空间（维度也不同）→ **必须同时换 Chroma 集合名并全量重建**。
# ════════════════════════════════════════════════════════════
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
EMBEDDING_MODEL_PATH = os.getenv(
    "EMBEDDING_MODEL_PATH", str(ROOT / "embbding_models" / "bge-small-zh-v1.5")
)
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
# bge-zh 系要求 query 侧加指令、document 侧不加（加错边不报错，只会悄悄掉召回）
EMBEDDING_QUERY_PREFIX = os.getenv(
    "EMBEDDING_QUERY_PREFIX", "为这个句子生成表示以用于检索相关文章："
)

# ════════════════════════════════════════════════════════════
# Chroma：维度写死在集合里，所以换模型必须换集合名（旧集合留着可一键回滚）
# ════════════════════════════════════════════════════════════
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "knowledge_base_bge512")
CHROMA_SPACE = os.getenv("CHROMA_SPACE", "cosine")  # bge 是归一化余弦模型，别用默认 l2


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
