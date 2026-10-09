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
# Cross-Encoder 精排（**当前未启用**）
# 现在线上走的是智谱 glm-4.5-air 打分，见 rag/glm_reranker.py；下面这两项只被
# 已注释的本地实现（rag/local_reranker.py）使用，留着是为了随时能换回去。
#
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


# ════════════════════════════════════════════════════════════
# 认证与授权
#
# 企业里"谁在用、他能看什么"是红线，所以这块的姿态是**失败关闭**：
#   · AUTH_ENABLED 默认开启；缺 AUTH_JWT_SECRET 时**不会**启动失败，而是签/验令牌
#     那一刻抛 AuthConfigError（见 auth/security.py:116）—— 服务看起来正常，但登录
#     与所有鉴权接口全废。（main.py 里并没有启动自检，别指望它把缺的 key 列出来。）
#     绝不使用内置默认密钥：那等于谁都能伪造 token；
#   · 用户存储（MySQL）不可用时，受保护接口返回 **503**，绝不放行；
#   · AUTH_ENABLED=false 只给本地演示用，启动时会打 WARNING。
#
# ⚠️ 副作用要清楚：开启认证之后 **MySQL 从"建议"变成"必需"** ——
#    账号体系存在 MySQL 里，没有它就无法登录（这是刻意的失败关闭）。
# ════════════════════════════════════════════════════════════


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    return default if not raw else raw not in {"0", "false", "off", "no"}


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


AUTH_ENABLED = _env_bool("AUTH_ENABLED", True)
AUTH_JWT_SECRET = (os.getenv("AUTH_JWT_SECRET") or "").strip()
AUTH_ACCESS_TTL = _env_int("AUTH_ACCESS_TTL_SECONDS", 15 * 60)  # 15 分钟
AUTH_REFRESH_TTL = _env_int("AUTH_REFRESH_TTL_SECONDS", 30 * 24 * 3600)  # 30 天
AUTH_BCRYPT_ROUNDS = _env_int("AUTH_BCRYPT_ROUNDS", 12)
AUTH_ALLOW_REGISTRATION = _env_bool("AUTH_ALLOW_REGISTRATION", True)
AUTH_DEFAULT_ROLE = (os.getenv("AUTH_DEFAULT_ROLE") or "user").strip().lower()
AUTH_DEFAULT_TENANT = (os.getenv("AUTH_DEFAULT_TENANT") or "default").strip()
AUTH_MAX_FAILED_LOGINS = _env_int("AUTH_MAX_FAILED_LOGINS", 5)
AUTH_LOCK_SECONDS = _env_int("AUTH_LOCK_SECONDS", 300)
AUTH_MIN_PASSWORD_LEN = _env_int("AUTH_MIN_PASSWORD_LEN", 8)

# 只有在**可信反向代理之后**才打开。
# X-Forwarded-For 是客户端可伪造的：打开它等于让调用方自己决定"被记成哪个 IP"，
# 而 IP 会进审计日志与风控判断。
TRUST_PROXY_HEADERS = _env_bool("TRUST_PROXY_HEADERS", False)

# ════════════════════════════════════════════════════════════
# 请求入口限流
# 目的不是"防 DDoS"（那该在网关/WAF 做），而是**别让一个客户端把共享的
# 线程池与模型算力吃光** —— 一次问答是几秒到几十秒的重活。
# ════════════════════════════════════════════════════════════
RATE_LIMIT_ENABLED = _env_bool("RATE_LIMIT_ENABLED", True)
RATE_LIMIT_PER_MINUTE = _env_int("RATE_LIMIT_PER_MINUTE", 30)  # 每用户每分钟
RATE_LIMIT_BURST = _env_int("RATE_LIMIT_BURST", 10)  # 允许的瞬时突发

# ════════════════════════════════════════════════════════════
# 知识库 ACL 与审核发布
#
# 企业知识库里既有"全员可见的手册"，也有"只有售后能看的内部判责标准"。
# 检索不做权限过滤 = 把内部文档发给所有用户，所以 ACL 默认**开启**。
#
# ⚠️ 开启后，**缺少 ACL 元数据的历史块会被判为不可见**（失败方向是"拒绝"），
#    存量库需要跑一次回填：python -m rag.acl_backfill
#    ACL_ENABLED=false 会退回"不过滤"，只应出现在本地演示。
# ════════════════════════════════════════════════════════════
ACL_ENABLED = _env_bool("ACL_ENABLED", True)
ACL_DEFAULT_VISIBILITY = (
    os.getenv("ACL_DEFAULT_VISIBILITY") or "tenant"
).strip().lower()

# 上传后是否需要审核才能被检索到。
# true（企业默认）：上传 → draft（检索不到）→ 审核发布 → published
# false（本地图省事）：上传即 published
ACL_REQUIRE_APPROVAL = _env_bool("ACL_REQUIRE_APPROVAL", True)

# ════════════════════════════════════════════════════════════
# 可观测性
# ════════════════════════════════════════════════════════════
LOG_LEVEL = (os.getenv("LOG_LEVEL") or "INFO").strip().upper()

# 应用版本：/health 与 OpenAPI 文档都用它，避免版本号散落各处（此前
# main.py 写 1.0.0、pyproject 写 0.1.0，两边对不上）
APP_VERSION = (os.getenv("APP_VERSION") or "0.2.0").strip()

# /metrics 是否要求管理员身份。默认要求 —— 指标会暴露业务量级
# （咨询量、转人工率、成本），不该对匿名访问者开放。
# 集群内用 Prometheus 抓取时：要么配 bearer token，要么设成 0 并只在内部网暴露。
METRICS_REQUIRE_AUTH = _env_bool("METRICS_REQUIRE_AUTH", True)

# ════════════════════════════════════════════════════════════
# 内容审核与日志脱敏
#
# 两件事分开：**违规内容拦截**（黑名单/外部服务）与 **PII 脱敏**（日志/审计）。
# 客服场景里用户主动给手机号、订单号是正常的，把 PII 当违规拦掉会把最需要
# 人工帮助的用户挡在门外。
# ════════════════════════════════════════════════════════════
MODERATION_ENABLED = _env_bool("MODERATION_ENABLED", True)
# rule（离线规则，默认）| http（外部内容安全服务，需出网，默认未接实现）
MODERATION_PROVIDER = (os.getenv("MODERATION_PROVIDER") or "rule").strip().lower()
# 额外的禁用词，逗号分隔。留空则用 moderation.py 里的少量内置样例。
MODERATION_TERMS = os.getenv("MODERATION_TERMS") or ""
# 审核组件出错时：true=放行（默认，审核故障不该让客服停摆），false=拦截
MODERATION_FAIL_OPEN = _env_bool("MODERATION_FAIL_OPEN", True)
# 单次审核扫描的最大字符数（限制成本；文档超长时只审前 N 字）
MODERATION_MAX_CHARS = _env_int("MODERATION_MAX_CHARS", 20000)
# 日志里的手机号/身份证/银行卡/邮箱是否打码
MODERATION_MASK_LOGS = _env_bool("MODERATION_MASK_LOGS", True)

# ════════════════════════════════════════════════════════════
# 会话记忆（LangGraph checkpointer）与 BM25 索引的外置
#
# 默认 memory = 进程内：**重启即丢、多副本各存一份**。这不是"已外置"，
# 只是一个零依赖的默认值；企业部署请显式改成 sqlite / postgres
# （需要额外安装 langgraph-checkpoint-* 包，见 agent/checkpoint.py）。
# ════════════════════════════════════════════════════════════
AGENT_CHECKPOINT_BACKEND = (os.getenv("AGENT_CHECKPOINT_BACKEND") or "memory").strip().lower()
AGENT_CHECKPOINT_SQLITE_PATH = Path(
    os.getenv("AGENT_CHECKPOINT_SQLITE_PATH") or (ROOT / "data" / "checkpoints.sqlite")
)
AGENT_CHECKPOINT_POSTGRES_URL = os.getenv("AGENT_CHECKPOINT_POSTGRES_URL") or ""
