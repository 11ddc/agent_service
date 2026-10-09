# ════════════════════════════════════════════════════════════════════════
# 智能客服问答 API —— 生产部署镜像（本仓库唯一的 Dockerfile）
#
# 与 docker-compose.yml 搭配：
#     docker compose up -d --build
#     docker compose exec api python -m db.init_schema     # 建表 + 结构自检
# 单容器跑：
#     docker build -t my-agent-api:latest .
#     docker run --rm -p 8000:8000 --env-file .env my-agent-api:latest
#
# 设计取舍（每条都对应一个真实踩过的坑）：
#   1. 单进程 uvicorn（--workers 1）。进程内限流与 BM25 索引都是「每进程一份」，
#      多 worker 只会把限流额度放大 N 倍。要吞吐请横向加副本。
#   2. 非 root 运行（uid 10001）。
#   3. 模型 / 向量库 / 知识库**不打进镜像**，走卷挂载（见 .dockerignore 与 compose）。
#      镜像因此小一个数量级，改文档也不必重建镜像。
#   4. torch 固定 CPU 版。Linux 上 PyPI 的 torch 默认带 CUDA，会连带拖进约 2.5GB
#      的 nvidia-* 依赖（镜像 ~4GB → ~11GB），而本项目运行时只用 CPU。
#      ⚠️ 真要用 GPU：换掉 TORCH_INDEX_URL，并删掉下面那步 assert。
#   5. 构建期就校验依赖闭包与 torch 变体：requirements.txt 出过「代码 import 了、
#      清单里却没有」的问题，这类问题要在构建期暴露，而不是上线后。
#   6. 1 号进程用 tini：应用会用 stdio 拉起 MCP 子进程（mcp_client.py），
#      需要一个能回收僵尸进程、并把 SIGTERM 转发给 uvicorn 的 init。
#
# 首次部署的必做一步（跟非 root 的 uid 10001 绑定）：
#     compose 里 knowledge_base / chroma_db 是 **bind mount**，宿主机目录的属主不会
#     自动变成 appuser，而应用要往这两个目录写上传的文档和向量库。所以先：
#         mkdir -p knowledge_base chroma_db
#         sudo chown -R 10001:10001 knowledge_base chroma_db
#     否则首次上传/建库会 EACCES。命名卷（app_data / bm25_cache / hf_home）会继承
#     镜像里的属主，不需要额外处理。
#
# 国内 / 内网构建加速：三个源都由构建参数注入，**不用改本文件**，
# 在 .env 里写同名变量即可（docker-compose.yml 的 build.args 会传进来）：
#     APT_MIRROR=mirrors.aliyun.com
#     PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
#     TORCH_INDEX_URL=https://mirrors.aliyun.com/pytorch-wheels/cpu
#   （阿里云 ECS 走内网更快且不计公网流量：mirrors.cloud.aliyuncs.com）
#
# ⚠️ 本文件没有在本机构建验证过（本机 Docker daemon 未运行），构建期那几处
#    校验（torch 变体、依赖闭包）是刻意加的护栏，首次真机构建会替你把关。
# ════════════════════════════════════════════════════════════════════════

# 基础镜像显式钉住 Debian 代号：`python:3.12-slim` 会随 Debian 新版本漂移，
# 而代号变化会影响 apt 源文件格式（deb822）与中文字体 / 语言包的包名。
FROM python:3.12-slim-bookworm

# ── 构建期可覆盖的包源（默认官方源，不传参与原行为一致）────────────────
ARG APT_MIRROR=deb.debian.org
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/models/hf

LABEL org.opencontainers.image.title="my-agent-api" \
      org.opencontainers.image.description="智能客服问答 API —— RAG 检索 + LangGraph Agent 编排" \
      org.opencontainers.image.source="https://github.com/11ddc/agent_service"

# 系统依赖：
#   libgomp1                      —— torch / onnxruntime 的 OpenMP 运行时
#   tesseract-ocr + chi_sim + eng —— 扫描件 OCR（rag/rag.py 用 lang="chi_sim+eng"）
#   fonts-noto-cjk                —— OCR 与图片渲染的中文字体
#   curl                          —— HEALTHCHECK
#   tini                          —— 1 号 init（见文件头第 6 条）
#   ca-certificates               —— 调 LLM / 拉模型走 HTTPS
# Debian 12+ 的 apt 源是 deb822 格式（/etc/apt/sources.list.d/debian.sources），
# 更老的镜像才是 /etc/apt/sources.list —— 两种都处理。
RUN if [ "$APT_MIRROR" != "deb.debian.org" ]; then \
        if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
            sed -i "s|deb.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list.d/debian.sources; \
        else \
            sed -i "s|deb.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list; \
        fi; \
    fi \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        tesseract-ocr \
        tesseract-ocr-chi-sim \
        tesseract-ocr-eng \
        fonts-noto-cjk \
        curl \
        tini \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 运行用户与运行时目录：先建好，后面的 COPY 直接 --chown，省掉一次
# 「把整个 /app 复制进新层」的 chown -R。这些目录在 compose 里会被卷挂载覆盖，
# 命名卷会继承镜像里的属主，所以属主必须在这里就设对（bind mount 不会继承）。
#   images / docx_images —— rag/rag.py 解析 PDF 内嵌图与 DOCX 图片的落盘目录
#                           （代码自己会 mkdir，这里建出来是为了属主明确：/app 归属
#                           appuser，将来给它们挂命名卷时也不会变成 root 属主）
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/knowledge_base /app/chroma_db /app/bm25_cache /app/models/hf \
                /app/data /app/logs /app/images /app/docx_images \
    && chown -R appuser:appuser /app

# ── 1) 先单独装 CPU 版 torch ─────────────────────────────────────────────
# requirements.txt 不含 torch，但 sentence-transformers 依赖它。若直接
# `pip install -r requirements.txt`，Linux 上 pip 会从 PyPI 解析出 **CUDA 版** torch
# （554MB wheel + 约 2GB nvidia-*）。先从 CPU 索引把 torch 装好，下一步解析时看到
# `torch>=...` 已满足就不会再动它。CPU 索引自带 torch 的全部依赖（sympy/networkx/
# filelock/...），所以不需要 --extra-index-url，也就不存在「被 PyPI 上的高版本顶回
# CUDA 版」的风险。
# 末尾的 assert 是构建期护栏：一旦装成 CUDA 版就直接构建失败，而不是悄悄产出一个 ~11GB 镜像。
RUN pip install --index-url "${TORCH_INDEX_URL}" torch \
    && python -c "import torch; assert torch.version.cuda is None, 'torch 装成了 CUDA 版: ' + torch.__version__ + ' —— 会多出约 7GB 的 nvidia-* 依赖，请检查 TORCH_INDEX_URL 是否指向 CPU wheel 索引'; print('torch', torch.__version__, '(CPU-only)')"

# ── 2) 业务依赖 ─────────────────────────────────────────────────────────
# 先装依赖再拷代码：改代码不会让依赖层失效。
COPY --chown=appuser:appuser requirements.txt ./
RUN pip install -i "${PIP_INDEX_URL}" -r requirements.txt

# ── 3) 应用代码 ─────────────────────────────────────────────────────────
# .dockerignore 已排除 venv / .env（含真实密钥）/ 模型 / 向量库 / 知识库。
COPY --chown=appuser:appuser . .

# ── 4) 构建期冒烟测试：依赖闭包 + 配置可导入 ──────────────────────────────
# 只 import 第三方包与 config（不 import main / agent：它们在 import 时就要读 API key）。
# 依赖清单漏项会让这一步构建失败 —— 这正是想要的。
RUN python -c "import fastapi, uvicorn, pydantic, pymysql, redis, jwt, bcrypt, chromadb, sentence_transformers, pymupdf, pytesseract, openpyxl, docx, rank_bm25, jieba, tiktoken, mcp, fastmcp, langgraph, langchain_openai, langchain_chroma, sse_starlette, dotenv, starlette, pytest; print('依赖闭包 OK')" \
    && python -c "import config; print('config OK, APP_VERSION =', config.APP_VERSION)"

USER appuser

EXPOSE 8000

# 存活探针走 /health：不需要鉴权，也**不会**触发模型加载。
# 就绪探针是 /ready（会真的探 MySQL / Redis 可达性），在 compose 里覆盖成它。
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

STOPSIGNAL SIGTERM

# tini 做 1 号进程：转发信号 + 回收僵尸（应用会拉起 MCP stdio 子进程）
ENTRYPOINT ["/usr/bin/tini", "--"]

# 容器里必须监听 0.0.0.0 —— main.py 里的 127.0.0.1 只适合本机开发。
#   --workers 1                  见文件头第 1 条（限流 / BM25 是进程内状态）
#   --no-access-log              访问日志由 observability.py 统一输出（带 request_id），别打两份
#   --no-server-header           不回 `server: uvicorn`，少一点版本指纹
#   --timeout-graceful-shutdown  收到 SIGTERM 后给在途请求留 30s（问答/流式响应会被硬砍）
# 这里**不加** --proxy-headers：X-Forwarded-For 的信任由应用自己的开关决定
# （config.TRUST_PROXY_HEADERS，默认关闭，见 auth/deps.py），
# 在 uvicorn 层无条件信任会让两处口径不一致 —— 客户端能伪造 IP 就等于绕过限流。
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log", "--no-server-header", "--timeout-graceful-shutdown", "30"]
