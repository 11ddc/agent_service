# 智能客服 API —— 生产镜像
#
# ⚠️ 这个文件**没有在本机构建验证过**（构建要联网拉依赖，本仓库的开发环境
#    是离线的）。它的结构是标准的，但首次真机构建时请留意下面标注的两处。
#
# 关键取舍：
#   1. 单进程 uvicorn（`--workers 1`）。多副本请用编排平台横向扩，不要把
#      --workers 调大：进程内限流/BM25 索引都是**每进程一份**，副本内多 worker
#      会让限流额度被放大成 N 倍（想多 worker 就先把 RATE_LIMIT 指向 Redis，
#      BM25 缓存已经是磁盘共享的了）。
#   2. 非 root 运行。
#   3. 模型与向量库走**卷挂载**，不打进镜像（镜像能小一个数量级，也避免每次
#      改文档都要重建镜像）。

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUTF8=1 \
    # HF 镜像（国内拉模型用），需要连外网时再改
    HF_HOME=/app/models/hf

# 系统依赖：
#   libgomp1        —— onnxruntime/chromadb 运行时需要
#   tesseract-ocr + 中文语言包 —— 扫描件 OCR（缺失时程序会跳过 OCR 而不是崩，见 rag/rag.py）
#   fonts-noto-cjk  —— OCR/图片渲染的中文字体
#   curl            —— HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        tesseract-ocr \
        tesseract-ocr-chi-sim \
        fonts-noto-cjk \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装依赖再拷代码：改代码不会让依赖层失效
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 运行时目录（挂卷覆盖）：知识库、向量库、BM25 缓存、模型、日志
RUN mkdir -p /app/knowledge_base /app/chroma_db /app/bm25_cache /app/models /app/data /app/logs \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# 存活探针打 /health（不需要鉴权，且**不会**触发模型加载）
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# 注意：这里的 --host 0.0.0.0 与本地开发脚本（127.0.0.1）不同 ——
# 容器里必须监听全部网卡，否则宿主机连不上。
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
