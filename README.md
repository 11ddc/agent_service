# 智能客服问答 API —— RAG 检索 + Agent 编排

面向智能家居售后场景的问答服务：**文档入库 → 混合检索 → 意图路由 → 生成/工具调用**，支持 SSE 流式输出。

项目重点不在"接一个大模型"，而在**检索质量可度量**与**失败路径可降级**：每个环节都有评估口径、有降级分支、有对应的回归测试。

| | |
|---|---|
| 检索层（165 条正样本标注集） | 精排后 **R@1 84.8% / R@20 98.2% / MRR 0.901** |
| 测试 | **671 个用例全绿**，零外部服务（不连 Redis / MySQL / 任何 LLM）；另有 12 个 `integration` 用例需真 MySQL，默认不跑 |
| 入库 | 167 份文档 / 1761 个子块 / 98.4 秒，含 2 项安全探针 |
| 技术栈 | FastAPI · LangGraph · Chroma · BM25(jieba) · MySQL · Redis · DashScope · MCP |

---

## 快速开始：克隆下来怎么跑起来

### 0. 前置条件

| 依赖 | 必需性 | 说明 |
|---|---|---|
| Python **3.11+**（实测 3.12） | 必需 | |
| **DeepSeek** + **阿里云百炼 DashScope** 的 API Key | 必需 | 缺了服务**启动即失败**（见第 2 步） |
| 磁盘 ~3 GB | 必需 | 两个本地模型首次运行时自动下载：向量 183MB + 精排 2.4GB |
| MySQL 8 | 建议 | 父块存储 + 文档清单。缺了不会崩，但 small-to-big 的父块扩展会降级 |
| Redis | 建议 | 会话历史 + 转人工计数窗口。缺了只影响这两项 |
| Tesseract OCR | 可选 | 只影响扫描件 PDF / 文档内嵌图的文字识别 |

> **只想跑测试？** 不需要任何 key、不需要 MySQL / Redis、不需要模型 —— 克隆完直接
> `pytest` 就是 **671 个用例全绿**（`tests/conftest.py` 会注入占位 key，所有 LLM 调用都走桩，
> 并且**强制拦截一切非回环出网连接**：真实出网 = 测试失败）。
> 机器上缺 tesseract 或 CJK 字体时，会跳过 3 个扫描件 OCR 用例（是 skip，不是 fail）。
> 另有 12 个 `integration` 用例（真 MySQL 的认证链路）默认不跑：`pytest -m integration -q`。
> 想先确认"这套东西是活的"，这是最快的路径。

### 1. 克隆 + 装依赖

```bash
git clone https://github.com/11ddc/agentstudy.git
cd agentstudy

python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt   # Windows
venv/bin/python -m pip install -r requirements.txt           # Linux / macOS
```

用 `uv` 也行（依赖清单与 requirements.txt 一致）：

```bash
uv sync
```

> ⚠️ 两个本地模型（`embbding_models/`、`Reank_models/`）体积太大已 gitignore，
> **仓库里没有**。不需要手动下载：默认路径不存在时会自动退回 HuggingFace 仓库 ID
> （`BAAI/bge-small-zh-v1.5` / `BAAI/bge-reranker-v2-m3`）并走 `hf-mirror.com` 镜像。
> 想放到本地目录也行，把 `EMBEDDING_MODEL_PATH` / `RERANK_MODEL_PATH` 指过去即可。

### 2. 配置 `.env`

```bash
copy .env.example .env      # Windows
cp .env.example .env        # Linux / macOS
```

`.env` 已被 gitignore，**真实 key 不会入库**。有 **2 个平台 key 是启动必需**的：

| 变量 | 去哪申请 | 用途 |
|---|---|---|
| `DEEPSEEK_API_KEY` | [platform.deepseek.com](https://platform.deepseek.com/api_keys) | 主模型：query 改写 / 意图仲裁 / 兜底 Agent / **RAG 答案生成** |
| `ZHI_PU_API_KEY` | [open.bigmodel.cn](https://open.bigmodel.cn/) | 工具调用（glm-4.5-air）、问题拆分（glm-4.5-air）、视觉 OCR（glm-4v-flash，免费） |
| `MYSQL_URL` | 自己起 | 父块存储，格式见模板 |

这两个 key 是**在 import 期**就被读走的（`agent/langchina.py:25`、
`intent/problemdecomposition.py:31`、`tools_agent/tool_llm.py:31` 都在模块级构造
OpenAI 客户端），而 `api_key=None` 会让 SDK **构造即抛** `OpenAIError` ——
所以缺一个就**启动即崩**，而且只给一段原始堆栈：`main.py` 里并**没有**启动前自检，
别指望它把缺哪个 key 友好地列出来。

> `QIANWEN_API_KEY`（百炼）是**可选**的：只有把 `EMBEDDING_PROVIDER` 设成 `dashscope`
> 用云端向量时才需要，默认走本地模型，不填也能跑。
>
> `GENERATE_API_KEY`、`BASE_URL`、`QIAN_WEN_QUERYSTION_API_KEY` 是**当前代码不读**的
> 历史变量：答案生成已改用 `DEEPSEEK_API_KEY`（`rag/generatellm.py` 的 `GENERATE_MODEL`
> 默认 `deepseek-chat`），问题拆分已从千问迁到智谱（`intent/problemdecomposition.py`
> 现在读 `ZHI_PU_API_KEY`）。填了不生效，不用为此申请 key。

> `QIANWEN_API_KEY`（云端向量 + 视觉 OCR）是**可选**的：默认走本地向量 + 本地 OCR，
> 不填也能跑；只有要处理纯图片扫描页、或想换成云端 embedding 时才需要。

### 3. 起 MySQL / Redis（可选，建议）

```bash
# MySQL：建库 + 建表（建表脚本幂等，可重复执行）
mysql -uroot -p -e "CREATE DATABASE rag DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
venv\Scripts\python.exe -m db.init_schema

# Redis：本地起一个即可，默认就连 redis://localhost:6379
docker run -d -p 6379:6379 redis:7
```

两个都没起也能启动，只是对应功能降级（父块扩展 / 会话历史）。

### 4. 灌一份知识库

仓库里不带语料（`knowledge_base/`、`chroma_db/` 都是产物）。用固定 seed 的语料生成器
造一份可复现的：

```bash
# 生成语料（171 份文档 + 1 份 manifest，含 4 个刻意准备的边界样本）
venv\Scripts\python.exe eval\gen\gen_corpus.py --profile L --seed 42

# 先起服务（另一个终端），再批量走真实 HTTP 接口入库
venv\Scripts\python.exe main.py
venv\Scripts\python.exe eval\ingest_corpus.py
```

也可以不用生成器，直接把自己的文档丢进 `knowledge_base/`，或调 `POST /api/upload`。

### 5. 启动

```bash
venv\Scripts\python.exe main.py
```

打开 **http://127.0.0.1:8000/docs** 就是交互式接口文档。主要接口：

| 接口 | 说明 |
|---|---|
| `GET /health` | **存活探针**（无需鉴权）：版本、运行时长，以及 AUTH/ACL/限流 开关状态 |
| `GET /ready` | **就绪探针**：MySQL 不可用且认证开启 → 503；Redis 挂了仍就绪（只是降级） |
| `GET /metrics` | Prometheus 文本格式指标（默认需 admin） |
| `POST /api/auth/register` | 注册（可用 `AUTH_ALLOW_REGISTRATION=0` 关闭） |
| `POST /api/auth/login` | 登录，返回访问令牌 + 刷新令牌 |
| `POST /api/auth/refresh` | 刷新令牌（**轮换**：旧的立即失效） |
| `POST /api/auth/logout` | 登出（访问令牌**即时**失效） |
| `GET /api/auth/me` | 当前身份 |
| `POST /api/auth/users` | 管理员建号（唯一能创建 kb_admin / operator 的入口） |
| `GET /api/auth/users` | 管理员查看本租户账号 |
| `POST /api/kb/documents/{doc_id}/publish` | 审核通过（从此参与回答）· 需 kb_admin |
| `POST /api/kb/documents/{doc_id}/archive` | 下架（立刻不参与回答）· 需 kb_admin |
| `POST /api/kb/documents/{doc_id}/visibility` | 改可见范围 · 需 kb_admin |
| `GET /api/kb/documents` | 文档清单（可按 status 过滤）· 需 kb_admin / operator |
| `POST /api/chat` | 一次性问答（需登录） |
| `POST /api/chat/stream` | SSE 流式问答（需登录） |
| `POST /api/upload` | 上传文档入库（需 `kb_admin`；默认进**待审核**状态） |

> **认证是强制的**：三个业务接口都要 `Authorization: Bearer <access_token>`。
> 缺令牌 401、角色不足 403、账号存储不可用 503（**失败关闭**，绝不放行）。
>
> 首个管理员用引导脚本创建（自助注册只能拿到最低角色 `user`）：
>
> ```bash
> # 生成 JWT 密钥（至少 32 字节）并填进 .env 的 AUTH_JWT_SECRET
> python -c "import secrets;print(secrets.token_urlsafe(48))"
> BOOTSTRAP_ADMIN_PASSWORD=你的密码 venv\Scripts\python.exe -m auth.bootstrap --username admin
> ```
>
> ⚠️ **开启认证后 MySQL 从"建议"变成"必需"**：账号体系在库里，连不上就 503。
> 本地演示可以设 `AUTH_ENABLED=0` 关掉认证，但启动会打 WARNING，生产禁用。

### 知识库的权限与审核（ACL）

每个子块都带 `tenant_id` / `owner_id` / `visibility` / `status` 四个字段，
**dense（Chroma where）与 sparse（BM25）两条召回通道都会强制过滤** ——
只过滤其中一条就等于权限形同虚设（BM25 是对全量语料打分的，不经过 Chroma）。

| 维度 | 取值 |
|---|---|
| `visibility` | `tenant`（本租户，默认）/ `private`（仅上传者）/ `public`（所有租户） |
| `status` | `draft`（待审核，**检不到**）/ `published`（参与回答）/ `archived`（下架） |

上传默认进 `draft`（`ACL_REQUIRE_APPROVAL=false` 可改成直接发布）。
流程：上传 → 拿 `doc_id` → `POST /api/kb/documents/{doc_id}/publish` → 才参与回答。

> **存量库需要跑一次回填**（ACL 新列默认 draft，会让老文档查不到；回填是幂等的）：
>
> ```bash
> venv\Scripts\python.exe -m rag.acl_backfill --dry-run   # 先看有多少要回填
> venv\Scripts\python.exe -m rag.acl_backfill             # 回填成"本租户已发布"
> ```

> 首次调用检索会加载模型，**冷启动几十秒是正常的**（GPU 加载 2.4GB 精排模型）；
> 之后复用一个进程内单例，不再重复加载。

### 内容审核、限流与运维

| 能力 | 说明 |
|---|---|
| **内容审核** | `input`/`output`/`document` 三个挂载点；默认离线规则（不出网）。命中黑名单或文档里的**提示注入**特征 → 拒答 / 拒绝入库（422） |
| **PII 脱敏** | 日志与审计里的手机号/身份证/银行卡/邮箱自动打码。**只脱敏、不拦截** —— 用户给手机号查订单是正常业务 |
| **入口限流** | 按**账号**的令牌桶（不按 IP：会被 NAT/代理池绕过，还会误伤整个出口）；429 带 `Retry-After` |
| **探针与指标** | `/health`、`/ready`、`/metrics`（Prometheus 文本）；每条日志带 `request_id` |
| **容器化** | 一条 `docker compose up -d --build` 起全套：`api`（本仓库 `Dockerfile`）+ `web`（**前端项目自己的 Dockerfile**，路径由 `FRONTEND_DIR` 指）+ `mysql` / `redis` / `nginx`（官方镜像；入口配置 `deploy/nginx/default.conf`，负责 SSE 长连接、50MB 上传、X-Forwarded-For）。后端镜像里 **torch 装的是 CPU 版**（省约 7GB、构建快一个数量级）；apt/pip/torch 三个源可用 `.env` 里的 `APT_MIRROR` / `PIP_INDEX_URL` / `TORCH_INDEX_URL` 覆盖，**不用改 Dockerfile** |

> **会话记忆默认是进程内的**：`AGENT_CHECKPOINT_BACKEND=memory` 意味着重启丢记忆、
> 多副本各存一份（症状是"客服怎么又忘了"）。生产请改成 `sqlite`/`postgres`
> 并安装 `langgraph-checkpoint-*`；启动时会打 WARNING 提醒你。

> **多副本时注意限流口径**：进程内限流是每副本一份，额度会被放大成 N 倍
> （BM25 缓存已经是磁盘共享的）。要么横向扩副本而不是调大 `--workers`，要么把限流指向 Redis。

### 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| 启动直接退出，提示缺 API Key | 照提示把 `.env` 里列出的变量填上；`.env` 得在**项目根目录** |
| 日志出现「重排失败，降级为 RRF 默认顺序」 | 精排模型没加载成功 → 指标会掉。检查 `RERANK_MODEL_PATH` / 显存，或先设 `RERANK_DEVICE=cpu` |
| 日志出现「tiktoken 不可用，改用字符粗估」 | 没装 tiktoken，只是预算估得粗，不影响功能 |
| 检索为空 | 还没入库（第 4 步），或 `knowledge_base/` 是空的。先在 `/docs` 里调 `/api/upload` |
| 上传扫描件 0 块入库 | 需要 OCR：装 Tesseract（+`chi_sim` 语言包），装在非默认位置时用 `TESSERACT_CMD` 指路 |
| 想确认改动没把东西跑坏 | `pytest -q` —— 全绿说明检索/路由/降级路径都正常 |

---

## 它能做什么

- **多格式文档入库**：PDF（含扫描页 OCR 判定）/ DOCX（标题层级、表格窗口、内嵌图 OCR）/ XLSX（表头感知的窗口切分）/ TXT / Markdown，统一切成"父块 + 子块"两级结构。
- **混合检索**：向量召回（dense）+ 关键词召回（BM25），RRF 融合，Cross-Encoder 精排，最后按 `parent_id` 展开成父块 —— 打分用小而准的子块，喂给模型用带章节标题的父块。
- **三级意图漏斗**：规则 → Embedding 相似度 → LLM 结构化输出仲裁；只在歧义时才付出 LLM 的延迟与成本，任一层不可用都能降级而不是整条链失效。
- **Agent 工具调用**：4 个知识库工具（全库检索 / 限定文档检索 / 文档清单 / 章节概览）+ 通过 MCP 接入的外部工具，由 LangGraph 子图完成"调用-观察-再调用"循环。
- **多问题拆分与改写**：带上下文的 query 改写（短查询才触发，单号/型号原样保留）→ 拆成子问题 → 多问题逐个走完整流程后合并。
- **SSE 流式**：检索/生成阶段状态、真 token 增量、"部分内容未走流式"时的重置对账。
- **上下文预算**：按 token 估算打包检索结果，超出预算截断，而不是静默丢给模型触发 400。

---

## 实测效果

### 检索层（可复现）

174 条标注样本（165 正 / 9 负），语料 profile=L（171 个文件，含 4 个刻意准备的边界样本），本地 embedding + 本地精排：

| 通道 | 命中 | R@1 | R@3 | R@5 | R@20 | MRR |
|---|---|---|---|---|---|---|
| dense（bge-small-zh + Chroma） | 120/165 | 41.2% | 53.3% | 57.0% | 72.7% | 0.492 |
| BM25（jieba） | 159/165 | 70.3% | 87.3% | 93.3% | 96.4% | 0.799 |
| RRF 融合 | 162/165 | 52.7% | 77.0% | 89.7% | **98.2%** | 0.676 |
| **+ 精排** | 162/165 | **84.8%** | **93.9%** | **96.4%** | **98.2%** | **0.901** |
| + 父块扩展 | 161/165 | 84.8% | 93.9% | 96.4% | 97.6% | 0.901 |

分桶看精排的价值（R@1，融合后 → 精排后）：

| 分桶 | n | RRF R@1 | 精排 R@1 |
|---|---|---|---|
| 原词（文档里的词） | 9 | 77.8% | 66.7% |
| 口语改写（同义不同词） | 8 | 37.5% | **75.0%** |
| 跨文档（区分相近条目） | 9 | 44.4% | **66.7%** |
| 编号检索（唯一 ID 查找） | 139 | 52.5% | 87.8% |

时延（p50 / p90，毫秒）：向量化 72 / 136 · dense 10 / 14 · BM25 10 / 17 · **精排 1632 / 5882**

> **怎么读这张表（很重要）**
> 1. 融合把 R@20 从 96.4% 拉到 98.2%、补住了 dense 的短板，但 **R@1 反而低于 BM25 单路**（52.7% vs 70.3%）；靠精排把 R@1 拉回 84.8%。也就是说**融合负责召回上限，精排负责排序质量**，两件事不能混着看。
> 2. 精排增益最大的正是两个最难的桶（口语改写 37.5%→75.0%、跨文档 44.4%→66.7%），这正是它该起作用的场景。
> 3. 样本分布偏"编号检索"（139/165），总指标主要由该桶决定；原词/口语改写/跨文档三桶各只有 8~9 条，**只能看趋势，不能当结论**。
> 4. 这一步是**纯检索层**评测，不含生成正确率，也不含路由层。

判定精度是 **chunk 级**而不是文件级：标注集里每条样本都带 `anchor`（必须是该 chunk 原文的精确子串），脚本启动时先校验锚点再算指标 —— 否则一个占 39/53 块的大文件会让 R@20 虚高到接近 1。方法论详见 [`eval/README.md`](eval/README.md)。

### 入库与安全探针

| 项 | 结果 |
|---|---|
| 应当成功的文档 | 167 / 167 全部成功（`failed_expectations` 为空） |
| 子块总数 | 1761 |
| 全量入库耗时 | 98.4 秒 |
| 刻意准备的 4 个边界样本 | 全部按预期表现：截断 PDF 与 GBK 编码文本 → 500 并留痕；空文件 → 200 且 0 块；不支持的格式（`.png`）→ 明确报错而不是静默入库 |
| 安全探针 1：路径穿越（`../../evil.pdf`） | 拒绝（400），知识库外无写入 |
| 安全探针 2：超大文件（51MB，上限 50MB） | 拒绝（413），无残留 `.part` 文件 |
| 安全探针 3：盘符相对路径（`C:evil.pdf`） | 拒绝（400）。这种名字**没有分隔符**，靠"取最后一段"挡不住，而 Windows 的 `ntpath.join` 遇到另一个盘符会**整个丢弃**知识库目录 → 落盘落到知识库之外 |
| 安全探针 4：超大请求体（> 51MB） | 在 multipart 解析**之前**就拒绝（413）。Starlette 对文件字段没有任何大小上限，守在接口里的判断是"事后检查"——超限请求会先被完整缓冲到内存/临时盘 |
| 安全探针 5：越权查订单 | 调用方身份走**子进程环境变量**注入（模型无法伪造），订单类工具只返回该身份名下的数据；`query_recent_orders` 已无 `phone` 入参（那是"按手机号枚举他人订单"的入口）。**查他人订单与查不存在的订单返回同一句话术**，不泄露存在性 |

---

## 架构

### 请求编排（LangGraph 三层图）

```mermaid
flowchart TD
    C[客户端] -->|POST /api/chat 或 /api/chat/stream| API[FastAPI]
    API --> RW[rewrite_node<br/>带上下文的 query 改写]
    RW --> SP[splitter<br/>拆成子问题]
    SP -->|单问题| QF[单问题子图]
    SP -->|多问题| ML[multi_loop<br/>逐个走子图后合并]
    ML --> QF
    QF --> IR[intent_router<br/>三级意图漏斗]
    IR -->|kb_question| RAG[rag_flow<br/>混合检索 + 生成]
    IR -->|tool_call| TA[tool_agent 子图<br/>llm_call 与 tool_call 循环]
    IR -->|模糊 / 分类器不可用| AG[agent_flow<br/>Agent 兜底]
    IR -->|chitchat / handoff / out_of_scope| SC[short_circuit<br/>直接回复]
    RAG --> M[merge<br/>answer + meta]
    TA --> M
    AG --> M
    SC --> M
```

### 检索链路

```mermaid
flowchart LR
    Q[query] --> V[bge-small-zh 向量化] --> D[dense top-k]
    Q --> B[BM25 + jieba] --> S[sparse top-k]
    D --> F[RRF 融合]
    S --> F
    F --> RR[Cross-Encoder 精排]
    RR --> P[按 parent_id 展开父块]
    P --> BD[上下文预算打包]
    BD --> G[生成答案]
```

### 模块职责

| 目录 | 职责 |
|---|---|
| `api/` | HTTP 接口：`/api/chat`、`/api/chat/stream`（SSE）、`/api/upload` |
| `agent/graph.py` | LangGraph 三层图：主图（改写→拆分→单/多问题）、单问题子图（意图→分支→汇总）、工具子图（LLM⇄工具循环） |
| `intent/` | 三级意图漏斗 + 意图/槽位数据结构 |
| `rag/rag.py` | 解析（PDF/DOCX/XLSX/图内 OCR）、父子切分、双路召回、RRF、精排、父块扩展、入库与状态 |
| `rag/local_embedding.py` · `rag/local_reranker.py` | 本地 embedding 与 Cross-Encoder 精排（懒加载，启动不加载 torch） |
| `query_rewrite/` | 改写（带会话历史、短查询闸门）+ Redis 历史读写 |
| `tools_agent/` | 知识库工具（4 个）+ 工具调用模型 + 工具分派 |
| `context_budget.py` | token 估算、预算打包、溢出识别 |
| `db/` · `redis_client.py` | MySQL 文档/父子块/账号存储 · Redis 会话历史与转人工计数窗口 |
| `auth/` | 认证与授权：bcrypt 密码、JWT 访问令牌、刷新令牌轮换、即时撤销、RBAC、审计、首个管理员引导 |
| `rag/acl.py` · `moderation.py` | 文档级 ACL（两条检索通道都过滤）· 内容审核与 PII 脱敏 |
| `metrics.py` · `observability.py` · `rate_limit.py` | 指标注册表 · request_id/结构化日志 · 入口限流 |
| `mcp_client.py` | 外部 MCP 服务接入，工具统一命名空间 `mcp__<server>__<tool>` |
| `eval/` | 语料生成器（固定 seed）、批量入库、检索召回评测脚本 |
| `tests/` | 671 个用例，全部零外部服务（出网被强制拦截） |




## 目录结构

```
my-agent-api/
├── main.py                 # 入口（uvicorn 127.0.0.1:8000）+ 启动前 .env 自检
├── config.py               # 配置集中入口（模型路径 / Chroma / Redis / OCR）
├── context_budget.py       # token 预算与打包
├── mcp_client.py           # MCP 外部工具接入
├── .env.example            # 环境变量模板（复制成 .env 后填 key）
├── requirements.txt        # 运行依赖（pip）
├── pyproject.toml          # 同一份依赖（uv），与 requirements.txt 保持同步
├── router/                 # 路由注册
├── api/                    # chat / chat_stream / upload
├── agent/                  # LangGraph 三层图 + 流式事件旁路
├── intent/                 # 三级意图漏斗
├── query_rewrite/          # 改写 + 会话历史
├── rag/                    # 解析 / 切分 / 检索 / 精排 / 生成
├── tools_agent/            # 知识库工具 + 工具调用模型
├── db/                     # MySQL 存储层 + schema.sql
├── eval/                   # 语料生成 + 批量入库 + 召回评测
└── tests/                  # 671 个用例
```

> 下面这些是**运行产物或大文件**，刻意不入库，克隆后按「快速开始」第 3~4 步补上：
> `knowledge_base/`（语料）、`chroma_db/`（向量库）、`images/`（OCR 中间图）、
> `embbding_models/`、`Reank_models/`（本地模型）、`.env`（密钥）。
