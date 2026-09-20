# eval —— RAG 检索层召回率评测

只测**检索层**（`rag.rag` 里的 dense / BM25 / RRF 三档），不含精排、不含 LLM 生成。
路由层（意图识别）和生成层（答案正确率）是另外两件事，别混在这张表里看。

## 跑法

```powershell
venv\Scripts\python.exe eval\retrieval_recall.py
```

- 需要网络 + `.env` 里的 `QIANWEN_API_KEY`（查 query 向量）+ `chroma_db/` 已建库
- 首次跑约 1~2 分钟；query 向量会缓存到 `eval/cache/query_vectors.json`，重跑不再花 API 额度
- **改了 query 文本或新增样本后，缓存对旧 query 仍然有效**，只有新 query 会新调用

## 标注集格式（`eval/queries.jsonl`，一行一条 JSON）

```jsonc
{"query": "问题", "bucket": "原词|口语改写|跨文档|负样本", "file": "答案所在文件", "anchor": "锚点句"}
{"query": "库里没有答案的问题", "bucket": "负样本", "file": null, "anchor": null}
```

| 字段 | 说明 |
|---|---|
| `file` | 答案所在**文件名**（脚本自动用 basename 比对，不受绝对路径影响） |
| `anchor` | **必须是该文件 chunk 原文里的精确子串**，脚本启动时会校验 |
| `bucket` | 分桶，用于分别统计。`原词`=文档里的词，`口语改写`=同义不同词，`跨文档`=需要区分相近条目 |
| `file: null` | 负样本，不参与召回率，只列出被硬召回的 top1 |

### 为什么必须有 anchor

库里 `11.docx` 一个文件占 39/53 块。如果只用文件名判定，命中这 39 块里**任意一块**都算"召回成功"，
`recall@20` 会虚高到接近 1.0，指标完全失去区分度。锚点句把判定精度拉到 chunk 级。

### 锚点写错的后果

锚点如果不在库里（打错字、多空格、全角/半角不符），该样本**永远不可能命中**，召回率会假性归零。
所以脚本每次启动都会先校验锚点，校验失败会明确报出来——**先修标注，再看指标**。

## 指标怎么读

| 现象 | 说明 | 该做什么 |
|---|---|---|
| `dense` 与 `bm25` 在"原词"组都高，`bm25` 在"口语改写"组明显掉 | 正常且互补 | BM25 该保留 |
| `rrf` 低于 `dense` 单路 | 融合被 BM25 噪声拖累 | 降权或改写融合，别直接删 |
| `recall@20` 高但 `recall@1` 低 | 召回没问题、**排序**有问题 | 上精排/调权重，别动切分 |
| `bm25` 的 `recall@20` 天然偏低 | `_sparse_search` 内有 `scores[i] > 0` 过滤，返回可能少于 fetch_k 条 | 属正常，别误判 |

## 未命中归因（`eval/misses.md`，比总指标更有用）

对每条 `rrf` 未命中或排名 >10 的样本，脚本会导出实际召回的 top5，人工归三类：

- **A 库里确实没有** → 标注写错了，或这该是负样本；不是检索的锅
- **B 正确块进了 top20 但排在第 11~20 名** → 排序问题，精排/加权能救
- **C 正确块完全没进 top20** → 真召回失败。再看是 embedding 语义不匹配（口语 vs 书面），
  还是**切分把答案切碎了**（`CHUNK_SIZE=1000` 配默认分隔符，中文按字符硬切，锚点句可能被劈成两半）

三类占比决定优化方向：C 多 → 改切分和 embedding；B 多 → 精排调权重；A 多 → 回去修标注。

## 已知边界（别把结论读过头）

1. **库只有 53 块 / 9 个文件**，`recall@20` 区分度天生有限 → 主要看 `recall@1/@3` 和 MRR。
2. 现在是**单 GT** 判定（一条 query 一个答案位置）。真实场景一个问题常有多个正确来源
   （例如 `z9_manual.txt` 说"4 档"、`z9_check_upload.txt` 说"6 档"，问"几档"两份都该召回）。
   要做多 GT 覆盖率，把 `file`/`anchor` 扩成列表即可，脚本改 `rank_of` 一处。
3. 不含精排档位。要测精排增益，需在脚本里追加 `rag.reordering()`，
   并且**必须先断言 `rag.reranker.model is not None`**——`reordering()` 内部 `except` 会
   静默降级成原顺序，不断言就会把"精排没跑成"误读成"精排无增益"。
4. 默认不跑精排是有意的：检索层的问题（切分/embedding/分词）应该在精排之前先修好，
   否则精排只是在给一份烂候选排序。

## 相关文件

- `eval/queries.jsonl`：标注集（要一起维护、一起提交）
- `eval/retrieval_recall.py`：评测脚本
- `eval/_chunks.json`：库内 53 块内容快照（标注取材用，已 gitignore）
- `eval/cache/`、`eval/detail.csv`、`eval/misses.md`：产物，已 gitignore
