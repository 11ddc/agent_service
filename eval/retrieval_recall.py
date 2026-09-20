"""
RAG 检索层召回率评测（只测检索，不含精排、不含 LLM 生成）

用法：
    venv\\Scripts\\python.exe eval\\retrieval_recall.py

为什么这么写：
1. 直接调用 rag.rag 的生产函数（_sparse_search / _rrf_fuse），测的就是线上那条路径，
   不复制一份检索逻辑，否则测出来的数字不代表线上。
2. 命中判定用「文件名 + 锚点句」：只用文件名会让 11.docx（39/53 块）虚高到接近 1.0。
3. 每条 query 只调一次 DashScope embedding，向量落盘缓存，重跑不重复花额度。
4. 启动时先校验每个锚点确实存在于库中对应文件里——锚点写错会让召回率假性归零，
   把标注问题误判成"检索不行"。
5. 未命中的样本自动导出 top5 命中内容，直接支撑"失败三类归因"：
   A 库里确实没有（标注错） / B 召回了但排序靠后 / C 完全没召回。

输出：控制台指标表 + eval/detail.csv（逐条明细） + eval/misses.md（未命中归因材料）
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter

try:  # Windows 控制台编码兜底，避免中文报表打不出来
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # 脚本在 eval/ 下跑，项目根目录要手动挂上才能 import rag
    sys.path.insert(0, str(ROOT))

from rag import rag  # noqa: E402

EVAL_DIR = ROOT / "eval"
QUERIES = EVAL_DIR / "queries.jsonl"
CACHE = EVAL_DIR / "cache" / "query_vectors.json"
DETAIL = EVAL_DIR / "detail.csv"
MISSES = EVAL_DIR / "misses.md"

KS = (1, 3, 5, 10, 20)
MAX_K = max(KS)
# 档位（粒度和候选池不同，读表时别混）：
#   dense / bm25   单路召回（chunk 级）
#   rrf            两路融合（chunk 级）——生产候选池
#   rerank         融合结果过精排（chunk 级）：**候选池与 rrf 完全相同**，
#                  只有顺序不同 → rerank - rrf 就是"精排这一段"的纯贡献
#   rerank_parent  生产真实路径：精排 + 父块展开（喂给 LLM 的那份上下文）。
#                  粒度不同（父块 ⊇ 子块），且生产参数把它压到 ≤8 条，别和上面几档直接比
MODES = ("dense", "bm25", "rrf", "rerank", "rerank_parent")
MISS_PRINT_LIMIT = 8  # 控制台最多打印几条未命中的 top5（全量在 misses.md）
# 负样本相关性打分：只算 RRF 前 SCORE_TOP 条并取最大分，
# 用来回答"能不能用一个精排分数阈值把'库里没有答案'挡掉"
SCORE_TOP = 3


# ── 数据加载 ────────────────────────────────────────────────
def load_queries() -> list[dict]:
    if not QUERIES.exists():
        raise SystemExit(f"缺少评测集：{QUERIES}")
    rows = []
    for ln, line in enumerate(QUERIES.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise SystemExit(f"queries.jsonl 第 {ln} 行不是合法 JSON：{e}") from e
    if not rows:
        raise SystemExit("评测集为空")
    return rows


def load_corpus() -> tuple[dict[str, list[str]], int]:
    """全量拉一次库内文本（和 _build_bm25_index 拉的是同一份数据），用于锚点校验。"""
    data = rag._vectorstore._collection.get(include=["documents", "metadatas"])
    texts = data.get("documents") or []
    metas = data.get("metadatas") or []
    by_file: dict[str, list[str]] = defaultdict(list)
    for text, meta in zip(texts, metas):
        by_file[Path((meta or {}).get("source", "")).name].append(text or "")
    return by_file, len(texts)


def validate_anchors(queries: list[dict], corpus: dict[str, list[str]]) -> int:
    """锚点校验：锚点不在库里 → 该样本永远不可能命中，指标会假性归零。"""
    bad = 0
    for row in queries:
        f, a = row.get("file"), row.get("anchor")
        if not f:
            continue
        if f not in corpus:
            print(f"[锚点校验失败] 库里没有文件 {f} —— query: {row['query']}")
            bad += 1
        elif a and not any(a in t for t in corpus[f]):
            print(f"[锚点校验失败] {f} 里找不到锚点句「{a}」 —— query: {row['query']}")
            bad += 1
    return bad


class EmbedCache:
    """query 向量磁盘缓存：同一条 query 永远只打一次 API。"""

    def __init__(self) -> None:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        self.data = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
        self.hit = 0
        self.miss = 0

    def get(self, query: str):
        if query in self.data:
            self.hit += 1
            return self.data[query]
        vec = rag._vectorstore.embeddings.embed_query(query)
        self.data[query] = vec
        self.miss += 1
        self.flush()  # 每条都落盘：中途中断也不丢已花的钱
        return vec

    def flush(self) -> None:
        CACHE.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")


# ── 命中判定 ────────────────────────────────────────────────
def rank_of(docs, row: dict, limit: int | None = None) -> int | None:
    """命中判定：文件名对上 且 锚点句出现在块内容里。返回 1-based 名次，未命中 None。

    支持**多 GT**：row["file"] 是主答案，row["also_files"] 是同样正确的其他出处。
    一个问题常有多个正确来源（README 已知边界 #2），单 GT 判定会把它们误算成未命中。
    """
    want = {Path(row["file"]).name.lower()}
    want |= {Path(f).name.lower() for f in (row.get("also_files") or [])}
    anchor = row.get("anchor") or ""
    for rank, doc in enumerate(docs[: limit or len(docs)], 1):
        src = Path(doc.metadata.get("source", "")).name.lower()
        if src in want and (not anchor or anchor in doc.page_content):
            return rank
    return None


def brief(doc) -> str:
    name = Path(doc.metadata.get("source", "?")).name
    text = " ".join((doc.page_content or "").split())
    return f"{name} :: {text[:60]}"


# ── 主流程 ──────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> None:
    global QUERIES, DETAIL, MISSES, CACHE

    ap = argparse.ArgumentParser(description="RAG 检索层召回率评测（含精排消融）")
    ap.add_argument("--queries", default=str(QUERIES),
                    help="标注集路径（默认 eval/queries.jsonl；语料生成的 golden 在 eval/golden/queries.jsonl）")
    ap.add_argument("--out-prefix", default=None,
                    help="产物前缀（默认取标注集文件名），用于多套标注集并存")
    args = ap.parse_args(argv)

    QUERIES = Path(args.queries)
    stem = args.out_prefix or QUERIES.stem
    DETAIL = EVAL_DIR / f"detail_{stem}.csv"
    MISSES = EVAL_DIR / f"misses_{stem}.md"
    # 缓存文件名带 embedding 标识：换 embedding 模型后自动换缓存文件，
    # 避免"拿旧模型的向量查新索引"——这种错不报错，只是指标全错。
    import config

    CACHE = EVAL_DIR / "cache" / f"query_vectors_{config.embedding_stamp()}.json"
    print(f"embedding: {config.embedding_stamp()} | Chroma 集合: {config.CHROMA_COLLECTION}")

    queries = load_queries()
    rag._ensure_ready()
    if rag._vectorstore is None:
        raise SystemExit("知识库连接失败：检查 .env 里的 QIANWEN_API_KEY 与 chroma_db/ 目录")
    if rag._vectorstore._collection.count() == 0:
        raise SystemExit("知识库为空，没有可评测的索引")

    corpus, total = load_corpus()
    print(f"标注集: {QUERIES}")
    print(f"库内 chunk = {total}，覆盖文件 = {len(corpus)} 个")
    bad = validate_anchors(queries, corpus)
    if bad:
        print(f"\n⚠ {bad} 条样本的锚点/文件校验失败，先把标注修好再跑，否则召回率不可信\n")
    else:
        print("锚点校验通过：所有锚点句都能在库中对应文件里找到\n")

    # 精排档位的前置断言：reordering() 内部 except 会静默降级成原顺序，
    # 不断言就会把"精排根本没跑成"读成"精排无增益"（README 已知边界 #3）
    rerank_ok = False
    try:
        model = rag.reranker.load_model()
        rerank_ok = model is not None
        print(f"精排模型: {type(model).__name__} @ {getattr(model, 'device', '?')}"
              if rerank_ok else "精排模型未加载，rerank 档位将不可信")
    except Exception as e:  # noqa: BLE001
        print(f"精排模型加载失败，rerank 档位不可信: {type(e).__name__}: {e}")
    if not rerank_ok:
        print("→ 下面 rerank / rerank_parent 两档的数字不要当结论用\n")

    emb = EmbedCache()
    fetch_k = max(MAX_K * 2, 6)  # 与 rag.retrieve 的 fetch_k = max(k*2, 6) 同口径
    detail, misses = [], []
    rerank_failed = parent_degraded = 0

    for i, row in enumerate(queries, 1):
        q = row["query"]
        is_neg = not row.get("file")

        t = perf_counter()
        vec = emb.get(q)
        ms_emb = (perf_counter() - t) * 1000
        t = perf_counter()
        dense = rag._vectorstore.similarity_search_by_vector(vec, fetch_k)
        ms_dense = (perf_counter() - t) * 1000
        t = perf_counter()
        sparse = rag._sparse_search(q, fetch_k)  # 注意内部 scores>0 过滤，可能少于 fetch_k 条
        ms_bm25 = (perf_counter() - t) * 1000
        rrf = rag._rrf_fuse(dense, sparse, top_n=MAX_K)

        # 精排档位 1：只过精排，不展开父块 —— 候选池与 rrf 完全一致，纯比排序
        # （local_reranker 会 print 整份重排后的文档，必须收进内存，否则报表被淹掉）
        t = perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            reranked = rag.reranker.rerank(q, rrf, MAX_K) if rrf else []
        ms_rerank = (perf_counter() - t) * 1000

        # 精排档位 2：生产真实路径（精排 + 父块展开）。
        # 打印收进内存：25 条 × 20 块会把报表彻底淹掉；同时用它判断有没有降级。
        buf = io.StringIO()
        t = perf_counter()
        with contextlib.redirect_stdout(buf):
            prod = rag.reordering(q, rrf) if rrf else []
        ms_prod = (perf_counter() - t) * 1000
        noise = buf.getvalue()
        if "重排失败" in noise:
            rerank_failed += 1
        if "降级为纯子块检索" in noise or "未找到，降级用命中的子块顶替" in noise:
            parent_degraded += 1

        # 负样本相关性打分：精排分数能不能把"库里没有答案"识别出来
        max_score = None
        if rrf:
            pairs = [[q, d.page_content] for d in rrf[:SCORE_TOP]]
            try:
                scores = rag.reranker.model.predict(pairs)
                max_score = round(float(max(scores)), 4)
            except Exception:  # noqa: BLE001
                max_score = None

        rec = {
            "query": q,
            "bucket": row.get("bucket", "?"),
            "kind": row.get("kind", "?"),
            "is_negative": is_neg,
            "ms_emb": round(ms_emb),
            "ms_dense": round(ms_dense),
            "ms_bm25": round(ms_bm25),
            "ms_rerank": round(ms_rerank),
            "ms_rerank_parent": round(ms_prod),
            "rerank_max_score": max_score if max_score is not None else "",
        }
        got = {}
        for mode, docs in (
            ("dense", dense),
            ("bm25", sparse),
            ("rrf", rrf),
            ("rerank", reranked),
            ("rerank_parent", prod),
        ):
            rank = None if is_neg else rank_of(docs, row, MAX_K)
            got[mode] = rank
            rec[f"{mode}_rank"] = rank if rank is not None else ""
            rec[f"{mode}_n"] = len(docs)
            rec[f"{mode}_top1"] = brief(docs[0]) if docs else ""
        detail.append(rec)

        if is_neg:
            print(f"[{i:>2}/{len(queries)}] 负样本  {q[:24]:<26} "
                  f"rrf top1 = {rec['rrf_top1'][:26]}  精排最高分={max_score}")
            continue

        flag = "✓" if got["rrf"] else "✗"
        print(
            f"[{i:>2}/{len(queries)}] {flag} {q[:24]:<26} "
            f"dense={got['dense']} bm25={got['bm25']} rrf={got['rrf']} "
            f"rerank={got['rerank']} prod={got['rerank_parent']}"
        )
        if got["rrf"] is None or got["rrf"] > 10:
            misses.append({"row": row, "got": got, "top": rrf[:5]})

    # ── 指标汇总 ────────────────────────────────────────────
    pos = [d for d in detail if not d["is_negative"]]
    neg = [d for d in detail if d["is_negative"]]

    def stats(mode: str, subset: list[dict]) -> dict:
        ranked = [d for d in subset if d[f"{mode}_rank"] != ""]
        out = {}
        for k in KS:
            out[k] = sum(1 for d in ranked if d[f"{mode}_rank"] <= k) / len(subset)
        out["mrr"] = sum(1.0 / d[f"{mode}_rank"] for d in ranked) / len(subset)
        out["miss"] = len(subset) - len(ranked)
        return out

    print("\n" + "=" * 68)
    print(f"检索层召回率  （正样本 {len(pos)} 条 / 负样本 {len(neg)} 条，fetch_k={fetch_k}）")
    print("=" * 68)
    head = "档位    " + "".join(f"recall@{k:<3} " for k in KS) + "MRR    未命中"
    print(head)
    for mode in MODES:
        s = stats(mode, pos)
        cells = "".join(f"{s[k]:<10.3f}" for k in KS)
        print(f"{mode:<8}{cells}{s['mrr']:<8.3f}{s['miss']}")

    print("\n分桶 recall@10（看 dense / bm25 各自短板）")
    print("-" * 68)
    for bucket in sorted({d["bucket"] for d in pos}):
        sub = [d for d in pos if d["bucket"] == bucket]
        cells = []
        for mode in MODES:
            s = stats(mode, sub)
            cells.append(f"{mode}={s[10]:.2f}")
        print(f"  {bucket:<8} n={len(sub):<3} " + "  ".join(cells))

    # 按 kind 展开：bucket 只有 4 个粗桶，语料 golden 里的 kind 更细
    # （多跳 / 表格取值 / 版本冲突 / 别名 / OCR 页 / 英文 / 干扰域），
    # 定位问题靠这张表
    if any(d.get("kind") not in (None, "?") for d in pos):
        print("\n按 kind 的 recall@10 / MRR")
        print("-" * 68)
        for kind in sorted({d["kind"] for d in pos}):
            sub = [d for d in pos if d["kind"] == kind]
            cells = []
            for mode in MODES:
                s = stats(mode, sub)
                cells.append(f"{mode} {s[10]:.2f}/{s['mrr']:.2f}")
            print(f"  {kind:<18} n={len(sub):<3} " + "  ".join(cells))

    # 精排净增益：候选池相同，差值就是排序带来的
    print("\n精排净增益（rerank - rrf，候选池相同 → 纯排序差异）")
    print("-" * 68)
    rrf_s, rk_s = stats("rrf", pos), stats("rerank", pos)
    for k in KS:
        d = rk_s[k] - rrf_s[k]
        print(f"  recall@{k:<3} rrf={rrf_s[k]:.3f}  rerank={rk_s[k]:.3f}  差={d:+.3f}")
    print(f"  MRR      rrf={rrf_s['mrr']:.3f}  rerank={rk_s['mrr']:.3f}  差={rk_s['mrr'] - rrf_s['mrr']:+.3f}")
    prod_s = stats("rerank_parent", pos)
    print(f"\n生产路径（精排+父块展开，喂给 LLM 的上下文）recall@1/3/5/10 = "
          f"{prod_s[1]:.3f} / {prod_s[3]:.3f} / {prod_s[5]:.3f} / {prod_s[10]:.3f}，MRR={prod_s['mrr']:.3f}")
    print("  注意：该档候选池被生产参数压到 ≤8 条，@20 对它无意义；粒度是父块（含子块原文）")

    # 降级计数：静默降级是这套链路最大的风险面，必须显式报出来
    if rerank_failed or parent_degraded:
        print(f"\n⚠ 降级计数：精排失败 {rerank_failed} 次 / 父块降级 {parent_degraded} 次"
              f"（数字进 meta 才能被调用方看到，现在只 print）")
    else:
        print("\n降级计数：精排 0 次失败，父块 0 次降级（本轮链路干净）")

    if neg:
        print("\n负样本：库里没有答案，看是否硬召回 + 精排最高分能否区分")
        print("-" * 68)
        pos_scores = [d["rerank_max_score"] for d in pos if d["rerank_max_score"] != ""]
        neg_scores = [d["rerank_max_score"] for d in neg if d["rerank_max_score"] != ""]
        for d in neg:
            print(f"  {d['query'][:22]:<24} rrf top1 = {d['rrf_top1'][:22]:<24} 精排最高分={d['rerank_max_score']}")
        if pos_scores and neg_scores:
            print(f"\n  正样本精排最高分：中位 {sorted(pos_scores)[len(pos_scores) // 2]:.4f} / "
                  f"最低 {min(pos_scores):.4f}")
            print(f"  负样本精排最高分：最高 {max(neg_scores):.4f} / 中位 "
                  f"{sorted(neg_scores)[len(neg_scores) // 2]:.4f}")
            gap = min(pos_scores) - max(neg_scores)
            print(f"  → 若最低正样本分 > 最高负样本分（当前差 {gap:+.4f}），"
                  f"说明存在可用阈值；否则不能只靠分数拒答")

    if neg:
        print("\n负样本（库里没有答案，看是否硬召回）")
        print("-" * 68)
        for d in neg:
            print(f"  {d['query'][:24]:<26} rrf top1 = {d['rrf_top1'][:38]}")

    ms_emb = [d["ms_emb"] for d in detail]
    lat = [d["ms_dense"] for d in detail]
    lat_b = [d["ms_bm25"] for d in detail]
    med = lambda xs: sorted(xs)[len(xs) // 2]  # noqa: E731
    print(
        f"\n延迟：dense(HNSW 本地) 中位 {med(lat)}ms / 最大 {max(lat)}ms"
        f"；bm25 中位 {med(lat_b)}ms / 首条 {lat_b[0]}ms（首次调用要建全量内存索引）"
    )
    if emb.miss == 0:
        print(f"embedding：本次 {emb.hit} 条全部命中缓存，延迟记 0 不是真实值（清掉 eval/cache/ 再跑才是冷启动数据）")
    else:
        print(f"embedding(每次检索都要调一次 DashScope)：新调用 {emb.miss} 条，命中缓存 {emb.hit} 条，"
              f"中位 {med(ms_emb)}ms / 最大 {max(ms_emb)}ms")

    # ── 落盘 ────────────────────────────────────────────────
    keys = sorted({k for d in detail for k in d})
    with DETAIL.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(detail)

    lines = ["# 未命中/排序靠后样本（人工归因材料）\n",
             "归因三类：**A** 库里确实没有（标注错/负样本）｜**B** 正确块在 top20 但排太后（排序问题）｜**C** 正确块完全没进 top20（真召回失败）\n"]
    for m in misses:
        row, got = m["row"], m["got"]
        klass = "C（完全没召回）" if got["dense"] is None and got["bm25"] is None else "B（召回了但排太后）"
        lines.append(f"## {row['query']}\n")
        lines.append(f"- 期望：`{row['file']}` 中的「{row['anchor']}」")
        lines.append(f"- 五档名次：dense={got['dense']} / bm25={got['bm25']} / rrf={got['rrf']} "
                     f"/ rerank={got['rerank']} / rerank_parent={got['rerank_parent']}")
        lines.append(f"- 初判：{klass}\n")
        lines.append("- RRF top5 实际召回：")
        lines.extend(f"  {i}. {brief(d)}" for i, d in enumerate(m["top"], 1))
        lines.append("")
    MISSES.write_text("\n".join(lines), encoding="utf-8")

    shown = min(len(misses), MISS_PRINT_LIMIT)
    if shown:
        print(f"\n未命中/靠后样本 {len(misses)} 条，前 {shown} 条：")
        for m in misses[:shown]:
            print(f"  · {m['row']['query'][:30]}  期望 {m['row']['file']}，rrf={m['got']['rrf']}")
    print(f"\n明细: {DETAIL}\n归因材料: {MISSES}")


if __name__ == "__main__":
    main()
