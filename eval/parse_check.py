"""入库前预检 —— 零成本（不连 MySQL/Chroma、不调 embedding、不写任何库）。

灌库要真花钱（每块都要过 DashScope embedding），所以在花钱之前先确认两件事：

1. **每个语料文件能不能解析出结构**：sections / 父块 / 子块 / 字符数，
   以及是哪种结构单元（text/table/image）；解析异常的必须在这里暴露，
   而不是等到灌完库才发现某个文件 0 块。
2. **golden 的 anchor 是否真的落在该文件的子块里**。子块才是被检索的粒度：
   anchor 必须是某个子块的**精确子串**，否则这条样本永远不可能命中，
   召回率会假性归零 —— 把标注问题误判成"检索不行"（eval/README.md 里专门警告过）。

用法：
    venv\\Scripts\\python.exe eval\\parse_check.py
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag import rag  # noqa: E402

CORPUS = ROOT / "eval" / "corpus"
GOLDEN = ROOT / "eval" / "golden" / "queries.jsonl"


def load_golden() -> list[dict]:
    if not GOLDEN.exists():
        raise SystemExit(f"缺少 golden：{GOLDEN}（先跑 eval/gen/gen_corpus.py）")
    return [
        json.loads(line)
        for line in GOLDEN.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_one(path: Path) -> dict:
    """解析 + 切分单个文件；解析期的大量 print 收进内存，只留结论。"""
    buf = io.StringIO()
    info = {"file": path.name, "sections": 0, "parents": 0, "children": 0,
            "chars": 0, "kinds": Counter(), "child_texts": [], "error": None}
    try:
        with contextlib.redirect_stdout(buf):
            sections = rag._load_file(str(path))
            info["sections"] = len(sections)
            info["kinds"] = Counter(s.kind for s in sections)
            if sections:
                parents, children = rag._split_document(str(path), sections)
                info["parents"] = len(parents)
                info["children"] = len(children)
                info["child_texts"] = [c[1] for c in children]
                info["chars"] = sum(len(c[1]) for c in children)
    except Exception as e:  # noqa: BLE001
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def main() -> int:
    manifest_path = CORPUS / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"缺少语料 manifest：{manifest_path}（先跑 eval/gen/gen_corpus.py）")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    golden = load_golden()

    print("=" * 96)
    print(f"解析预检：{len(manifest['files'])} 个文件（profile={manifest['profile']}，seed={manifest['seed']}）")
    print("=" * 96)
    print(f"{'文件':<36}{'预期':<7}{'sec':>4}{'父':>4}{'子':>5}{'字符':>8}  结构 / 异常")
    print("-" * 96)

    results = {}
    unexpected = []
    for f in manifest["files"]:
        path = CORPUS / f["file"]
        info = parse_one(path)
        results[f["file"]] = info
        structs = ",".join(f"{k}×{v}" for k, v in sorted(info["kinds"].items())) or "-"
        note = structs
        if info["error"]:
            note = f"⚠ {info['error'][:60]}"
        print(
            f"{f['file']:<36}{f['expect']:<7}{info['sections']:>4}{info['parents']:>4}"
            f"{info['children']:>5}{info['chars']:>8}  {note}"
        )
        # 预期与实际是否一致（哨兵用例就是靠这个断言）
        expect = f["expect"]
        ok = (
            (expect == "ok" and info["error"] is None and info["children"] > 0)
            or (expect in ("fail",) and (info["error"] is not None or info["children"] == 0))
            or (expect == "empty" and info["children"] == 0)
            # reject：后缀不在白名单，接口层就挡掉了，永远不会走到解析
            or (expect == "reject" and info["sections"] == 0)
        )
        if not ok:
            unexpected.append((f["file"], expect, info["error"] or f"children={info['children']}"))

    # ── anchor 校验：必须精确落在子块里 ──────────────────────────
    print()
    print("=" * 96)
    print("anchor 校验（必须是该文件某个子块的精确子串）")
    print("=" * 96)
    anchor_bad = []
    sentinel_skipped = []
    checked = 0
    expect_by_file = {f["file"]: f["expect"] for f in manifest["files"]}
    for row in golden:
        file, anchor = row.get("file"), row.get("anchor")
        if not file or not anchor:
            continue
        # 哨兵用例（GBK / 截断 PDF）本来就预期解析失败，锚点必然找不到 ——
        # 那不是标注错误，别让它污染"锚点校验"的结论
        if expect_by_file.get(file) not in (None, "ok"):
            sentinel_skipped.append((row["query"], file, expect_by_file.get(file)))
            continue
        checked += 1
        info = results.get(file)
        if info is None:
            anchor_bad.append((row["query"], file, "该文件不在 manifest 里"))
            continue
        if info["error"]:
            anchor_bad.append((row["query"], file, f"文件解析失败：{info['error'][:40]}"))
            continue
        if not any(anchor in t for t in info["child_texts"]):
            anchor_bad.append((row["query"], file, "锚点不在任何子块里"))
    print(f"校验 {checked} 条（负样本无锚点，不参与）")
    if sentinel_skipped:
        print(f"（{len(sentinel_skipped)} 条指向哨兵文件，按预期失败处理，不计入锚点校验）")
        for query, file, expect in sentinel_skipped:
            print(f"   · {query[:30]:<32} {file:<24} 预期={expect}")
    if anchor_bad:
        print(f"✗ {len(anchor_bad)} 条锚点有问题 —— 先修标注/生成器，否则召回率不可信：")
        for query, file, why in anchor_bad:
            print(f"   · {query[:34]:<36} {file:<30} {why}")
    else:
        print("✓ 全部锚点都能在对应文件的子块里找到")

    # ── 汇总 ────────────────────────────────────────────────────
    total_children = sum(r["children"] for r in results.values())
    total_chars = sum(r["chars"] for r in results.values())
    print()
    print(f"合计可入库子块：{total_children} 个（{total_chars / 10000:.1f} 万字符）")
    if unexpected:
        print(f"\n⚠ 与 manifest 预期不符 {len(unexpected)} 个：")
        for name, expect, actual in unexpected:
            print(f"   · {name}: 预期 {expect}，实际 {actual}")

    ok = not anchor_bad and not unexpected
    print("\n预检结论:", "通过，可以灌库" if ok else "未通过，先修上面的问题")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
