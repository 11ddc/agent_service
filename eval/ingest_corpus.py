"""把语料灌进知识库 —— **走真实 HTTP 接口**（POST /api/upload），不直接调 init_rag。

为什么坚持走接口：接口层也是被测对象。直接调 init_rag 会绕过文件名清洗、大小上限、
落盘方式、事件循环阻塞这些真实存在的风险面。

用法：
    # 先起服务：venv\\Scripts\\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
    venv\\Scripts\\python.exe eval\\ingest_corpus.py

做的事：
  1. 按 manifest 逐个上传，记录 HTTP 状态、耗时、返回的 document_count；
  2. 把实际结果与 manifest 的预期对比（ok/fail/empty/reject 四类哨兵）；
  3. 额外发两个**安全探针**（路径穿越文件名、超大文件），验证上传防护在真实服务里生效；
  4. 落盘 eval/reports/ingest_report.json。

注意：这一步会真实调用 DashScope embedding（每块一次），是有成本的。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # 脚本在 eval/ 下跑，项目根目录要手动挂上才能 import db / rag
    sys.path.insert(0, str(ROOT))
CORPUS = ROOT / "eval" / "corpus"
REPORT = ROOT / "eval" / "reports" / "ingest_report.json"
BASE = "http://127.0.0.1:8000"
TIMEOUT = 600.0  # 单个大文件（含 OCR/embedding）可能跑几分钟


def wait_ready(client: httpx.Client, seconds: int = 180) -> bool:
    """等服务就绪：冷启动 import 要 40~50 秒，不是服务坏了。"""
    t0 = time.time()
    while time.time() - t0 < seconds:
        try:
            r = client.get(f"{BASE}/openapi.json", timeout=5)
            if r.status_code == 200:
                print(f"服务就绪（等待 {time.time() - t0:.1f}s）")
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2)
    return False


def upload(client: httpx.Client, path: Path) -> dict:
    t0 = time.perf_counter()
    with path.open("rb") as fh:
        files = {"file": (path.name, fh, "application/octet-stream")}
        r = client.post(f"{BASE}/api/upload", files=files, timeout=TIMEOUT)
    ms = (time.perf_counter() - t0) * 1000
    try:
        body = r.json()
    except Exception:  # noqa: BLE001
        body = {"_raw": r.text[:200]}
    return {"status": r.status_code, "ms": round(ms), "body": body}


def verdict(expect: str, res: dict) -> tuple[bool, str]:
    """实际结果 vs manifest 预期。"""
    status, body = res["status"], res["body"]
    ok_flag = body.get("success")
    count = body.get("document_count")
    if expect == "ok":
        good = status == 200 and ok_flag is True and (count or 0) > 0
        return good, f"count={count}"
    if expect == "fail":
        # 解析失败：要么 500（异常上抛），要么 200 但 0 块（被吞掉）
        good = status >= 500 or (status == 200 and (count == 0))
        return good, f"status={status} count={count}（失败是否留痕见 documents 表）"
    if expect == "empty":
        return status == 200 and count == 0, f"status={status} count={count}"
    if expect == "reject":
        return status == 200 and ok_flag is False, f"status={status} error={str(body.get('error'))[:40]}"
    return False, "未知预期"


def security_probes(client: httpx.Client) -> list[dict]:
    """两个探针，验的是"上传防护在真实服务里生效"，不是单元测试。"""
    out = []
    # 1) 路径穿越文件名：必须 400，且不能在知识库外留下文件
    r = client.post(
        f"{BASE}/api/upload",
        files={"file": ("../../evil.pdf", b"%PDF-1.4 pwned", "application/pdf")},
        timeout=30,
    )
    outside = (ROOT.parent / "evil.pdf").exists()
    out.append(
        {
            "probe": "路径穿越文件名",
            "status": r.status_code,
            "detail": str(r.json())[:80],
            "pass": r.status_code == 400 and not outside,
            "note": f"知识库外是否被写入={outside}",
        }
    )
    # 2) 超大文件：必须 413（上限 50MB，这里发 51MB）
    big = b"x" * (51 * 1024 * 1024)
    r = client.post(
        f"{BASE}/api/upload",
        files={"file": ("huge.pdf", big, "application/pdf")},
        timeout=120,
    )
    leftovers = list((ROOT / "knowledge_base").glob("*.part")) if (ROOT / "knowledge_base").exists() else []
    out.append(
        {
            "probe": "超大文件(51MB)",
            "status": r.status_code,
            "detail": str(r.json())[:80],
            "pass": r.status_code == 413 and not leftovers,
            "note": f"遗留 .part 文件={len(leftovers)}",
        }
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="把语料灌进知识库（走真实 HTTP 接口）")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="跳过 documents 表里已解析成功的文档（中断/欠费后续跑用，避免重复花 embedding 额度）",
    )
    args = ap.parse_args()

    manifest = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    files = manifest["files"]
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    # MySQL 预检：**必须**。父块与文档元数据只有 MySQL 有，而 db.mysql 的设计是
    # "不可用就静默降级" —— 实测踩过一次：MySQL 挂着灌完 104 份文档，子块全进了
    # Chroma、接口全部返回 success，但 parents/documents 一条都没写，
    # 报告里却写着"全部符合预期"。这类静默降级必须在开跑前挡住。
    try:
        from db import mysql as mysql_mod

        ok, info = mysql_mod.ping()
    except ImportError as e:
        # 别把"导入失败"误报成"MySQL 挂了"（真踩过：脚本没把项目根目录挂进 sys.path）
        ok, info = False, f"无法导入 db 包（{e}）——检查脚本的 sys.path 设置"
    except Exception as e:  # noqa: BLE001
        ok, info = False, f"{type(e).__name__}: {e}"
    if not ok:
        print(
            f"MySQL 不可用（{info}）——父块与文档元数据会全部丢失，先启动 MySQL 再灌库",
            file=sys.stderr,
        )
        return 3
    print(f"MySQL 预检通过：{info}")

    # 续跑：documents 表里已 ok 的文档直接跳过。
    # 动机很实际——本轮 L 档灌库中途遇到 DashScope 账户欠费（code=Arrearage），
    # 134 份已成功、33 份失败；充值后如果整批重跑，那 134 份的 embedding 要重花一遍。
    done: set[str] = set()
    if args.resume:
        from db import DocumentStore

        done = {d.source for d in DocumentStore().all() if d.parsed_status == "ok"}
        print(f"续跑模式：documents 表里已成功 {len(done)} 份，将跳过")

    with httpx.Client() as client:
        if not wait_ready(client):
            print(f"服务未就绪：{BASE}（先起 uvicorn main:app --port 8000）", file=sys.stderr)
            return 2

        rows, bad = [], []
        skipped = 0
        print()
        print(f"{'文件':<34}{'预期':<7}{'状态':>5}{'耗时ms':>8}{'块数':>6}  结论")
        print("-" * 88)
        for f in files:
            path = CORPUS / f["file"]
            # documents.source 记的是 knowledge_base/ 下的绝对路径（接口落盘位置）
            kb_path = str(ROOT / "knowledge_base" / f["file"])
            if args.resume and f["expect"] == "ok" and kb_path in done:
                skipped += 1
                continue
            res = upload(client, path)
            good, detail = verdict(f["expect"], res)
            body = res["body"]
            rows.append(
                {
                    "file": f["file"],
                    "format": f["format"],
                    "size": f["size"],
                    "purpose": f["purpose"],
                    "expect": f["expect"],
                    **res,
                    "pass": good,
                    "detail": detail,
                }
            )
            if not good:
                bad.append(f["file"])
            print(
                f"{f['file']:<34}{f['expect']:<7}{res['status']:>5}{res['ms']:>8}"
                f"{str(body.get('document_count', '-')):>6}  {'✓' if good else '✗'} {detail[:38]}"
            )

        print()
        probes = security_probes(client)

    ok_files = [r for r in rows if r["expect"] == "ok" and r["pass"]]
    total_chunks = sum((r["body"].get("document_count") or 0) for r in rows if r["expect"] == "ok")
    total_ms = sum(r["ms"] for r in rows)
    print("=" * 88)
    print(f"入库成功 {len(ok_files)}/{len([r for r in rows if r['expect'] == 'ok'])} 个正式文档，"
          f"共 {total_chunks} 个子块，总耗时 {total_ms / 1000:.1f}s")
    for p in probes:
        print(f"安全探针 {p['probe']:<16} status={p['status']:<4} {'✓ 通过' if p['pass'] else '✗ 未通过'}  {p['note']}")

    report = {
        "base": BASE,
        "files": rows,
        "security_probes": probes,
        "summary": {
            "ok_docs": len(ok_files),
            "expected_ok": len([r for r in rows if r["expect"] == "ok"]),
            "total_chunks": total_chunks,
            "total_seconds": round(total_ms / 1000, 1),
            "failed_expectations": bad,
            "security_probes_passed": all(p["pass"] for p in probes),
        },
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告: {REPORT}")
    all_pass = not bad and all(p["pass"] for p in probes)
    print("灌库结论:", "全部符合预期" if all_pass else "有不符合预期的项，见上表")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
