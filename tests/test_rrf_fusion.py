"""RRF 融合的回归测试 —— 单路的第 1 名不能被"两路都出现但都不对"的块挤掉。

实测背景（488 块的语料库，5 条未命中里有 3 条就是这个原因）：
    一次解决率是什么意思？   dense=None  bm25=1  → 融合后 rrf=None
    装锁师傅多久能到我家？   dense=11    bm25=None → 融合后 rrf=None
    锁体安装孔距是多少毫米？ dense=None  bm25=11 → 融合后 rrf=None

机制：score = Σ 1/(k+rank)。
  - k=60 时，某一路的第 1 名是 1/61 = 0.0164，而**任何**两路都命中的块至少是
    2/(60+40) = 0.02 > 0.0164 —— 只要有 ≥20 个两路都命中的平庸块，单路第 1 名
    就一定掉出 top 20。
  - k=10 时，1/11 = 0.0909，两路块要超过它需要 rank ≤10，最多 10 个块能压住它，
    所以单路第 1 名稳定落在第 11 名以内（不会被浅候选池淹没）。
"""

import pytest
from langchain_core.documents import Document

import rag.rag as rag


def _doc(name: str) -> Document:
    """_doc_key 优先用 child_id 作身份，所以这里必须带 child_id。"""
    return Document(
        page_content=f"内容 {name}", metadata={"child_id": name, "source": f"{name}.md"}
    )


def _rank_of_correct(k: int, n_both: int, n_dense_only: int = 40) -> int | None:
    """构造：正确块只被 BM25 命中且排第 1；另有 n_both 个"两路都命中但都不对"的块。"""
    correct = _doc("correct")
    both = [_doc(f"b{i}") for i in range(n_both)]
    dense_only = [_doc(f"d{i}") for i in range(n_dense_only)]
    fused = rag._rrf_fuse(both + dense_only, [correct] + both, k=k, top_n=20)
    keys = [d.metadata["child_id"] for d in fused]
    return keys.index("correct") + 1 if "correct" in keys else None


def test_k10_keeps_single_channel_top1_inside_candidates():
    rank = _rank_of_correct(rag.RRF_K, n_both=25)
    assert rank is not None, "单路第 1 名被挤出融合候选 —— 这正是要修的回归"
    assert rank <= 20


def test_k60_would_lose_the_same_case():
    """把旧参数钉在测试里：k=60 在这个场景下确实会丢（证明修复是有针对性的）。"""
    rank = _rank_of_correct(60, n_both=25)
    assert rank is None or rank > 20


@pytest.mark.parametrize("n_both", [5, 20, 40, 60])
def test_robust_across_overlap_sizes(n_both):
    """无论两路重叠多少，单路第 1 名都不该掉出候选（旧参数在 ≥20 时就掉）。"""
    rank = _rank_of_correct(rag.RRF_K, n_both=n_both)
    assert rank is not None and rank <= 20


def test_fusion_still_rewards_agreement():
    """两路都命中应优于只被一路命中的同排名块 —— RRF 的基本性质不能改坏。"""
    agree, single = _doc("agree"), _doc("single")
    fused = rag._rrf_fuse([agree, single], [agree], top_n=10)
    assert [d.metadata["child_id"] for d in fused][0] == "agree"


def test_dedup_by_child_id_not_by_content():
    """同一个 child 在两路出现只能算一个身份，否则同一块被计两次分。"""
    a1 = Document(page_content="同样的正文", metadata={"child_id": "c1"})
    a2 = Document(page_content="同样的正文", metadata={"child_id": "c1"})
    fused = rag._rrf_fuse([a1], [a2], top_n=10)
    assert len(fused) == 1


def test_no_results_is_safe():
    assert rag._rrf_fuse([], [], top_n=10) == []
    only = [_doc("x")]
    assert len(rag._rrf_fuse(only, [], top_n=10)) == 1
