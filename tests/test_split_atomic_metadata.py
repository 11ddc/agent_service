"""两级切分的回归测试：**atomic 单元成父块边界 + 元数据必须传给子块**。

背景（实测数据）：v3 切分改造把 sections 拍平成一段文本再切，导致两件事同时坏掉：
  1. `Section.atomic`（表格/图片/扫描页）作废 —— 实测一个 3 工作表的 xlsx 被合并成
     **1 个父块**，跨表内容混在一起；超长表格会被切开、子块失去表头；
  2. 老 chunk 有 `breadcrumb/section_path/atomic/chunk_schema_ver/part` 5 个字段，
     新 chunk 一个都没有 → 生成侧无法把 [docN] 标注回章节，引用不可核验。

这里把两条都钉住（纯函数测试，不碰 MySQL/Chroma）。
"""

from rag import rag
from rag.structure import Section


def _sec(text, path, order, *, atomic=False, kind="text", page=None) -> Section:
    return Section(
        text=text,
        section_path=path,
        order=order,
        page_start=page,
        atomic=atomic,
        kind=kind,
    )


def test_atomic_section_is_its_own_parent():
    """atomic 单元必须是父块边界：3 张表 = 3 个父块，不能被合并。"""
    long_table = "| 型号 | 保修 |\n| --- | --- |\n" + "\n".join(
        f"| M{i} | {i * 12} 个月 |" for i in range(1, 60)
    )
    sections = [
        _sec(long_table, ["整机参数"], 0, atomic=True, kind="table"),
        _sec("| 备件 | 价格 |\n| --- | --- |\n| 电池 | 89 |", ["备件价格"], 1, atomic=True, kind="table"),
        _sec("| 城市 | 工单量 |\n| --- | --- |\n| 杭州 | 61 |", ["城市分布"], 2, atomic=True, kind="table"),
    ]

    parents, children = rag._split_document("F:/kb/参数表.xlsx", sections)

    assert len(parents) == 3, "每个 atomic section 应各成一个父块（拍平的话会合并）"
    assert [p["section_path"] for p in parents] == [["整机参数"], ["备件价格"], ["城市分布"]]
    assert all(p["atomic"] for p in parents)
    # 跨 sheet 内容不能混进同一个父块
    assert "电池" not in parents[0]["text"] and "杭州" not in parents[0]["text"]
    # 超长表也不能被切开（父块数不变，且内容完整）
    assert len(parents[0]["text"]) == len(long_table)


def test_children_inherit_breadcrumb_and_metadata():
    sections = [
        _sec("第3章 保修政策", ["手册.pdf", "第3章 保修政策"], 0),
        _sec("L1 Pro 整机保修期为 36 个月。", ["手册.pdf", "第3章 保修政策"], 1, page=7),
    ]

    parents, children = rag._split_document("F:/kb/手册.pdf", sections)

    assert parents[0]["breadcrumb"] == "手册.pdf > 第3章 保修政策"
    assert parents[1]["page_start"] == 7
    # 每个子块都继承**它所属父块**的出处（页码是按父块来的，不是全局）
    meta_by_parent = {m["parent_id"]: m for _c, _t, m in children}
    for parent in parents:
        meta = meta_by_parent[parent["pid"]]
        # 这 5 个字段是 v3 丢掉的，必须回来
        assert meta["breadcrumb"] == "手册.pdf > 第3章 保修政策"
        assert meta["section_path"] == "手册.pdf > 第3章 保修政策"
        assert meta["atomic"] is False
        assert meta["chunk_schema_ver"]
        if parent["page_start"] is None:
            assert "page_start" not in meta
        else:
            assert meta["page_start"] == parent["page_start"]


def test_metadata_omits_none_values_for_chroma():
    """Chroma 拒绝 None：取不到就**不写这个键**，不能写 0 假装有值。"""
    sections = [_sec("一段没有章节路径也没有页码的正文。", [], 0)]

    parents, children = rag._split_document("F:/kb/裸文本.txt", sections)

    assert parents[0]["breadcrumb"] is None and parents[0]["page_start"] is None
    for _cid, _text, meta in children:
        assert "breadcrumb" not in meta
        assert "section_path" not in meta
        assert "page_start" not in meta
        # 但身份/版本这些必须一直在
        for key in ("source", "parent_id", "child_id", "doc_id", "order_idx"):
            assert key in meta


def test_parent_ids_stay_deterministic_across_sections():
    """父块 id 仍按文档内序号生成：同一文件重复入库必须得到同样的 id。"""
    sections = [
        _sec("甲。", ["一"], 0),
        _sec("乙。", ["二"], 1, atomic=True, kind="table"),
        _sec("丙。", ["三"], 2),
    ]
    a1, _ = rag._split_document("F:/kb/x.md", sections)
    a2, _ = rag._split_document("F:/kb/x.md", sections)
    assert [p["pid"] for p in a1] == [p["pid"] for p in a2]
    assert len({p["pid"] for p in a1}) == len(a1), "父块 id 不能重复"
