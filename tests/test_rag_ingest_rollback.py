"""init_rag 写入顺序的回归测试 —— 零服务（Chroma / MySQL 全换替身）。

锁住一件事：**子块是"先加后清"**。

原来是 `store.delete(where={"source": ...})` 然后 `add_texts`。add 失败
（embedding 额度 / 网络抖动 —— 正是外面那层重试要防的场景）时：
  - 旧子块已经删光 → 文档检索不到了
  - 新父块已经写进 MySQL → 库里还留着孤儿父块
  - documents 表仍是上次的 parsed_status='ok' → 没有任何失败痕迹
也就是说"重新上传失败"会静默毁掉一份已经入库的文档。这两条路径现在都被钉住。
"""
import pytest

import rag.rag as rag
from rag.structure import Section


class _FakeStore:
    """只实现 init_rag 用到的那几个方法，并记录调用顺序。"""

    def __init__(self, existing_ids=(), fail_add=False):
        self.existing = list(existing_ids)
        self.events = []
        self._fail_add = fail_add

    def add_texts(self, texts, metadatas, ids):
        self.events.append(("add", list(ids)))
        if self._fail_add:
            raise RuntimeError("embedding 503")
        self.existing = list(dict.fromkeys([*self.existing, *ids]))

    def get(self, where=None, **kwargs):
        return {"ids": list(self.existing)}

    def delete(self, ids=None, where=None, **kwargs):
        # 先加后清的关键：清理只能按精确 id 删，绝不能按 source 全删
        assert where is None, "不允许再按 source 整片删除"
        self.events.append(("delete", list(ids or [])))
        gone = set(ids or [])
        self.existing = [i for i in self.existing if i not in gone]


def _wire(monkeypatch, store):
    """换掉 init_rag 的所有外部依赖，只保留纯函数的切分逻辑。"""
    monkeypatch.setattr(
        rag,
        "_load_file",
        lambda _p: [
            Section(text="退换货政策：7 天无理由退货。", section_path=["售后"], order=0)
        ],
    )
    monkeypatch.setattr(rag, "_save_parents", lambda source, parents: len(parents))
    # embedding 现在是工厂函数（本地模型或云端由 config 决定），patch 掉它避免真加载模型
    monkeypatch.setattr(rag, "get_embeddings", lambda: object())
    monkeypatch.setattr(rag, "Chroma", lambda **kw: store)

    records = []
    monkeypatch.setattr(
        rag,
        "_record_document",
        lambda source, chunk_count, parent_count, error=None: records.append(
            {"chunks": chunk_count, "parents": parent_count, "error": error}
        ),
    )
    return records


def test_success_path_adds_before_deleting_stale_ids(monkeypatch):
    store = _FakeStore(existing_ids=["p_old|0", "p_old|1"])
    records = _wire(monkeypatch, store)

    n = rag.init_rag("F:/kb/手册.pdf")

    assert n == 1
    assert [e[0] for e in store.events] == ["add", "delete"], "必须是先 add 后 delete"
    added = set(store.events[0][1])
    deleted = set(store.events[1][1])
    assert deleted == {"p_old|0", "p_old|1"}, "只删本次没写进去的旧块"
    assert deleted.isdisjoint(added), "新块不能被自己的清理步骤删掉"
    assert set(store.existing) == added, "清理后库里恰好剩新块"
    assert records == [{"chunks": 1, "parents": 1, "error": None}]


def test_reupload_with_same_ids_deletes_nothing(monkeypatch):
    """同一个文件原样重传：id 是确定性的，新块集合 == 旧块集合 → 不该删任何东西。"""
    store = _FakeStore()
    _wire(monkeypatch, store)
    rag.init_rag("F:/kb/手册.pdf")  # 第一次入库

    store.events.clear()
    rag.init_rag("F:/kb/手册.pdf")  # 原样重传

    assert [e[0] for e in store.events] == ["add"], "没有旧块要清时不该调 delete"


def test_failed_add_keeps_old_blocks_and_records_failure(monkeypatch):
    store = _FakeStore(existing_ids=["keep|0", "keep|1"], fail_add=True)
    records = _wire(monkeypatch, store)

    with pytest.raises(RuntimeError, match="embedding 503"):
        rag.init_rag("F:/kb/手册.pdf")

    assert [e[0] for e in store.events] == ["add"], "add 失败后不允许再删任何东西"
    assert store.existing == ["keep|0", "keep|1"], "旧块必须原样留着，否则文档直接查不到"
    assert len(records) == 1, "失败必须写进 documents 表"
    assert records[0]["error"] and "embedding 503" in records[0]["error"]
    assert (records[0]["chunks"], records[0]["parents"]) == (0, 0)


def test_parse_exception_is_recorded_and_reraised(monkeypatch):
    """解析阶段就抛异常（GBK 编码 / 损坏 PDF）也必须留痕。

    实测过的漏洞：`_load_file` 抛 UnicodeDecodeError / FileDataError 时，
    documents 表一个字都没写 → 接口 500，但库里查不到这个文件来过。
    """
    store = _FakeStore()
    records = _wire(monkeypatch, store)

    def _boom(_path):
        raise UnicodeDecodeError("utf-8", b"\xdc", 0, 1, "invalid start byte")

    monkeypatch.setattr(rag, "_load_file", _boom)

    with pytest.raises(UnicodeDecodeError):
        rag.init_rag("F:/kb/术语表_GBK.txt")

    assert store.events == [], "解析失败时不该碰 Chroma"
    assert len(records) == 1, "解析失败必须写进 documents 表"
    assert records[0]["error"] and "UnicodeDecodeError" in records[0]["error"]
    assert (records[0]["chunks"], records[0]["parents"]) == (0, 0)


def test_empty_split_does_not_wipe_existing_blocks(monkeypatch):
    """切不出子块时（罕见）宁可留着旧版本，也不要把文档清空。"""
    store = _FakeStore(existing_ids=["keep|0"])
    records = _wire(monkeypatch, store)
    monkeypatch.setattr(rag, "_split_document", lambda source, sections: ([], []))

    assert rag.init_rag("F:/kb/手册.pdf") == 0

    assert store.events == [], "既不该 add 也不该 delete"
    assert store.existing == ["keep|0"]
    assert records == [{"chunks": 0, "parents": 0, "error": None}]
