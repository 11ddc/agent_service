"""BM25 索引磁盘缓存测试（A7 的"外置"部分）—— 完全离线。

守的性质：

1. **缓存能来回**（save→load 拿到同样的语料与 metadata），且写的是
   **JSON+gzip 而不是 pickle**：pickle 反序列化会执行代码，一个可写的缓存文件
   就等于一个后门；
2. **不一致就当作没有缓存**：块数变了、版本变了、文件坏了 → 重建，绝不静默用脏数据；
3. ⚠️ **元数据变化必须绕过缓存**：`Acl.allows()` 判权限用的是这里缓存的 metadata。
   发布/下架/改可见性**不改块数** —— 如果那时从缓存恢复，就会拿旧 status 判权限，
   表现为"已下架的文档仍被检索到"且不报错。
4. 缓存写失败/目录不可写不能影响检索。
"""

import gzip
import json

import pytest
from langchain_core.documents import Document

import rag.rag as rag


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "BM25_CACHE_DIR", tmp_path / "bm25_cache")
    monkeypatch.setattr(rag, "_bm25_skip_cache", False)
    return tmp_path / "bm25_cache"


def _docs(n: int = 3):
    return [
        Document(page_content=f"第{i}块正文", metadata={"source": f"s{i}.pdf", "status": "published"})
        for i in range(n)
    ]


# ══════════════════════════════════════════════════════════════
# 存取往返
# ══════════════════════════════════════════════════════════════
def test_save_then_load_roundtrip(cache_dir):
    docs = _docs(3)
    tokens = [["第", "0"], ["第", "1"], ["第", "2"]]

    rag._save_bm25_cache(3, tokens, docs)
    loaded = rag._load_bm25_cache(3)

    assert loaded is not None
    bm25, got_tokens, got_docs = loaded
    assert got_tokens == tokens
    assert [d.page_content for d in got_docs] == [d.page_content for d in docs]
    # metadata 必须一起还原：ACL 判定依赖它（tenant_id/visibility/status）
    assert got_docs[0].metadata["status"] == "published"
    assert bm25 is not None


def test_cache_file_is_gzip_json_not_pickle(cache_dir):
    """⚠️ 写成 pickle 就等于给"能写这个文件的人"开了一个代码执行入口。"""
    rag._save_bm25_cache(1, [["a"]], _docs(1))

    path = next(cache_dir.glob("*.json.gz"))
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        data = json.load(fh)  # 能被 json 解析 → 不是 pickle

    assert set(data) == {"version", "collection", "count", "tokens", "texts", "metadatas"}
    assert "cos" not in str(data)  # 不是 pickle 协议的二进制串


def test_write_is_atomic_no_tmp_left_behind(cache_dir):
    rag._save_bm25_cache(2, [["a"], ["b"]], _docs(2))

    assert not list(cache_dir.glob("*.tmp")), "临时文件必须被 os.replace 掉"


# ══════════════════════════════════════════════════════════════
# 不一致 → 当作没有缓存
# ══════════════════════════════════════════════════════════════
def test_count_mismatch_is_ignored(cache_dir):
    rag._save_bm25_cache(3, [["a"]] * 3, _docs(3))

    assert rag._load_bm25_cache(5) is None, "块数对不上必须重建（文件名不同）"


def test_version_mismatch_is_ignored(cache_dir):
    rag._save_bm25_cache(3, [["a"]] * 3, _docs(3))
    path = next(cache_dir.glob("*.json.gz"))
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    data["version"] = "v0-old"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(data, fh)

    assert rag._load_bm25_cache(3) is None


def test_corrupt_file_is_ignored_without_raising(cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    rag._bm25_cache_path(3).write_bytes(b"this is not gzip at all")

    assert rag._load_bm25_cache(3) is None


def test_truncated_payload_is_ignored(cache_dir):
    """文件能解开但条数对不上：同样必须重建，不能拿半份语料去检索。"""
    rag._save_bm25_cache(3, [["a"]] * 3, _docs(3))
    path = next(cache_dir.glob("*.json.gz"))
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    data["tokens"] = data["tokens"][:1]  # 语料被截断
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(data, fh)

    assert rag._load_bm25_cache(3) is None


def test_missing_cache_returns_none(cache_dir):
    assert rag._load_bm25_cache(3) is None


def test_write_failure_does_not_raise(monkeypatch, tmp_path):
    """缓存目录不可写时只告警：检索不能因为缓存坏掉而不可用。"""
    monkeypatch.setattr(rag, "BM25_CACHE_DIR", tmp_path / "a" / "b")

    def _boom(*_a, **_kw):
        raise OSError("磁盘满")

    monkeypatch.setattr(rag.gzip, "open", _boom)

    rag._save_bm25_cache(1, [["a"]], _docs(1))  # 不抛即通过


# ══════════════════════════════════════════════════════════════
# 构建流程：用缓存 / 绕过缓存
# ══════════════════════════════════════════════════════════════
class _FakeCollection:
    def __init__(self, docs):
        self._docs = docs
        self.get_calls = 0

    def count(self):
        return len(self._docs)

    def get(self, include=None):
        self.get_calls += 1
        return {
            "documents": [d.page_content for d in self._docs],
            "metadatas": [d.metadata for d in self._docs],
        }


class _FakeStore:
    def __init__(self, docs):
        self._collection = _FakeCollection(docs)


@pytest.fixture
def store(monkeypatch):
    docs = _docs(3)
    fake = _FakeStore(docs)
    monkeypatch.setattr(rag, "_vectorstore", fake)
    monkeypatch.setattr(rag, "_bm25", None)
    monkeypatch.setattr(rag, "_chunk_docs", [])
    monkeypatch.setattr(rag, "_bm25_corpus", [])
    return fake


def test_build_uses_the_cache_and_skips_chroma(store, cache_dir):
    """冷启动命中缓存时不该再全量拉 Chroma + 重新分词。"""
    rag._save_bm25_cache(3, [["第", "0"], ["第", "1"], ["第", "2"]], _docs(3))

    rag._build_bm25_index()

    assert store._collection.get_calls == 0, "命中缓存却仍去拉了 Chroma"
    assert len(rag._chunk_docs) == 3


def test_build_falls_back_to_chroma_when_cache_missing(store, cache_dir):
    rag._build_bm25_index()

    assert store._collection.get_calls == 1
    assert len(rag._chunk_docs) == 3
    # 构建成功后应当落盘，供下次冷启动使用
    assert list(cache_dir.glob("*.json.gz")), "构建后应当写缓存"


def test_invalidate_bypasses_the_disk_cache(store, cache_dir):
    """⚠️ 权限变更后必须重读 Chroma。

    这条是 A4 那条 ACL 规则的延伸：`Acl.allows()` 用的是这里的 metadata，
    而发布/下架/改可见性**不改块数** —— 从缓存恢复就会拿旧 status 判权限，
    表现为"已下架的文档仍被检索到"，且完全不报错。
    """
    rag._save_bm25_cache(3, [["第", "0"], ["第", "1"], ["第", "2"]], _docs(3))
    before = store._collection.get_calls

    rag.invalidate_index_cache()   # 模拟"刚下架了一份文档"
    rag._build_bm25_index()

    assert store._collection.get_calls == before + 1, "失效后仍然用了磁盘缓存"
    assert rag._bm25_skip_cache is False, "重建完成后应当复位"


def test_metadata_from_cache_reflects_stored_status(store, cache_dir):
    """缓存里的 metadata 就是权限判据 —— 它必须原样带回来。"""
    docs = [
        Document(page_content="已发布", metadata={"status": "published", "tenant_id": "t"}),
        Document(page_content="已下架", metadata={"status": "archived", "tenant_id": "t"}),
    ]
    rag._save_bm25_cache(2, [["a"], ["b"]], docs)
    store._collection._docs = docs

    rag._build_bm25_index()

    assert [d.metadata["status"] for d in rag._chunk_docs] == ["published", "archived"]


# ══════════════════════════════════════════════════════════════
# 清理
# ══════════════════════════════════════════════════════════════
def test_clear_cache_removes_files(cache_dir):
    rag._save_bm25_cache(3, [["a"]] * 3, _docs(3))
    rag._save_bm25_cache(4, [["a"]] * 4, _docs(4))

    removed = rag.clear_bm25_cache()

    assert removed == 2
    assert not list(cache_dir.glob("*.json.gz"))


def test_clear_cache_on_missing_dir_is_safe(cache_dir):
    rag.clear_bm25_cache()

    assert rag.clear_bm25_cache() == 0


def test_clear_cache_ignores_other_collections(cache_dir, monkeypatch):
    """只清本集合的缓存：换 embedding 模型时集合名不同，不该误删别人的。"""
    rag._save_bm25_cache(3, [["a"]] * 3, _docs(3))
    other = cache_dir / "other_collection-v1-9.json.gz"
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_bytes(b"x")

    rag.clear_bm25_cache()

    assert other.exists()
