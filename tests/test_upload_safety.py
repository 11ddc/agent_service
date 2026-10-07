"""上传接口的安全回归测试 —— 零服务（init_rag 换成替身，不碰 Chroma / MySQL）。

覆盖三件事：
1. 文件名里带目录成分（`../`、`..\\`、绝对路径、**盘符相对路径**）必须被拒，
   且磁盘上不留任何文件；
2. 超过大小上限的请求必须在落盘前被拒，不能留下 .part 半截文件；
3. **纵深防御**：即使将来有人放宽了文件名校验，落盘前的"父目录必须是知识库目录"
   这条不变量也必须兜住。

背景：multipart 的 filename 是客户端随便写的字符串，httpx/浏览器都会原样带过来，
而 `save_dir / file.filename` 有三条逃逸路径：
  - `"../../main.py"`          爬到知识库目录之外；
  - `"C:/Windows/Temp/x.txt"`  绝对路径直接顶掉 save_dir；
  - `"C:evil.pdf"`             **盘符相对路径**：没有分隔符，靠"取最后一段再比对"
    挡不住，而 ntpath.join 遇到另一个盘符会整个丢弃 save_dir。
"""
import asyncio
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from api import upload_file as uf


def _client() -> TestClient:
    """只挂上传路由的最小 app：不必为了这个测试去 import 整张 LangGraph 图。"""
    app = FastAPI()
    app.include_router(uf.router, prefix="/api")
    return TestClient(app)


@pytest.fixture
def kb_dir(tmp_path, monkeypatch):
    """把知识库目录指到 tmp_path，测试永远不会碰到真实 knowledge_base/。"""
    d = tmp_path / "knowledge_base"
    monkeypatch.setattr(uf, "KNOWLEDGE_BASE_DIR", d)
    return d


def _saved_files(kb_dir: Path) -> list[str]:
    return sorted(p.name for p in kb_dir.rglob("*")) if kb_dir.exists() else []


# ==================== 正常路径：先证明测试本身没写错 ====================
def test_valid_upload_saves_file_and_indexes_it(kb_dir, monkeypatch):
    seen = {}

    def _fake_init_rag(path):
        seen["path"] = path
        return 7

    monkeypatch.setattr(uf, "init_rag", _fake_init_rag)

    resp = _client().post(
        "/api/upload",
        files={"file": ("客服手册.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )

    assert resp.status_code == 200
    assert resp.json() == {
        "success": True,
        "filename": "客服手册.pdf",
        "document_count": 7,
    }
    saved = kb_dir / "客服手册.pdf"
    assert saved.read_bytes() == b"%PDF-1.4 fake"
    assert seen["path"] == str(saved), "传给 init_rag 的应是落盘后的绝对路径"
    assert _saved_files(kb_dir) == ["客服手册.pdf"], "不该留下 .part 临时文件"


def test_unsupported_suffix_keeps_old_behaviour(kb_dir, monkeypatch):
    """后缀不支持仍是 200 + success=False（保持原有接口契约，本次不动它）。"""
    monkeypatch.setattr(uf, "init_rag", lambda p: pytest.fail("不该走到索引"))

    resp = _client().post("/api/upload", files={"file": ("x.exe", b"MZ", "application/x-dosexec")})

    assert resp.status_code == 200
    assert resp.json()["success"] is False
    assert _saved_files(kb_dir) == []


# ==================== 路径穿越：必须 400，且一个字节都不落盘 ====================
@pytest.mark.parametrize(
    "name",
    [
        "../../evil.pdf",  # 向上爬
        "..\\..\\evil.pdf",  # Windows 分隔符（POSIX 上不是分隔符，最容易漏）
        "C:/Windows/Temp/evil.pdf",  # 绝对路径顶掉 save_dir
        "C:evil.pdf",  # 盘符相对路径：没有分隔符，会绕过"取最后一段"的检查
        "c:evil.pdf",  # 小写盘符同理
        "NUL:evil.pdf",  # NTFS 交换数据流（ADS）：冒号是它的分隔符
        "CON.pdf",  # Windows 保留设备名：任何扩展名下都会被当成设备
        "aux.md",  # 实测写盘会抛 OSError（接口 500）
        "COM1.pdf",
        "lpt9.txt",
        "报表.pdf.",  # 结尾的点会被 Windows 静默去掉 → 存下来的名字与回报的不一致
        "/etc/passwd.md",
        "sub/dir/evil.md",  # 合法但带目录：一律拒绝，不做"帮我建子目录"
        "..",
        ".",
        "  ",
    ],
)
def test_path_in_filename_is_rejected(name, kb_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(uf, "init_rag", lambda p: pytest.fail("不该走到索引"))

    resp = _client().post(
        "/api/upload", files={"file": (name, b"payload", "application/pdf")}
    )

    assert resp.status_code == 400, resp.text
    assert _saved_files(kb_dir) == [], "被拒的请求不能在知识库里留下任何文件"
    assert not kb_dir.exists(), "连目录都不该被创建"
    # 穿越目标位置也不该出现文件
    assert not (tmp_path / "evil.pdf").exists()
    assert not (tmp_path.parent / "evil.pdf").exists()


def test_safe_filename_accepts_only_bare_names():
    assert uf._safe_filename("手册.pdf") == "手册.pdf"
    assert uf._safe_filename(" a.md ") == "a.md"
    for bad in (
        "../a.pdf",
        "..\\a.pdf",
        "d/a.pdf",
        "C:/a.pdf",
        "C:a.pdf",  # 盘符相对路径：曾经能绕过（ntpath.join 会丢掉 save_dir）
        "NUL:a.md",  # NTFS 交换数据流：冒号是分隔符
        "CON.pdf",  # Windows 保留设备名
        "aux.md",
        "a.pdf.",  # 结尾的点会被静默去掉
        "",
        "..",
        None,
    ):
        with pytest.raises(HTTPException) as exc:
            uf._safe_filename(bad)
        assert exc.value.status_code == 400


def test_safe_filename_keeps_names_that_merely_contain_reserved_words():
    """别把正常名字误伤：只有**词干等于**保留名才算设备名。"""
    for ok in ("Console.pdf", "auxiliary.md", "COM10.pdf", "nulls.txt", "回归测试.pdf"):
        assert uf._safe_filename(ok) == ok


def test_landing_path_must_stay_inside_the_knowledge_base(kb_dir, monkeypatch):
    """纵深防御：即使 `_safe_filename` 被放宽，落盘前的包含性断言也必须拦住。

    这里刻意把它换成一个"什么都放行"的实现（模拟一次错误的放宽），
    请求仍必须是 400，且磁盘上不能留下任何文件。
    """
    monkeypatch.setattr(uf, "_safe_filename", lambda raw: "../../evil.pdf")
    monkeypatch.setattr(uf, "init_rag", lambda p: pytest.fail("不该走到索引"))

    resp = _client().post(
        "/api/upload", files={"file": ("../../evil.pdf", b"payload", "application/pdf")}
    )

    assert resp.status_code == 400, resp.text
    assert _saved_files(kb_dir) == []


# ==================== 大小上限 ====================
def test_oversized_upload_is_rejected_before_writing(kb_dir, monkeypatch):
    """有 Content-Length 时走预检：连临时文件都不会出现。"""
    monkeypatch.setattr(uf, "MAX_UPLOAD_BYTES", 1024)
    monkeypatch.setattr(uf, "init_rag", lambda p: pytest.fail("不该走到索引"))

    resp = _client().post(
        "/api/upload", files={"file": ("big.pdf", b"x" * 4096, "application/pdf")}
    )

    assert resp.status_code == 413
    assert _saved_files(kb_dir) == []


class _FakeUpload:
    """size 取不到的 UploadFile（chunked 上传时拿不到长度）→ 走边收边判那条路。"""

    def __init__(self, filename, chunks):
        self.filename = filename
        self._chunks = list(chunks)
        self.size = None

    async def read(self, _n: int = -1) -> bytes:
        return self._chunks.pop(0) if self._chunks else b""


def test_oversized_upload_is_caught_while_streaming(kb_dir, monkeypatch):
    monkeypatch.setattr(uf, "MAX_UPLOAD_BYTES", 1024)
    monkeypatch.setattr(uf, "init_rag", lambda p: pytest.fail("不该走到索引"))

    upload = _FakeUpload("big.pdf", [b"x" * 800, b"y" * 800])

    with pytest.raises(HTTPException) as exc:
        asyncio.run(uf.upload_file(file=upload))

    assert exc.value.status_code == 413
    assert _saved_files(kb_dir) == [], "超限时必须清掉 .part 半截文件"
