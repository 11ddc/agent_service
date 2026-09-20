"""
Excel（.xlsx / .xlsm）解析入库的单测。

零服务、零网络：用 openpyxl 现场生成工作簿文件，直接调 rag 的 loader。

⚠️ 切分改成 split_parents / split_children（纯字符窗口）之后，有两处行为变更，
下面按**当前行为**断言，并在注释里标明是降级项：
1. 表格窗口不再"永不被二次切断"：1200 字的窗口会被 300 字窗口切开，续块没有表头
2. Excel 的 sheet/rows **不再进子块 metadata**，但仍写在正文里
   （"表格：z9_params（工作表：参数表，第 2-3 行）"），所以仍可被检索到
"""
from langchain_core.documents import Document
from openpyxl import Workbook, load_workbook

from rag import rag


def _make_xlsx(path, sheets: dict):
    """sheets: {"工作表名": [[单元格, ...], ...]} → 生成真实 .xlsx 文件"""
    wb = Workbook()
    wb.remove(wb.active)  # 去掉默认的 Sheet，避免多出空表干扰断言
    for name, rows in sheets.items():
        ws = wb.create_sheet(title=name)
        for row in rows:
            ws.append(row)
    wb.save(path)
    return str(path)


def _ingest(f: str):
    """走完整条切分链路：解析 → 拍平成文本 → 切父块 → 切子块。

    直接调 rag._split_document（纯函数，不碰 MySQL/Chroma），
    这样测的是真正入库用的那段代码，不是测试里重写的一份。
    """
    _, children = rag._split_document(f, rag._load_xlsx(f))
    return [Document(page_content=t, metadata=m) for _, t, m in children]


# ── _cell_text:单元格格式化 ──


def test_cell_text_integral_float_drops_trailing_zero():
    assert rag._cell_text(2999.0) == "2999"
    assert rag._cell_text(24.0) == "24"


def test_cell_text_keeps_real_decimal():
    assert rag._cell_text(4.35) == "4.35"


def test_cell_text_bool_before_number():
    # bool 是 int 的子类，判断顺序写反这里就会变成 "1"/"0"
    assert rag._cell_text(True) == "TRUE"
    assert rag._cell_text(False) == "FALSE"


def test_cell_text_date_and_datetime():
    from datetime import date, datetime

    assert rag._cell_text(date(2026, 8, 1)) == "2026-08-01"
    assert rag._cell_text(datetime(2026, 8, 1, 0, 0, 0)) == "2026-08-01"
    assert rag._cell_text(datetime(2026, 8, 1, 9, 23, 15)) == "2026-08-01 09:23:15"


def test_cell_text_none_and_inner_newline():
    assert rag._cell_text(None) == ""
    assert rag._cell_text("客户：你好\n订单没发货") == "客户：你好 订单没发货"


# ── _load_xlsx:基本结构 ──


def test_load_xlsx_builds_markdown_with_origin_in_text(tmp_path):
    f = _make_xlsx(
        tmp_path / "z9_params.xlsx",
        {"参数表": [["型号", "零售价", "保修"], ["Z9", 2999, "24个月"], ["Z9 Pro", 3999, "24个月"]]},
    )

    docs = _ingest(f)

    assert len(docs) == 1
    doc = docs[0]
    assert doc.metadata["source"] == f
    # ⚠️ 降级：sheet/rows 不再进 metadata，改为断言它们仍在正文里（可被检索到）
    assert "工作表：参数表" in doc.page_content
    assert "第 2-3 行" in doc.page_content  # 第 1 行是表头，数据从第 2 行起
    # 表头 + Markdown 分隔行 + 数据行
    assert "| 型号 | 零售价 | 保修 |" in doc.page_content
    assert "| --- | --- | --- |" in doc.page_content
    assert "| Z9 | 2999 | 24个月 |" in doc.page_content
    assert "z9_params" in doc.page_content  # 带表格名，块单独被检索出来也知道出处


def test_load_xlsx_sheet_name_is_in_text_not_metadata(tmp_path):
    f = _make_xlsx(
        tmp_path / "book.xlsx",
        {
            "价格": [["型号", "价格"], ["Z9", 2999]],
            "售后": [["型号", "保修"], ["Z9", "24个月"]],
        },
    )

    docs = _ingest(f)

    # ⚠️ 降级：原来靠 metadata["sheet"] 区分工作表，现在只能靠正文前缀
    body = "\n".join(d.page_content for d in docs)
    assert "工作表：价格" in body
    assert "工作表：售后" in body
    assert all(d.metadata["source"] == f for d in docs)


def test_load_xlsx_empty_sheet_returns_nothing(tmp_path):
    f = _make_xlsx(tmp_path / "empty.xlsx", {"空表": []})

    assert rag._load_xlsx(f) == []


def test_load_xlsx_skips_blank_rows_and_trims_trailing_columns(tmp_path):
    f = _make_xlsx(
        tmp_path / "gaps.xlsx",
        {"S": [["型号", "价格", None], [None, None, None], ["Z9", 2999, None]]},
    )

    docs = _ingest(f)

    body = docs[0].page_content
    assert "| 型号 | 价格 |" in body  # 行尾空列被裁掉
    assert body.count("| Z9 | 2999 |") == 1
    # 空行被跳过，但行号仍报 Excel 里的真实行号（第 2 行是空行，内容来自第 3 行）
    # → 所以行号区间可能有跳号，这是有意的：报的是"这块内容在源文件的哪一行"
    # ⚠️ 降级：原来是 metadata["rows"]，现在只能从正文里看到
    assert "第 3-3 行" in body


# ── 分窗:重复表头 + 表格不被二次切断 ──


def test_xlsx_windows_repeat_header_and_are_never_split(tmp_path):
    """表格窗口永不被切碎 —— 每个块都必须带表头。

    ⚠️ 这个保证在两轮改造里丢过一次、又补回来了：纯字符窗口时代表格会被腰斩
    （续块没有表头），现在 split_children 用 _is_table_block 判定"整块就是表格"
    就直接放行，所以保证重新成立。这条断言要是红了，说明表格保护又被拆掉了。
    """
    rows = [["型号", "零售价", "保修"]]
    rows += [[f"Z9-{i:03d}", 2000 + i, "24个月"] for i in range(200)]
    f = _make_xlsx(tmp_path / "big.xlsx", {"参数": rows})

    sections = rag._load_xlsx(f)
    docs = _ingest(f)
    header = "| 型号 | 零售价 | 保修 |"

    assert len(sections) > 1, "200 行应该被切成多个窗口"
    # 每个块都带表头:数据行脱离表头就没法理解
    assert all(header in d.page_content for d in docs), "有子块丢了表头 —— 表格被切碎了"
    # 表格行结构完整:没有任何一行被从中间截断
    for doc in docs:
        for line in doc.page_content.splitlines():
            if line.startswith("|"):
                assert line.endswith("|")


def test_load_xlsx_window_row_ranges_are_contiguous(tmp_path):
    """窗口的行号区间在 loader 层仍然连续（只是不再透传进子块 metadata）。"""
    rows = [["型号", "零售价", "保修"]]
    rows += [[f"Z9-{i:03d}", 2000 + i, "24个月"] for i in range(200)]
    f = _make_xlsx(tmp_path / "ranges.xlsx", {"参数": rows})

    sections = rag._load_xlsx(f)
    spans = []
    for s in sections:
        start, end = (int(x) for x in s.extra["rows"].split("-"))
        spans.append((start, end))

    assert spans[0][0] == 2  # 数据从第 2 行开始
    assert spans[-1][1] == 201  # 200 行数据 → 最后一行是 201
    for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
        assert next_start == prev_end + 1, "窗口之间不能漏行也不能重叠"


def test_load_xlsx_respects_total_row_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "XLSX_MAX_TOTAL_ROWS", 3)
    rows = [["型号", "价格"]] + [[f"Z9-{i}", i] for i in range(10)]
    f = _make_xlsx(tmp_path / "huge.xlsx", {"参数": rows})

    sections = rag._load_xlsx(f)

    # 表头 1 行 + 上限 3 行 = 4 行入库,其余丢弃
    covered = sum(
        int(s.extra["rows"].split("-")[1]) - int(s.extra["rows"].split("-")[0]) + 1
        for s in sections
    )
    assert covered == 3


# ── 分发与上传白名单 ──


def test_load_file_dispatches_xlsx(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(rag, "_load_xlsx", lambda path: calls.append(path) or [])

    rag._load_file(str(tmp_path / "a.xlsx"))
    rag._load_file(str(tmp_path / "a.xlsm"))

    assert len(calls) == 2


def test_load_file_rejects_legacy_xls(tmp_path):
    # 旧版二进制 .xls 不支持,应返回空而不是抛异常
    assert rag._load_file(str(tmp_path / "old.xls")) == []


def test_upload_whitelist_includes_excel():
    from api.upload_file import ALLOWED_SUFFIXES

    assert ".xlsx" in ALLOWED_SUFFIXES
    assert ".xlsm" in ALLOWED_SUFFIXES


def test_generated_file_is_readable_by_openpyxl(tmp_path):
    """兜底:确认测试造的文件本身是合法工作簿（避免测试假通过）"""
    f = _make_xlsx(tmp_path / "ok.xlsx", {"S": [["a", "b"], [1, 2]]})

    wb = load_workbook(f, data_only=True, read_only=True)
    try:
        assert wb.sheetnames == ["S"]
    finally:
        wb.close()
