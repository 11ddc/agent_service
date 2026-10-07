"""指标注册表测试（A5）—— 完全离线。

守的性质：

1. **counter 不能减**（Prometheus 的 counter 语义；允许减会让 rate() 算出负数）；
2. **同名指标幂等**：多个模块各自声明一次 `http_requests_total` 不能分裂成两条序列；
3. **histogram 的桶必须单调累加**，`+Inf` 桶等于总数 —— 否则
   `histogram_quantile()` 会算出离谱的分位；
4. 导出格式合法（每组都有 HELP/TYPE），可被采集器直接解析。
"""

import re

import pytest

from metrics import Counter, Histogram, Registry


@pytest.fixture
def reg() -> Registry:
    return Registry()


# ── counter ─────────────────────────────────────────────────
def test_counter_increments_and_reads_back(reg):
    c = reg.counter("t_total", "测试计数", ("a",))

    c.inc()
    c.inc(2)
    c.inc(a="x")

    assert c.value() == 3
    assert c.value(a="x") == 1


def test_counter_rejects_negative(reg):
    c = reg.counter("t_total", "测试计数")

    with pytest.raises(ValueError, match="不能减少"):
        c.inc(-1)


def test_same_name_returns_the_same_object(reg):
    """幂等：不同模块各声明一次，必须落到同一个序列。"""
    first = reg.counter("dup_total", "一")
    second = reg.counter("dup_total", "二（这个 help 会被忽略）")

    first.inc()
    assert second.value() == 1
    assert first is second


# ── histogram ───────────────────────────────────────────────
def test_histogram_counts_buckets_sum_and_count(reg):
    h = reg.histogram("d_seconds", "耗时", buckets=(0.1, 1.0))

    h.observe(0.05)
    h.observe(0.5)
    h.observe(5.0)

    assert h.count() == 3
    assert h.total() == pytest.approx(5.55)


def test_histogram_buckets_are_cumulative_and_inf_equals_count(reg):
    h = reg.histogram("d_seconds", "耗时", buckets=(0.1, 1.0))
    for value in (0.05, 0.5, 5.0):
        h.observe(value)

    text = "\n".join(h.render())
    # le=0.1 → 1 个；le=1.0 → 2 个；le=+Inf → 3 个（必须等于 count）
    assert 'd_seconds_bucket{le="0.1"} 1' in text
    assert 'd_seconds_bucket{le="1"} 2' in text
    assert 'd_seconds_bucket{le="+Inf"} 3' in text
    assert "d_seconds_count 3" in text


def test_histogram_quantiles_use_the_recorded_samples(reg):
    h = reg.histogram("d_seconds", "耗时")
    for value in range(1, 101):
        h.observe(value)

    qs = h.quantiles((0.5, 0.9, 0.99))

    assert qs[0.5] == pytest.approx(51, abs=1)
    assert qs[0.9] == pytest.approx(91, abs=1)
    assert 90 <= qs[0.9] <= 92


def test_histogram_ignores_nan_and_none(reg):
    """NaN 会让桶比较全部为 False 而落到 +Inf，污染分位 —— 直接丢弃。"""
    h = reg.histogram("d_seconds", "耗时")

    h.observe(float("nan"))
    h.observe(None)

    assert h.count() == 0


def test_histogram_labels_are_separate_series(reg):
    h = reg.histogram("d_seconds", "耗时", ("channel",))

    h.observe(0.5, channel="dense")
    h.observe(0.5, channel="sparse")
    h.observe(0.5, channel="sparse")

    assert h.count(channel="dense") == 1
    assert h.count(channel="sparse") == 2


# ── 导出格式 ────────────────────────────────────────────────
def test_registry_render_is_parseable_prometheus_text(reg):
    reg.counter("a_total", "计数", ("k",)).inc(k="v")
    reg.histogram("b_seconds", "耗时").observe(0.3)

    text = reg.render()

    # 每个指标都要有 HELP 与 TYPE
    assert "# HELP a_total 计数" in text
    assert "# TYPE a_total counter" in text
    assert "# TYPE b_seconds histogram" in text

    # 每行要么是注释，要么是 `名字{标签} 值`
    line_re = re.compile(
        r'^(#|(?:[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})? \S+$)'
    )
    for line in text.strip().splitlines():
        assert line_re.match(line), f"格式不合法: {line!r}"

    assert 'a_total{k="v"} 1' in text


def test_render_is_sorted_for_stable_output(reg):
    """顺序稳定：否则每次抓取都产生无意义的 diff（也会让快照测试抖动）。

    注：没被打过点的 counter **不会**出现采样行（Prometheus 的常规行为：
    序列在第一次自增后才存在），所以这里先各打一次。
    """
    reg.counter("zzz_total", "z").inc()
    reg.counter("aaa_total", "a").inc()

    lines = [l for l in reg.render().splitlines() if not l.startswith("#")]

    assert lines == ["aaa_total 1", "zzz_total 1"]


def test_counter_without_samples_renders_nothing(reg):
    """只有 HELP/TYPE、没有采样行 —— 这也是合法且常见的。"""
    reg.counter("never_total", "从未自增")

    text = reg.render()
    samples = [l for l in text.splitlines() if l and not l.startswith("#")]

    assert "# TYPE never_total counter" in text
    assert samples == [], f"不该有采样行: {samples}"


def test_reset_clears_everything(reg):
    reg.counter("t_total", "x").inc()

    reg.reset()

    assert reg.render() == "\n"
