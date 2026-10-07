"""进程内指标注册表 —— Prometheus 文本格式导出，**零依赖**。

## 为什么企业里必须有

客服系统要能回答这些问题，否则"变慢/变差"只能靠用户投诉才发现：

- 现在 QPS 多少、延迟 p50/p90/p99 多少？
- 检索命中率多少、**兜底率**多少、转人工率多少？
- 每个会话花多少 token / 多少钱？

## 为什么自己写而不是引 prometheus_client

本项目对依赖已经够克制（而且刚因为"依赖清单漏声明"吃过一次亏）。需要的只是
counter + histogram，几十行就能覆盖，并且**离线可测**。将来要接正式监控栈时，
把 `render()` 换成 prometheus_client 的实现即可，**打点代码一行都不用改**。

## 百分位怎么算

用 Prometheus 的标准做法：导出的 `_bucket{le=...}` 交给 PromQL 的
`histogram_quantile()` 算分位。这里额外保留一个有界样本队列，
`quantiles()` 可以**就地**给出精确分位（给 /health 这类自检用），
不依赖任何查询端。

## 线程安全

打点会从工作线程（`asyncio.to_thread` 里的图节点）调用，所以每个指标对象内部
都加锁。锁的粒度是"单个指标"，不同指标之间不互相阻塞。
"""

import math
import threading
from bisect import insort
from collections import deque
from dataclasses import dataclass, field

# 延迟桶（秒）：覆盖从缓存命中到"模型很慢"的整个区间
DEFAULT_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
)

_MAX_SAMPLES = 2048  # 每标签组合保留的样本数上限（够算 p50/p90/p99，且不会无界增长）


def _key(labels: dict) -> tuple:
    return tuple(sorted((k, str(v)) for k, v in (labels or {}).items()))


class _Base:
    def __init__(self, name: str, help_text: str, label_names: tuple[str, ...] = ()):
        self.name = name
        self.help = help_text
        self.label_names = tuple(label_names)
        self._lock = threading.Lock()

    def _fmt_labels(self, key: tuple, extra: tuple | None = None) -> str:
        parts = [f'{k}="{v}"' for k, v in key]
        if extra:
            parts.extend(extra)
        return "{" + ",".join(parts) + "}" if parts else ""


@dataclass
class Counter(_Base):
    """单调递增计数。"""

    def __init__(self, name, help_text, label_names=()):
        super().__init__(name, help_text, label_names)
        self._values: dict[tuple, float] = {}

    def inc(self, amount: float = 1, **labels) -> None:
        if amount < 0:
            raise ValueError("计数器不能减少（Prometheus 的 counter 语义）")
        key = _key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def value(self, **labels) -> float:
        return self._values.get(_key(labels), 0.0)

    def render(self) -> list[str]:
        out = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        with self._lock:
            items = sorted(self._values.items())
        for key, val in items:
            out.append(f"{self.name}{self._fmt_labels(key)} {val:g}")
        return out


class Histogram(_Base):
    """分布统计：桶 + 总和 + 计数（可算分位）。"""

    def __init__(self, name, help_text, label_names=(), buckets=DEFAULT_BUCKETS):
        super().__init__(name, help_text, label_names)
        self.buckets = tuple(sorted(buckets))
        self._counts: dict[tuple, list[int]] = {}
        self._sums: dict[tuple, float] = {}
        self._totals: dict[tuple, int] = {}
        self._samples: dict[tuple, deque] = {}

    def observe(self, value: float, **labels) -> None:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return
        value = float(value)
        key = _key(labels)
        idx = 0
        for i, edge in enumerate(self.buckets):
            if value <= edge:
                idx = i
                break
        else:
            idx = len(self.buckets)  # +Inf 桶
        with self._lock:
            counts = self._counts.setdefault(key, [0] * (len(self.buckets) + 1))
            counts[idx] += 1
            self._sums[key] = self._sums.get(key, 0.0) + value
            self._totals[key] = self._totals.get(key, 0) + 1
            samples = self._samples.setdefault(key, deque(maxlen=_MAX_SAMPLES))
            insort(samples, value)

    def count(self, **labels) -> int:
        return self._totals.get(_key(labels), 0)

    def total(self, **labels) -> float:
        return self._sums.get(_key(labels), 0.0)

    def quantiles(self, qs=(0.5, 0.9, 0.99), **labels) -> dict[float, float]:
        """就地从有界样本算分位（样本不足时按现有样本算）。"""
        samples = self._samples.get(_key(labels))
        if not samples:
            return {q: 0.0 for q in qs}
        data = list(samples)  # 已排序
        out = {}
        for q in qs:
            pos = min(len(data) - 1, max(0, int(round(q * (len(data) - 1)))))
            out[q] = data[pos]
        return out

    def render(self) -> list[str]:
        out = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        with self._lock:
            keys = sorted(set(self._counts) | set(self._sums))
            snapshot = {
                k: (
                    list(self._counts.get(k, [0] * (len(self.buckets) + 1))),
                    self._sums.get(k, 0.0),
                    self._totals.get(k, 0),
                )
                for k in keys
            }
        for key in keys:
            counts, total_sum, total_count = snapshot[key]
            cumulative = 0
            for i, edge in enumerate(self.buckets):
                cumulative += counts[i]
                out.append(
                    f"{self.name}_bucket{self._fmt_labels(key, (f'le=\"{edge:g}\"',))} {cumulative}"
                )
            cumulative += counts[len(self.buckets)]
            out.append(
                f"{self.name}_bucket{self._fmt_labels(key, ('le=\"+Inf\"',))} {cumulative}"
            )
            out.append(f"{self.name}_sum{self._fmt_labels(key)} {total_sum:g}")
            out.append(f"{self.name}_count{self._fmt_labels(key)} {total_count}")
        return out


class Registry:
    """指标注册表。

    同名指标重复注册时返回**同一个对象**（幂等）—— 打点代码可能在多个模块里
    各自声明一次，不能因此分裂成两个序列。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._metrics: dict[str, _Base] = {}

    def counter(self, name: str, help_text: str, label_names=()) -> Counter:
        return self._get(name, lambda: Counter(name, help_text, label_names))

    def histogram(
        self, name: str, help_text: str, label_names=(), buckets=DEFAULT_BUCKETS
    ) -> Histogram:
        return self._get(name, lambda: Histogram(name, help_text, label_names, buckets))

    def _get(self, name: str, factory):
        with self._lock:
            metric = self._metrics.get(name)
            if metric is None:
                metric = factory()
                self._metrics[name] = metric
            return metric

    def render(self) -> str:
        """Prometheus 文本格式（供 /metrics 直接返回）。"""
        with self._lock:
            metrics = [self._metrics[k] for k in sorted(self._metrics)]
        lines: list[str] = []
        for metric in metrics:
            lines.extend(metric.render())
        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        """清空（测试用；生产不需要）。"""
        with self._lock:
            self._metrics.clear()


# 全局注册表：`from metrics import REGISTRY`
REGISTRY = Registry()


# ── 本项目用到的指标（集中声明，避免名字/标签在各处漂移）────────
def http_requests() -> Counter:
    return REGISTRY.counter(
        "http_requests_total", "HTTP 请求数", ("method", "path", "status")
    )


def http_latency() -> Histogram:
    return REGISTRY.histogram(
        "http_request_duration_seconds", "HTTP 请求耗时", ("method", "path")
    )


def chat_outcomes() -> Counter:
    return REGISTRY.counter(
        "chat_requests_total",
        "问答请求数（outcome: ok/empty/fallback/handoff）",
        ("outcome",),
    )


def retrieval_results() -> Counter:
    return REGISTRY.counter(
        "retrieval_results_total",
        "检索结果数（outcome: hit/empty/degraded）",
        ("channel", "outcome"),
    )


def retrieval_latency() -> Histogram:
    return REGISTRY.histogram(
        "retrieval_duration_seconds", "检索耗时（含重排）", ("channel",)
    )


def generation_latency() -> Histogram:
    return REGISTRY.histogram("llm_generation_duration_seconds", "答案生成耗时")


def handoff_events() -> Counter:
    return REGISTRY.counter("handoff_events_total", "转人工信号触发次数")


def auth_events() -> Counter:
    return REGISTRY.counter("auth_events_total", "认证事件数", ("action", "result"))


def rate_limited() -> Counter:
    return REGISTRY.counter("rate_limited_total", "被限流拒绝的请求数", ("scope",))


def moderation_events() -> Counter:
    return REGISTRY.counter(
        "moderation_events_total", "内容审核命中数", ("stage", "result")
    )
