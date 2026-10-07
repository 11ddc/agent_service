"""订单/物流/退款模拟数据层（`mcp_order_data.py`）测试 —— 零服务、零网络。

为什么这层单独存在、单独测：它是"模拟后端"，逻辑本身要经得起推敲 ——
如果模拟数据只会回一句"查询成功"，那它和之前的假工具没有区别，测不出任何东西。
这里守三件事：

1. **确定性**：同一输入永远同一结果。所有依赖"今天"的判断走 `today` 注入，
   不偷偷读系统时钟 —— 否则测试会随日期漂移。
2. **失败可分辨**：订单号格式非法 / 订单不存在 / 该单没有退款记录，是三种
   不同的情况，必须给出不同的话术。模型只有靠这个区别才能正确回复用户。
3. **能推出结论，而不只是回显字段**：运输超期、退款超过预计到账时间这类，
   要在文本里点出来 —— 客服工具的价值就在这里。
"""
from datetime import date
import json

import pytest

from mcp_order_data import (
    _MAX_RECENT_LIMIT,
    DEFAULT_CUSTOMER_ID,
    DEMO_ENV_VAR,
    PRINCIPAL_MAP_ENV_VAR,
    customer_id_for_phone,
    describe_logistics,
    describe_order,
    describe_recent_orders,
    describe_refund,
    normalize_order_no,
    resolve_principal,
)

TODAY = date(2025, 9, 20)


# ── 1. 订单号规范化与格式校验 ───────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("ORD-20250820-001", "ORD-20250820-001"),
        ("ord-20250820-001", "ORD-20250820-001"),
        ("  ORD-20250820-001  ", "ORD-20250820-001"),
    ],
)
def test_normalize_order_no(raw, expected):
    assert normalize_order_no(raw) == expected


@pytest.mark.parametrize("raw", ["12345", "ORD-2025-01", "订单号001", ""])
def test_malformed_order_no_is_reported_as_format_error(raw):
    out = describe_order(raw)

    assert "格式" in out
    assert "ORD-" in out  # 必须告诉用户正确格式长什么样


def test_unknown_order_no_says_not_found_not_format_error():
    """格式对但查不到，和格式错，是两回事 —— 话术必须不同。"""
    out = describe_order("ORD-20990101-999")

    assert "格式" not in out
    assert "查不到" in out or "没有找到" in out


# ── 2. 订单详情 ─────────────────────────────────────────────


def test_order_detail_contains_the_fields_customers_ask_about():
    out = describe_order("ord-20250820-001")

    assert "ORD-20250820-001" in out
    assert "已签收" in out  # 状态
    assert "2025-08-20" in out  # 下单时间


def test_order_detail_hides_full_phone_number():
    """工具结果是回灌给模型的，不该把完整手机号塞进上下文。"""
    out = describe_order("ORD-20250820-001")

    assert "13800001111" not in out
    assert "138****1111" in out


# ── 3. 物流进度 ─────────────────────────────────────────────


def test_logistics_for_shipped_order_has_carrier_and_tracking_no():
    out = describe_logistics("ORD-20250825-002")

    assert "承运" in out or "快递" in out
    assert "SF" in out  # 运单号前缀
    assert "已发货" in out or "在途" in out


def test_logistics_for_unpaid_order_says_not_shipped_yet():
    """待支付订单没有物流 —— 要说清原因，而不是回一句"没有物流信息"。"""
    out = describe_logistics("ORD-20250905-005")

    assert "未发货" in out or "还没发货" in out
    assert "待支付" in out


def test_logistics_flags_overdue_transit():
    """在途超过 5 天要点出来 —— 这正是用户打电话来问的原因。"""
    out = describe_logistics("ORD-20250825-002", today=date(2025, 9, 20))

    assert "超" in out
    assert "建议" in out


def test_logistics_does_not_flag_when_still_within_normal_transit():
    out = describe_logistics("ORD-20250825-002", today=date(2025, 8, 26))

    assert "超过" not in out


# ── 4. 退款进度 ─────────────────────────────────────────────


def test_refund_for_order_without_refund_says_no_record():
    out = describe_refund("ORD-20250820-001")

    assert "没有" in out and "退款" in out


def test_refund_in_progress_shows_expected_arrival_date():
    out = describe_refund("ORD-20250828-003", today=TODAY)

    assert "退款中" in out
    assert "预计" in out


def test_refund_flags_when_past_expected_arrival():
    """超过预计到账时间要主动指出来 —— 这条最能减少一次无意义的转人工。"""
    out = describe_refund("ORD-20250828-003", today=date(2025, 10, 20))

    assert "超过" in out


def test_refund_arrived_shows_arrival_date():
    out = describe_refund("ORD-20250901-004", today=TODAY)

    assert "已到账" in out
    assert "2025-09-1" in out or "2025-09-0" in out


def test_refund_rejected_explains_reason():
    out = describe_refund("ORD-20250910-007", today=TODAY)

    assert "驳回" in out
    assert len(out) > 20  # 不能只回一个状态词


# ── 5. 按调用方身份找订单（真实场景：用户不知道自己的订单号）──


def test_recent_orders_are_limited_to_the_newest_ones():
    out = describe_recent_orders(
        limit=2, customer_id=customer_id_for_phone("13800001111"), today=TODAY
    )

    # 该客户名下有 3 单，limit=2 必须给出**最新的两单**，且不泄露最旧那单
    assert "ORD-20250828-003" in out
    assert "ORD-20250825-002" in out
    assert "ORD-20250820-001" not in out


def test_recent_orders_fails_closed_when_demo_mode_is_off(monkeypatch):
    """关掉单租户演示模式后，**认不出的调用方必须失败关闭**。

    数据层在 customer_id=None（未绑定）时是宽松的，那是给单测直调的路径；
    身份一旦给出却认不出来，就绝不能退回"不过滤"。
    """
    monkeypatch.setenv(DEMO_ENV_VAR, "0")

    out = describe_recent_orders(customer_id="C-不存在", today=TODAY)

    assert "身份" in out or "转人工" in out
    assert "ORD-" not in out


def test_demo_mode_maps_every_principal_to_one_customer(monkeypatch):
    """没有认证层时，演示模式把所有 principal 映射到**同一个**客户。

    关键性质：调用方**不能**通过"把自己取名叫某个客户号"来选中客户 ——
    映射只认服务端来源（演示模式 / 显式映射表），不接受"看起来像就采信"。
    """
    monkeypatch.delenv(DEMO_ENV_VAR, raising=False)
    monkeypatch.delenv(PRINCIPAL_MAP_ENV_VAR, raising=False)

    assert resolve_principal("session-aaa") == DEFAULT_CUSTOMER_ID
    assert resolve_principal("session-bbb") == DEFAULT_CUSTOMER_ID
    # ⚠️ 回归点：直接报一个客户号不该被采信（上一版正是这样留下越权的）
    assert resolve_principal("C3") == DEFAULT_CUSTOMER_ID


def test_principal_map_maps_only_configured_server_side_principals(monkeypatch):
    """关掉演示模式后，只有**服务端显式配置过的** principal 才能拿到客户号。"""
    other = customer_id_for_phone("13900002222")
    monkeypatch.setenv(DEMO_ENV_VAR, "0")
    monkeypatch.setenv(PRINCIPAL_MAP_ENV_VAR, json.dumps({"u_real": other}))

    assert resolve_principal("u_real") == other
    assert resolve_principal("u_unknown") is None, "没配置的 principal 必须失败关闭"
    assert resolve_principal(other) is None, "配置表里没写的客户号同样不采信"
    assert resolve_principal(None) is None
    assert resolve_principal("") is None


def test_orders_fail_closed_when_demo_mode_is_off(monkeypatch):
    """同一个开关对订单详情也必须生效，不能只作用于"最近订单"。"""
    monkeypatch.setenv(DEMO_ENV_VAR, "0")

    out = describe_order("ORD-20250820-001", customer_id="session-xyz")

    assert "身份" in out or "转人工" in out
    assert "已签收" not in out


def test_recent_orders_only_returns_the_callers_own_orders():
    """换一个客户身份，看不到别人的订单。"""
    out = describe_recent_orders(
        customer_id=customer_id_for_phone("13900002222"), today=TODAY
    )

    assert "ORD-20250820-001" not in out  # 那单属于 13800001111


def test_recent_orders_never_echoes_a_phone_number():
    out = describe_recent_orders(
        customer_id=customer_id_for_phone("13800001111"), today=TODAY
    )

    assert "13800001111" not in out


def test_recent_orders_caps_model_supplied_limit():
    """`limit` 是模型能填的数，必须有上限，否则能把整张表灌进上下文。"""
    out = describe_recent_orders(
        limit=10**6, customer_id=customer_id_for_phone("13800001111"), today=TODAY
    )

    assert out.count("ORD-") <= _MAX_RECENT_LIMIT


# ── 6. 确定性 ───────────────────────────────────────────────


def test_same_input_gives_identical_output():
    """没有随机、没有隐式读时钟 —— 否则工具输出不可复现，测试也会漂。"""
    a = describe_order("ORD-20250820-001")
    b = describe_order("ORD-20250820-001")

    assert a == b


def test_clock_dependent_tools_accept_injected_today():
    """依赖"今天"的两个工具必须能被注入日期，否则测试会随运行日期漂移。"""
    fixed = date(2025, 9, 20)

    assert describe_logistics("ORD-20250825-002", today=fixed) == describe_logistics(
        "ORD-20250825-002", today=fixed
    )
    assert describe_refund("ORD-20250828-003", today=fixed) == describe_refund(
        "ORD-20250828-003", today=fixed
    )


def test_refund_without_injected_today_still_works():
    """生产路径不传 today 时必须能跑（内部回落到系统日期），不能 TypeError。"""
    out = describe_refund("ORD-20250828-003")

    assert "退款中" in out
