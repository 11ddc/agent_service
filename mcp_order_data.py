"""订单 / 物流 / 退款 —— **模拟数据层**（没有 mcp 依赖，纯函数，可直接单测）。

## 这是什么

MCP 订单服务的后端。它**不连任何真实系统**，数据是进程内写死的样本，
用来把"工具调用链路"跑通并可回归测试。

**必须说清楚：这是 mock，不是真实订单系统。** 它和之前那个埋在 Agent 里、
返回 `"搜索成功，您的订单为。。。。。。。。"` 的假工具的区别在于：

1. 它在一个**明确的服务边界**后面（独立 MCP server），调用方知道自己在接一个服务；
2. 它**覆盖了真实的失败模式**（格式错 / 查不到 / 未发货 / 无退款记录 / 已驳回），
   所以能真的检验模型会不会区别对待这些情况；
3. 它会**推出结论**而不只是回显字段（运输超期、退款超过预计到账），
   这才是客服工具的价值所在。

接真实系统时，只需把这里的 `_find` / 渲染换成真实 API 调用，上层一行不用改。

## 确定性

所有依赖"今天"的判断都走 `today` 参数注入，不读系统时钟 —— 否则同一问题
今天答"运输正常"、明天答"运输超期"，测试也会随日期漂移。
"""

import re
from dataclasses import dataclass
from datetime import date, timedelta

# 订单号格式：ORD-YYYYMMDD-NNN
ORDER_NO_RE = re.compile(r"^ORD-\d{8}-\d{3}$")

# 在途超过这么多天就认为异常，要主动提示用户
TRANSIT_ALERT_DAYS = 5


@dataclass(frozen=True)
class Refund:
    status: str  # 审核中 / 退款中 / 已到账 / 已驳回
    amount: float
    applied_on: date
    arrive_by: date | None  # 预计到账日；已驳回时为 None
    reason: str


@dataclass(frozen=True)
class Order:
    order_no: str
    phone: str
    status: str  # 待支付 / 已发货 / 已签收 / 已取消
    items: tuple[str, ...]
    amount: float
    created_on: date
    carrier: str | None = None
    tracking_no: str | None = None
    shipped_on: date | None = None
    delivered_on: date | None = None
    refund: Refund | None = None


# ── 模拟数据集 ──────────────────────────────────────────────
# 刻意覆盖客服真实会遇到的状态，而不是一个"一切正常"的样本：
#   已签收无退款 / 在途 / 退款中 / 已退款到账 / 待支付 / 退款被驳回
_ORDERS: dict[str, Order] = {
    "ORD-20250820-001": Order(
        order_no="ORD-20250820-001",
        phone="13800001111",
        status="已签收",
        items=("云枢S3 Pro 智能门锁", "安装服务"),
        amount=2699.00,
        created_on=date(2025, 8, 20),
        carrier="顺丰速运",
        tracking_no="SF1234567890",
        shipped_on=date(2025, 8, 21),
        delivered_on=date(2025, 8, 23),
    ),
    "ORD-20250825-002": Order(
        order_no="ORD-20250825-002",
        phone="13800001111",
        status="已发货",
        items=("岚盾L2 Pro 智能门锁",),
        amount=1899.00,
        created_on=date(2025, 8, 25),
        carrier="顺丰速运",
        tracking_no="SF9876543210",
        shipped_on=date(2025, 8, 25),
    ),
    "ORD-20250828-003": Order(
        order_no="ORD-20250828-003",
        phone="13800001111",
        status="已签收",
        items=("云枢S3 Plus 智能音箱",),
        amount=899.00,
        created_on=date(2025, 8, 28),
        carrier="中通快递",
        tracking_no="ZT5566778899",
        shipped_on=date(2025, 8, 29),
        delivered_on=date(2025, 8, 31),
        refund=Refund(
            status="退款中",
            amount=899.00,
            applied_on=date(2025, 9, 10),
            arrive_by=date(2025, 9, 25),
            reason="七天无理由退货，商品已寄回并验收通过",
        ),
    ),
    "ORD-20250901-004": Order(
        order_no="ORD-20250901-004",
        phone="13900002222",
        status="已签收",
        items=("云枢S3 Max 智能门锁",),
        amount=3299.00,
        created_on=date(2025, 9, 1),
        carrier="京东物流",
        tracking_no="JD1122334455",
        shipped_on=date(2025, 9, 2),
        delivered_on=date(2025, 9, 4),
        refund=Refund(
            status="已到账",
            amount=3299.00,
            applied_on=date(2025, 9, 5),
            arrive_by=date(2025, 9, 12),
            reason="商品到货破损，全额退款",
        ),
    ),
    "ORD-20250905-005": Order(
        order_no="ORD-20250905-005",
        phone="13900002222",
        status="待支付",
        items=("岚盾L1Pro 智能门锁",),
        amount=1299.00,
        created_on=date(2025, 9, 5),
    ),
    "ORD-20250910-007": Order(
        order_no="ORD-20250910-007",
        phone="13700003333",
        status="已签收",
        items=("云枢音箱 网关套装",),
        amount=599.00,
        created_on=date(2025, 9, 10),
        carrier="圆通速递",
        tracking_no="YT6677889900",
        shipped_on=date(2025, 9, 11),
        delivered_on=date(2025, 9, 13),
        refund=Refund(
            status="已驳回",
            amount=599.00,
            applied_on=date(2025, 9, 14),
            arrive_by=None,
            reason="超出七天无理由退货期限，且商品有使用痕迹",
        ),
    ),
}


def normalize_order_no(raw: object) -> str:
    """去掉空格、统一大写 —— 用户手输的订单号大小写和空格都不稳定。"""
    return str(raw or "").strip().upper()


def mask_phone(phone: str) -> str:
    """手机号打码。工具结果会回灌进模型上下文，不该出现完整手机号。"""
    p = str(phone or "")
    return f"{p[:3]}****{p[-4:]}" if len(p) >= 7 else "***"


def _find(raw: object) -> tuple[Order | None, str | None]:
    """返回 (订单, 错误话术)。格式错与查不到是**两种不同的**错误。"""
    no = normalize_order_no(raw)
    if not ORDER_NO_RE.match(no):
        return None, (
            f"订单号格式不正确：「{raw}」。正确格式是 ORD-YYYYMMDD-NNN，"
            f"例如 ORD-20250820-001。请让用户核对后重新提供。"
        )
    order = _ORDERS.get(no)
    if order is None:
        return None, (
            f"查不到订单号 {no}。请确认订单号是否输入有误；"
            f"若用户不记得订单号，可以改用下单手机号查询最近订单。"
        )
    return order, None


def _fmt(d: date | None) -> str:
    return d.isoformat() if d else "-"


def _today(today: date | None) -> date:
    """时钟注入点：调用方传就用传的，不传才读系统日期。

    测试**必须**注入固定日期 —— 否则"运输是否超期"这类判断会随运行日期漂移，
    同一份用例今天过、下个月挂。
    """
    return today or date.today()


# ── 订单详情 ────────────────────────────────────────────────


def describe_order(raw: object) -> str:
    """订单详情。（不需要时钟：这里没有依赖"今天"的判断）"""
    order, err = _find(raw)
    if order is None:
        return err

    lines = [
        f"订单 {order.order_no}",
        f"- 状态：{order.status}",
        f"- 下单时间：{_fmt(order.created_on)}",
        f"- 商品：{'、'.join(order.items)}",
        f"- 订单金额：¥{order.amount:.2f}",
        f"- 收货手机号：{mask_phone(order.phone)}",
    ]
    if order.tracking_no:
        lines.append(f"- 物流：{order.carrier} {order.tracking_no}")
    if order.delivered_on:
        lines.append(f"- 签收时间：{_fmt(order.delivered_on)}")
    if order.refund:
        lines.append(
            f"- 退款：{order.refund.status}（¥{order.refund.amount:.2f}），"
            f"详情可查退款进度"
        )
    return "\n".join(lines)


# ── 物流进度 ────────────────────────────────────────────────


def describe_logistics(raw: object, today: date | None = None) -> str:
    order, err = _find(raw)
    if order is None:
        return err

    if not order.tracking_no or not order.shipped_on:
        return (
            f"订单 {order.order_no} 当前状态是「{order.status}」，**未发货**，"
            f"因此暂无物流信息。付款后一般 24 小时内发出。"
        )

    lines = [
        f"订单 {order.order_no} 物流进度",
        f"- 订单状态：{order.status}",
        f"- 承运商：{order.carrier}",
        f"- 运单号：{order.tracking_no}",
        f"- 发货时间：{_fmt(order.shipped_on)}",
    ]

    if order.delivered_on:
        lines.append(f"- 签收时间：{_fmt(order.delivered_on)}")
        lines.append("轨迹：已揽收 → 运输中 → 派送中 → 已签收")
        return "\n".join(lines)

    transit = (_today(today) - order.shipped_on).days
    lines.append(f"- 当前状态：运输中（已发出 {transit} 天）")
    lines.append("轨迹：已揽收 → 运输中")
    if transit > TRANSIT_ALERT_DAYS:
        lines.append(
            f"⚠️ 在途已超过 {TRANSIT_ALERT_DAYS} 天（{transit} 天），可能存在异常。"
            f"建议：先按运单号向 {order.carrier} 核实，必要时转人工跟进。"
        )
    return "\n".join(lines)


# ── 退款进度 ────────────────────────────────────────────────


def describe_refund(raw: object, today: date | None = None) -> str:
    order, err = _find(raw)
    if order is None:
        return err

    refund = order.refund
    if refund is None:
        return (
            f"订单 {order.order_no} 没有退款记录（该订单当前状态：{order.status}）。"
            f"如果用户认为应当已退款，请先核对订单号，或转人工核实。"
        )

    lines = [
        f"订单 {order.order_no} 退款进度",
        f"- 退款状态：{refund.status}",
        f"- 退款金额：¥{refund.amount:.2f}",
        f"- 申请时间：{_fmt(refund.applied_on)}",
        f"- 申请原因：{refund.reason}",
    ]

    if refund.status == "已到账":
        lines.append(f"- 到账时间：{_fmt(refund.arrive_by)}")
    elif refund.arrive_by is not None:
        lines.append(f"- 预计到账：{_fmt(refund.arrive_by)} 前")
        if _today(today) > refund.arrive_by:
            overdue = (_today(today) - refund.arrive_by).days
            lines.append(
                f"⚠️ 已超过预计到账时间 {overdue} 天，建议转人工核实退款去向。"
            )
    return "\n".join(lines)


# ── 按手机号查最近订单（用户往往不记得订单号）──────────────


def describe_recent_orders(
    phone: object, limit: int = 3, today: date | None = None
) -> str:
    key = str(phone or "").strip()
    if not key:
        return "需要提供下单时使用的手机号，才能查询最近订单。"

    hits = sorted(
        (o for o in _ORDERS.values() if o.phone == key),
        key=lambda o: o.created_on,
        reverse=True,
    )[: max(1, int(limit))]

    if not hits:
        return f"查不到手机号 {mask_phone(key)} 名下的订单，请让用户核对手机号。"

    lines = [f"手机号 {mask_phone(key)} 名下最近的 {len(hits)} 笔订单："]
    lines += [
        f"- {o.order_no}｜{o.status}｜{_fmt(o.created_on)}｜¥{o.amount:.2f}"
        for o in hits
    ]
    lines.append(
        "（说明：真实系统需先做身份校验才能返回他人订单，此处为模拟数据。）"
    )
    return "\n".join(lines)
