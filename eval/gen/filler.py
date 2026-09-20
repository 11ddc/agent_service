"""语料填充层 —— 程序化生成"体量与干扰"，让语料达到企业级规模。

为什么不把手写事实堆到 300k 字符：手写的价值在**ground truth 的精确性**，
体量的价值在**让检索变难**。两者分开：
  - 事实句（eval/gen/gen_corpus.py 里的 F_* 常量）手写 → 锚点必然精确；
  - 本文的程序化内容只负责"像真实企业文档那样又长又杂"：
    步骤、故障码、参数详解、工单样例、会议纪要、问答、术语、变更记录……

关键约束（必须遵守，否则 golden 会被污染）：
  1. **不重复任何 F_* 事实句**，也不给出与 golden 相同口径的答案；
  2. 只给"近义干扰"——同结构、同词汇、**不同值**（例如 L2 电池 4100mAh 是干扰，
     L1 Pro 电池 8000mAh 是 golden）；
  3. 绝不写入负样本问题的答案：不出现 L3 型号、不写上市年份、不给门锁固件报价。

全部函数都由传入的 random.Random 驱动 → 同一 seed 完全可复现。
"""

from __future__ import annotations

import random

MODELS = ["L1", "L1 Pro", "L2", "L2 Pro"]
DISTRACTOR = ["S3", "S3 Pro"]
SPARES = ["标准锁体", "天地钩锁体", "指纹模块", "网关模块", "电池", "面板", "排线", "锁芯"]
CITIES = ["杭州", "南京", "苏州", "宁波", "合肥", "无锡", "常州", "绍兴"]
FAULT_SYMPTOMS = [
    "指纹识别失败", "密码面板无响应", "低电量告警频繁", "联网中断", "机械卡滞",
    "门磁误报", "蓝牙配对失败", "人脸识别超时", "应急供电无效",
]
CAUSES = [
    "手指表面潮湿或有油污", "面板表面残留清洁剂", "电池老化容量衰减",
    "网关配对码被修改", "锁体安装孔距不符", "门体形变导致锁舌受阻",
    "固件版本过低", "环境温度超出工作范围", "蓝牙被其他设备占用",
]
ACTIONS = [
    "清洁感应区后重试", "更换电池并重新校准", "在 App 内重新配对网关",
    "调整锁体位置并复测", "升级到最新固件", "联系上门服务复检",
    "执行恢复出厂设置", "更换指纹模块",
]
STAGES = ["待受理", "已派单", "上门中", "维修中", "待客户确认", "已关闭", "已升级二级"]
ENGINEERS = ["王工", "李工", "张工", "陈工", "刘工", "赵工", "周工", "吴工"]
CHANNELS = ["App 在线客服", "400 电话", "微信小程序", "门店报修", "电商平台工单"]


def _pick(rng: random.Random, seq: list[str]) -> str:
    return seq[rng.randrange(len(seq))]


def _date(rng: random.Random, year: int = 2025) -> str:
    return f"{year}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"


def install_steps(rng: random.Random, n: int) -> list[str]:
    """安装步骤段落：带编号的连续叙述。"""
    out = []
    for i in range(1, n + 1):
        model = _pick(rng, MODELS)
        city = _pick(rng, CITIES)
        out.append(
            f"步骤 {i}：在{city}的{model}安装作业中，需先确认门体开孔尺寸与锁体规格匹配，"
            f"再用扭矩扳手按 1.2N·m 固定面板；若门体存在形变，应先加装垫片校正，"
            f"否则后续{_pick(rng, FAULT_SYMPTOMS)}的概率会明显上升。"
        )
    return out


def troubleshooting(rng: random.Random, n: int) -> list[list[str]]:
    """故障排查表（表格行）。"""
    rows = [["故障码", "现象", "可能原因", "处理建议", "严重级别"]]
    for i in range(1, n + 1):
        rows.append(
            [
                f"E{1000 + i}",
                _pick(rng, FAULT_SYMPTOMS),
                _pick(rng, CAUSES),
                _pick(rng, ACTIONS),
                _pick(rng, ["低", "中", "高"]),
            ]
        )
    return rows


def param_prose(rng: random.Random, n: int) -> list[str]:
    """参数详解段落：与参数表同结构、不同措辞（近义干扰的主要来源）。"""
    out = []
    for i in range(n):
        model = _pick(rng, MODELS)
        battery = rng.choice([4100, 5000, 8000, 10000])
        weight = round(rng.uniform(3.0, 4.3), 2)
        out.append(
            f"{model} 的电池标称容量为 {battery}mAh，整机重量约 {weight}kg，"
            f"在 -10~55℃ 环境下可稳定工作；实测待机时间随开门频次变化，"
            f"建议每 {rng.choice([6, 9, 12])} 个月检查一次电量曲线。"
        )
    return out


def spare_rows(rng: random.Random, n: int) -> list[list[str]]:
    """备件库存表行（长表，制造"表格窗口"与行号映射压力）。"""
    rows = []
    for i in range(1, n + 1):
        part = f"{_pick(rng, SPARES)}-{i:04d}"
        rows.append(
            [
                part,
                _pick(rng, MODELS),
                str(rng.choice([19, 29, 39, 59, 89, 129, 199, 259, 349])),
                str(rng.randint(0, 400)),
                _pick(rng, ["合格", "待检", "返修"]),
                _date(rng),
            ]
        )
    return rows


def tickets(rng: random.Random, n: int) -> list[list[str]]:
    """历史工单样例行。"""
    rows = []
    for i in range(1, n + 1):
        rows.append(
            [
                f"WO-2025-{i:05d}",
                _pick(rng, MODELS),
                _pick(rng, CITIES),
                _pick(rng, FAULT_SYMPTOMS),
                _pick(rng, STAGES),
                _pick(rng, ENGINEERS),
                str(rng.randint(1, 72)),
                _pick(rng, CHANNELS),
            ]
        )
    return rows


def meeting_notes(rng: random.Random, n: int) -> list[str]:
    """会议纪要：带日期的决议叙述（会造成"过期信息"，是真实语料的典型噪声）。"""
    out = []
    for i in range(n):
        out.append(
            f"会议纪要 {_date(rng)}：与会人员讨论第 {i + 1} 项服务质量议题，"
            f"结论是{_pick(rng, ['加强首响考核', '补充备件安全库存', '优化派单半径', '延长回访周期'])}，"
            f"由{_pick(rng, ENGINEERS)}负责在两周内给出方案，下次例会复盘执行情况。"
        )
    return out


def faq_entries(rng: random.Random, n: int) -> list[tuple[str, str]]:
    """问答对：只问"操作类/流程类"，不回答 golden 事实（避免污染单 GT 判定）。"""
    questions = [
        "App 提示设备离线怎么办？",
        "如何添加第二枚指纹？",
        "门锁提示低电量还能用多久？",
        "怎么把管理员权限转给别人？",
        "更换手机后需要重新绑定吗？",
        "临时密码有效期可以设置多长？",
        "如何查看最近的开门记录？",
        "门锁被强制撬动会报警吗？",
        "支持接入哪些智能家居平台？",
        "忘记管理员密码怎么重置？",
    ]
    out = []
    for i in range(n):
        out.append(
            (
                f"{_pick(rng, questions)}（第 {i + 1} 则）",
                f"请先在 App 的「设备设置」中确认网络与电量状态，"
                f"再进行{_pick(rng, ['重新配网', '固件升级', '权限重置', '恢复出厂设置'])}；"
                f"若仍无法解决，请通过{_pick(rng, CHANNELS)}提交工单，"
                f"我们会在{_pick(rng, ['2 小时', '4 小时', '当日', '24 小时'])}内响应。",
            )
        )
    return out


def term_entries(rng: random.Random, n: int) -> list[str]:
    """术语条目列表。"""
    words = ["天地钩", "锁芯等级", "指纹误识率", "活体检测", "网关", "子设备", "防撬报警",
             "虚位密码", "常开模式", "一次性密码", "门磁", "离合结构", "把手方向", "猫眼联动"]
    out = []
    for i in range(n):
        out.append(
            f"- {_pick(rng, words)}（编号 T-{i + 1:03d}）：指"
            f"{_pick(rng, ['门锁结构中的加固部件', '安全等级的衡量指标', '与网关通信的子设备', '开锁方式的实现机制'])}，"
            f"相关参数请以产品参数表为准。"
        )
    return out


def changelog(rng: random.Random, n: int) -> list[str]:
    """变更记录条目（长列表 → 长文档切分压力）。"""
    out = []
    for i in range(1, n + 1):
        out.append(
            f"- {_date(rng)}：变更单 CR-{2000 + i}，涉及{_pick(rng, MODELS)}的"
            f"{_pick(rng, ['装配公差', '固件默认值', '包装清单', '出厂检测项', '供应商切换'])}，"
            f"影响范围已评估，产线于次周执行。"
        )
    return out


def maintenance(rng: random.Random, n: int) -> list[str]:
    """维护保养建议段落。"""
    out = []
    for i in range(n):
        out.append(
            f"保养建议 {i + 1}：每 {rng.choice([3, 6, 12])} 个月用干布擦拭"
            f"{_pick(rng, ['指纹感应区', '密码面板', '锁体导向片', '电池仓触点'])}，"
            f"避免使用含酒精的清洁剂；若长期不使用，建议取出电池并置于干燥环境。"
        )
    return out


def training_dialogues(rng: random.Random, n: int) -> list[str]:
    """客服培训对话片段（口语化 → 对 embedding 有干扰价值）。"""
    customer = [
        "我这个锁老是提示没电", "装完以后门有点关不严", "指纹老是识别不出来",
        "能不能给我换个新的", "你们上门要收多少钱", "我买的是不是正品",
    ]
    agent = [
        "非常理解您的心情，我先帮您确认一下设备状态。",
        "这个问题通常可以通过重新校准解决，我这边为您登记工单。",
        "为了尽快处理，麻烦您提供订单号后四位。",
        "我先把您的诉求记录下来，稍后由专员跟进。",
    ]
    out = []
    for i in range(n):
        out.append(
            f"客户：{_pick(rng, customer)}（第 {i + 1} 轮）\n客服：{_pick(rng, agent)}"
        )
    return out


def distractor_paragraphs(rng: random.Random, n: int) -> list[str]:
    """干扰域内容：云枢音箱线，词汇与门锁高度重叠但对象不同。"""
    out = []
    for i in range(n):
        model = _pick(rng, DISTRACTOR)
        out.append(
            f"{model} 的语音唤醒在嘈杂环境下需要{_pick(rng, ['提高唤醒词音量', '靠近设备 1 米内', '关闭背景音乐'])}，"
            f"蓝牙连接距离约 {rng.randint(5, 15)} 米，支持多房间同步播放；"
            f"若出现{_pick(rng, ['断连', '延迟高', '无声'])}，建议重启路由器后重试。"
        )
    return out
