"""
意图识别 —— 数据结构定义
"""

from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field


class IntentName(str, Enum):
    """意图枚举。

    注意：AMBIGUOUS 是管道内部的兜底标记，LLM 结构化输出时不允许返回它
    （提示词里只列 5 个合法意图）。
    """

    KB_QUESTION = "kb_question"  # 知识库问答 → 走 RAG
    CHITCHAT = "chitchat"  # 寒暄闲聊 → 直接回复
    STATUS_QUERY = "status_query"  # 知识库/系统状态 → 直接回复
    HUMAN_HANDOFF = "human_handoff"  # 转人工/投诉 → 直接回复
    # 工具调用（查订单/查物流/查发货等 → 对接订单系统/外部 API）。
    # 注意：该意图目前只靠 A 级规则命中（见 examples.py RULE_PATTERNS），
    # 不参与 B/C 级（INTENT_EXAMPLES 无示例、LLM 仲裁提示词也不含它），
    # 所以规则覆盖不到的说法会自然落回 kb_question 等其他意图。
    TOOL_CALL = "tool_call"  # 工具调用 → 后续接外部系统/API
    OUT_OF_SCOPE = "out_of_scope"  # 超出服务范围 → 直接回复
    
    # 意图模糊直接走llm
    AMBIGUOUS = "ambiguous"  # 意图模糊。（如果是空问题则置信度为0 ，另一种情况llm都拿不准（置信度低于写死的置信度））


class IntentReason(str, Enum):
    """兜底原因：区分"问题为空"与"分类器自身不可用"。

    历史坑：以前两种情况都用 confidence == 0 表示，于是 LLM 仲裁一失败
    （异常分支同样把 confidence 置 0），真实问题会被回一句
    "您好，我没收到您的问题"（见 agent/graph.py 的 route_by_intent）。
    """

    EMPTY = "empty"  # 空问题 / 无有效字符
    LOW_CONFIDENCE = "low_confidence"  # LLM 自己也不确定
    LLM_ERROR = "llm_error"  # LLM 仲裁调用失败（异常/超时）
    EMBEDDING_ERROR = "embedding_error"  # embedding 分层不可用（已降级到 LLM）


class IntentSlots(BaseModel):
    """槽位：从 query 中抽取的结构化信息（LLM 仲裁时顺带抽取，可选）"""

    source: str | None = None  # 文件名，如 "客服手册.pdf"
    keyword: str | None = None  # 检索关键词
    time_range: str | None = None  # 时间范围，如 "最近一周"


class IntentOutput(BaseModel):
    """LLM 结构化输出 schema（三级漏斗 C 级）"""

    intent: IntentName = Field(..., description="用户问题的意图")
    confidence: float = Field(..., ge=0, le=1, description="0~1 置信度")
    slots: IntentSlots = Field(default_factory=IntentSlots, description="抽取的槽位")


@dataclass
class IntentResult:
    """三级漏斗的最终判定结果"""

    intent: IntentName
    confidence: float = 1.0
    method: str = "rule"  # rule / embedding / llm / error
    slots: IntentSlots = field(default_factory=IntentSlots)
    scores: dict = field(
        default_factory=dict
    )  # 仅调试用：embedding 阶段各意图最高相似度
    # 兜底原因（None = 正常判定）。路由只看 reason，不再看 confidence == 0
    reason: IntentReason | None = None
