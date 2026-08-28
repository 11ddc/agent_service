"""
意图识别 —— 数据结构定义
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class IntentName(str, Enum):
    """意图枚举。

    注意：AMBIGUOUS 是管道内部的兜底标记，LLM 结构化输出时不允许返回它
    （提示词里只列 5 个合法意图）。
    """
    KB_QUESTION = "kb_question"        # 知识库问答 → 走 RAG 
    CHITCHAT = "chitchat"              # 寒暄闲聊 → 直接回复
    STATUS_QUERY = "status_query"      # 知识库/系统状态 → 直接回复
    HUMAN_HANDOFF = "human_handoff"    # 转人工/投诉 → 直接回复
    OUT_OF_SCOPE = "out_of_scope"      # 超出服务范围 → 直接回复
    #意图模糊直接走llm
    AMBIGUOUS = "ambiguous"            # 意图模糊。（如果是空问题则置信度为0 ，另一种情况llm都拿不准（置信度低于写死的置信度））
    


class IntentSlots(BaseModel):
    """槽位：从 query 中抽取的结构化信息（LLM 仲裁时顺带抽取，可选）"""
    source: Optional[str] = None       # 文件名，如 "客服手册.pdf"
    keyword: Optional[str] = None      # 检索关键词
    time_range: Optional[str] = None   # 时间范围，如 "最近一周"


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
    method: str = "rule"               # rule / embedding / llm
    slots: IntentSlots = field(default_factory=IntentSlots)
    scores: dict = field(default_factory=dict)   # 仅调试用：embedding 阶段各意图最高相似度
