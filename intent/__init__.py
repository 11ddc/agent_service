"""
意图识别模块 —— 三级漏斗：规则 → Embedding 分类 → LLM 仲裁。

用法：
    from intent import classify
    result = classify("退换货流程是什么")
    result.intent, result.confidence, result.method
"""
from intent.classifier import IntentClassifier, classify, get_classifier
from intent.schemas import IntentName, IntentOutput, IntentResult, IntentSlots

__all__ = [
    "IntentClassifier",
    "IntentName",
    "IntentOutput",
    "IntentResult",
    "IntentSlots",
    "classify",
    "get_classifier",
]
