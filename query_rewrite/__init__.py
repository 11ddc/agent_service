"""
查询改写包 —— 在问题拆分之前做上下文补全（短期记忆：Redis 会话历史）。

对外 API：
    rewrite_query(question, history)     纯函数：改写核心（门控 + 低温 LLM）
    get_history(session_id) / append_history(session_id, question, answer)
                                         同步存取会话历史（图节点/非异步场景）
    aget_history / aappend_history       异步存取会话历史（api/chat.py 场景）
"""

# 整个query_rewrite 下的导出包

from query_rewrite.history import (
    aappend_history,
    aget_history,
    append_history,
    get_history,
)
from query_rewrite.rewriter import rewrite_query

__all__ = [
    "aappend_history",
    "aget_history",
    "append_history",
    "get_history",
    "rewrite_node",
    "rewrite_query",
]
