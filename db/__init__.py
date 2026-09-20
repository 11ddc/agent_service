"""MySQL 持久化层：父块存储 + 文档元数据。

对外只暴露这几样：
    parent_store                   父块读写（只按 id 取，不参与检索）
    DocumentStore / DocumentRow    文档元数据与重建一致性校验
    MySQLUnavailable               统一异常，调用方据此降级为纯子块检索
    mysql_cursor / ping            连接原语与自检

**降级约定**：本包的任何调用都可能抛 MySQLUnavailable。调用方必须捕获并
降级（父块缺失时用命中的子块顶替），绝不能让父块库的可用性决定能不能回答问题。
"""

from db import parent_store
from db.document_store import STATUS_FAILED, STATUS_OK, DocumentRow, DocumentStore
from db.mysql import MySQLUnavailable, connect, mysql_cursor, ping

__all__ = [
    "STATUS_FAILED",
    "STATUS_OK",
    "DocumentRow",
    "DocumentStore",
    "MySQLUnavailable",
    "connect",
    "mysql_cursor",
    "parent_store",
    "ping",
]
