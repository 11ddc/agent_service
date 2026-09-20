"""MySQL 连接层：从 .env 的 MYSQL_URL 读配置，按需建立短连接。

设计要点：
- **import 零开销**：配置在调用时解析。缺 .env / 缺驱动都不会让 import 失败，
  与 rag/vision_ocr.py 的设计一致。
- **短连接而非连接池**：写入只发生在上传文档时（低频），读取是一次 IN 查询。
  本地 MySQL 建连接 2~5ms，相对 embedding 与 rerank 推理可忽略，
  但省掉了连接池 / 超时 / 断线重连的全部复杂度。
- **快速失败**：connect_timeout=5，MySQL 挂掉时检索侧立刻降级，不拖住请求。
- **字符集**：连接、表、库三处都必须是 utf8mb4。父块正文含大量 OCR 文本，
  生僻字在 MySQL 的 3 字节 "utf8" 下会直接报 Incorrect string value。
"""

import os
from contextlib import contextmanager
from urllib.parse import parse_qs, unquote, urlparse

from dotenv import load_dotenv

# 连接超时（秒）：MySQL 不可用时快速失败，避免拖住检索请求
CONNECT_TIMEOUT = 5

# 默认字符集：必须 utf8mb4（4 字节），"utf8" 是 3 字节别名，存不下生僻字
DEFAULT_CHARSET = "utf8mb4"


class MySQLUnavailable(RuntimeError):
    """MySQL 不可用（未配置 / 驱动缺失 / 连不上 / 查询失败）。

    调用方（rag.py）捕获后降级为纯子块检索，保证"父块库挂了也能回答问题"。
    """


def _dsn() -> str:
    """读取 MYSQL_URL。load_dotenv 不覆盖已有环境变量，重复调用安全。"""
    load_dotenv(encoding="utf-8-sig")  # utf-8-sig: 兼容带 BOM 的 .env
    url = os.getenv("MYSQL_URL")
    if not url or not url.strip():
        raise MySQLUnavailable("未配置 MYSQL_URL（请在 .env 里设置）")
    return url.strip()


def _connect_kwargs(url: str) -> dict:
    """把 SQLAlchemy 风格的 DSN 解析成 pymysql.connect 的参数。

    容忍两种写法：
      mysql+pymysql://user:pass@host:3306/db?charset=utf8mb4
      mysql://user:pass@host:3306/db
    用户名/密码里的特殊字符必须已 URL 编码（@ → %40 等），这里 unquote 还原。
    """
    # 将开头换成 mysql://，让 urlparse 正确解析用户名密码
    parsed = urlparse(url.replace("mysql+pymysql://", "mysql://", 1))
    if not parsed.hostname:
        raise MySQLUnavailable(f"MYSQL_URL 格式不正确（解析不出 host）: {url!r}")

    query = parse_qs(parsed.query or "")
    charset = (query.get("charset") or [DEFAULT_CHARSET])[0]

    return {
        "host": parsed.hostname,
        "port": parsed.port or 3306,
        "user": unquote(parsed.username or ""),
        "password": unquote(parsed.password or ""),
        "database": (parsed.path or "").lstrip("/") or None,
        "charset": charset or DEFAULT_CHARSET,
        "connect_timeout": CONNECT_TIMEOUT,
        "autocommit": False,
    }


def connect():
    """建立一条新连接。失败统一抛 MySQLUnavailable。"""
    url = _dsn()
    # _connect_kwargs返回建立数据库需要的参数
    kwargs = _connect_kwargs(url)
    try:
        import pymysql
    except ImportError as e:  # pragma: no cover - 依赖缺失时的明确提示
        raise MySQLUnavailable("缺少 MySQL 驱动，请执行: pip install PyMySQL") from e

    try:
        # 建立mysql连接，返回连接对象
        return pymysql.connect(**kwargs)
    except Exception as e:
        raise MySQLUnavailable(
            f"连接 MySQL 失败（{kwargs['host']}:{kwargs['port']}/"
            f"{kwargs.get('database')}）: {type(e).__name__}: {e}"
        ) from e


@contextmanager
def mysql_cursor():
    """游标上下文：正常退出自动 commit，异常自动 rollback，最后必定关连接。

    用法：
        with mysql_cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
    """
    conn = connect()
    try:
        # 创建游标对象（mysql连接对象不执行sql，游标才执行）
        cur = conn.cursor()
        try:
            # 将游标对象给with块去执行（with块会自动管理资源的获取和释放）
            # 这里会交出cur
            yield cur
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass  # rollback 本身失败不掩盖原始异常
            raise
        finally:
            # 怎么样都关闭
            cur.close()
    except MySQLUnavailable:
        raise
    except Exception as e:
        # 把底层驱动异常（查询报错 / 连接中断）统一收敛成 MySQLUnavailable，
        # 调用方只需要捕获一种异常
        raise MySQLUnavailable(f"MySQL 操作失败: {type(e).__name__}: {e}") from e
    finally:
        try:
            conn.close()
        except Exception:
            pass


def ping() -> tuple[bool, str]:
    """连通性自检。返回 (是否可用, 说明信息)。用于启动检查和上传前预检。"""
    try:
        with mysql_cursor() as cur:
            cur.execute("SELECT VERSION()")
            row = cur.fetchone()
        return True, f"ok (MySQL {row[0] if row else '?'})"
    except MySQLUnavailable as e:
        return False, str(e)
    except Exception as e:  # pragma: no cover - 兜底
        return False, f"{type(e).__name__}: {e}"
