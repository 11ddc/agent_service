"""
共享测试配置。

作用:
1. 把项目根目录加进 sys.path,保证 `import agent.graph` 这类顶层模块导入可用;
2. 在导入任何项目模块之前注入"占位 API key"——项目里多个模块在 import 时就会
   构造 OpenAI/智谱/DeepSeek 客户端对象,缺 key 会直接抛 Missing credentials。
   占位 key 只过构造校验,不会真的发请求;测试里所有 LLM 调用都走 mock/桩,
   因此整套默认测试 = 零服务、零密钥、零网络。
3. **把上面那句承诺从"约定"变成"机制"**:`_no_outbound_network` 会拦掉一切
   非回环的出网连接。见下面那个 fixture 的说明。
"""
import os
import socket
import sys
from pathlib import Path

import pytest

for _key in (
    "DEEPSEEK_API_KEY",
    "ZHI_PU_API_KEY",
    "QIANWEN_API_KEY",
    # 问题拆分模块在 import 时就构造客户端（缺 key 直接抛 Missing credentials）
    "QIAN_WEN_QUERYSTION_API_KEY",
    "GENERATE_API_KEY",
    "DASHSCOPE_API_KEY",
    "OPENAI_API_KEY",
):
    os.environ.setdefault(_key, "test-dummy-key")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# 只放行回环：Windows 上 asyncio 的内部自管道会用回环 socket，
# 挡掉它会把 `asyncio.run` 一起弄坏（而好几个模块的同步桥正靠它）。
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost", ""}


@pytest.fixture(autouse=True)
def _no_outbound_network(monkeypatch):
    """禁止任何真实出网连接 —— "零外部服务"必须是被**强制**的，而不是靠自觉。

    为什么需要这条机制：项目各处都写着"所有 LLM 调用都走 mock"，但那是**约定**。
    只要有人新写一个忘了 mock 的用例，它就会真的去调 DeepSeek / DashScope ——
    可能烧钱、可能在 CI 上"碰巧通过"、也可能因为网络抖动变成偶发失败。
    装上这道闸之后，"真实出网"直接等于测试失败，问题在第一次运行时就暴露。

    只拦"连出去"这一步（connect / connect_ex / create_connection）：
    DNS 解析不产生业务请求，挡掉反而会误伤。
    """
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_create_connection = socket.create_connection

    def _host_of(address):
        if isinstance(address, tuple) and address:
            return address[0]
        return str(address)

    def _deny(address):
        raise RuntimeError(
            f"测试禁止真实出网连接（{address!r}）。"
            f"请为这条路径补 mock，或改写成离线可验证的断言 —— "
            f"不要为了让它通过而放开这道闸。"
        )

    def _connect(self, address):
        if _host_of(address) not in _LOOPBACK_HOSTS:
            _deny(address)
        return real_connect(self, address)

    def _connect_ex(self, address):
        if _host_of(address) not in _LOOPBACK_HOSTS:
            _deny(address)
        return real_connect_ex(self, address)

    def _create_connection(address, *args, **kwargs):
        if _host_of(address) not in _LOOPBACK_HOSTS:
            _deny(address)
        return real_create_connection(address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", _connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _connect_ex)
    monkeypatch.setattr(socket, "create_connection", _create_connection)
