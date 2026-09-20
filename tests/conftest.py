"""
共享测试配置。

作用:
1. 把项目根目录加进 sys.path,保证 `import agent.graph` 这类顶层模块导入可用;
2. 在导入任何项目模块之前注入"占位 API key"——项目里多个模块在 import 时就会
   构造 OpenAI/智谱/DeepSeek 客户端对象,缺 key 会直接抛 Missing credentials。
   占位 key 只过构造校验,不会真的发请求;测试里所有 LLM 调用都走 mock/桩,
   因此整套默认测试 = 零服务、零密钥、零网络。
"""
import os
import sys
from pathlib import Path

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
