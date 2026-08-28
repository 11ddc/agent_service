from openai import OpenAI
import os
from dotenv import load_dotenv


load_dotenv()

#创建生成模型客户端
generate_client  = OpenAI(
      api_key=os.getenv("GENERATE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def generate_answer(queyr:str,reranked_docs: list)->str:
    """
    基于重排序后的文档，生成最终回答
    :param query: 用户原始问题
    :param reranked_docs: 重排序后的文档列表
    """
    print(f"问题：query：",queyr)
    print(f"重排序之后的内容：",reranked_docs)
    return True


