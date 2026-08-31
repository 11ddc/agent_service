import os
import traceback

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

RAG_PROMPT = """你是一个严谨的知识助手。请严格基于以下【参考资料】回答用户问题，不要编造资料中不存在的信息。

【参考资料】
{context}

【回答要求】
1. 只用参考资料里的内容作答，不得引入外部知识
2. 在每个关键论断后用 [doc{{i}}] 标注来源（i 对应上面文档编号）
3. 如果资料不足以回答，直接说"根据现有资料无法回答"，不要猜测
4. 答案结构清晰，分点叙述

【用户问题】
{question}
"""


def _format_docs(reranked_docs: list) -> str:
    """把重排后的文档拼成带 [doc{i}] 编号的上下文文本，供提示词里的引用要求对应上。

    兼容 Document 对象（取 page_content）和普通字符串。
    """
    parts = []
    for i, doc in enumerate(reranked_docs, 1):
        content = doc.page_content if hasattr(doc, "page_content") else str(doc)
        parts.append(f"[doc{i}] {content}")
    return "\n\n".join(parts)


class RAGGenerator:
    def __init__(self):
        # 创建生成模型客户端
        self.generate_client = OpenAI(
            api_key=os.getenv("GENERATE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def generate_answer(self, query: str, reranked_docs: list) -> str:
        """
        基于重排序后的文档，生成最终回答
        :param query: 用户原始问题
        :param reranked_docs: 重排序后的文档列表
        """
        print("问题：query：", query)
        # 先拼成带编号的纯文本，直接把 Document 列表塞进 format 会得到一堆
        # Document(page_content=..., metadata=...) 的对象字符串，污染上下文
        context_text = _format_docs(reranked_docs)
        try:
            res = self.generate_client.chat.completions.create(
                model="qwen3-32b",
                messages=[
                    {"role": "system", "content": "你是严谨的中文知识助手。"},
                    {
                        "role": "user",
                        "content": RAG_PROMPT.format(
                            context=context_text, question=query
                        ),
                    },
                ],
                extra_body={"enable_thinking": False},
                stream=False,
                # 温度越高llm越自由发挥
                temperature=0.3,
                max_tokens=2000,
            )
        except Exception as e:
            # 打印完整异常类型和堆栈（之前 KeyError 只打印出 'i' 就是吃了这个亏），
            # 然后抛给上层：graph.py 的 rag_node 会接住并走降级文案
            print(f"生成发生问题：{e!r}")
            traceback.print_exc()
            raise

        print("经过生成之后的文档：", res)

        return res.choices[0].message.content
