"""重排序冒烟脚本：真调一次智谱 glm-4.5-air，确认 key / 模型名 / 解析链都是通的。

    python -m rag._smoke_glm_rerank

正常输出：两条"退款"相关的块拿高分、闲聊块拿 0 分，最后打印 OK。
（线上不依赖这个脚本，它只是换 key / 换模型后的一条自检命令。）
"""

from langchain_core.documents import Document

from rag.glm_reranker import GLMReranker

QUERY = "怎么申请退款？"
DOCS = [
    Document(
        page_content="退款流程：进入订单详情页，点击申请退款，填写原因后提交，1~3 个工作日到账。",
        metadata={"source": "退款.md"},
    ),
    Document(page_content="今天天气不错，适合出门散步。", metadata={"source": "闲聊.txt"}),
    Document(
        page_content="如何申请退款？请在订单页提交退款申请，审核通过后原路退回。",
        metadata={"source": "FAQ.md"},
    ),
]

if __name__ == "__main__":
    reranker = GLMReranker()
    key = reranker.api_key or ""
    print(f"model = {reranker.model_name} | base_url = {reranker.base_url} | key = {key[:8]}...")

    scored = reranker.rerank_scored(QUERY, DOCS)
    for doc, score in scored:
        print(f"  {score:>5}  {doc.metadata['source']}  {doc.page_content[:24]}")

    assert scored == sorted(scored, key=lambda x: x[1], reverse=True), "分数没有降序"
    assert scored[0][0].metadata["source"] != "闲聊.txt", "不相关块不该排第一"
    print("OK：智谱 glm-4.5-air 重排序链路正常")
