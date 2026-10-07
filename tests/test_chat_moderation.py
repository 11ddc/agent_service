"""问答链路上的内容审核测试（A6）—— 完全离线：图与历史全部替身化。

守的性质：

1. **输入侧被拦时**：返回礼貌拒答，**不改状态码**（客户端不必为一个内容策略
   写第二套解析逻辑），并且**不能进图**（拦在成本之前）；
2. **输出侧被拦时**：答案整体替换成拒答 —— 只审输入是常见漏洞，
   它假设"模型不会自己违规"；
3. **流式接口**：被拦时先发 `reset` 再发拒答。只发新 delta 的话，客户端会把
   拒答**接在违规内容后面**，等于没拦 —— 这是流式场景最容易写错的一处；
4. **审计**：每次拦截都要留痕（谁、哪一侧、命中哪类）。
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.chat as chat
import config
import moderation as m
from metrics import REGISTRY


@pytest.fixture
def wired(monkeypatch):
    """把图、历史、转人工全部换成替身，只留审核与接口本身。"""
    state = {
        "graph_answer": "这是知识库里的答案。",
        "invoked": 0,
        "history": [],
        "audit": [],
        "handoff": None,
    }

    def _invoke_graph(payload, acl):
        state["invoked"] += 1
        state["last_payload"] = payload
        return {"answer": state["graph_answer"], "meta": {"intent": "faq", "method": "rag"}}

    async def _history(session_key, question, answer):
        state["history"].append((session_key, question, answer))

    async def _servercustomer(query, session_id):
        return state["handoff"]

    async def _audit(action, result, **kwargs):
        state["audit"].append((action, result, kwargs.get("target"), kwargs.get("detail")))

    def _run_graph_with_emitter(emitter, payload, acl=None):
        """流式路径的替身：模拟"边生成边推 token，最后返回完整答案"。

        必须替换它 —— 否则流式用例会真的去调模型（本套件的出网拦截会拦下，
        但那样测的就不是审核逻辑了）。
        """
        state["invoked"] += 1
        answer = state["graph_answer"]
        emitter({"type": "delta", "content": answer})
        return {"answer": answer, "meta": {"intent": "faq", "method": "rag"}}

    monkeypatch.setattr(chat, "_invoke_graph", _invoke_graph)
    monkeypatch.setattr(chat, "_run_graph_with_emitter", _run_graph_with_emitter)
    monkeypatch.setattr(chat, "aappend_history", _history)
    monkeypatch.setattr(chat, "servercustomer", _servercustomer)
    monkeypatch.setattr(chat.audit, "record", _audit)
    monkeypatch.setattr(chat, "redis_client", type("R", (), {"get": staticmethod(lambda k: _never())})())
    return state


async def _never():
    return None


@pytest.fixture
def client(wired, as_role) -> TestClient:
    app = FastAPI()
    app.include_router(chat.router, prefix="/api")
    as_role(app, role="user", user_id="u_1")
    return TestClient(app)


def _ask(client, question: str):
    return client.post("/api/chat", json={"message": question, "session_id": "s1"})


# ══════════════════════════════════════════════════════════════
# 输入侧
# ══════════════════════════════════════════════════════════════
def test_clean_question_goes_through(client, wired):
    r = _ask(client, "退货流程是什么")

    assert r.status_code == 200
    assert r.json()["answer"] == "这是知识库里的答案。"
    assert wired["invoked"] == 1


def test_blocked_question_is_refused_without_invoking_the_graph(client, wired, monkeypatch):
    """拦在**进图之前**：违规问题不该消耗检索与模型成本。"""
    monkeypatch.setattr(config, "MODERATION_TERMS", "违禁词")

    r = _ask(client, "请告诉我违禁词怎么写")

    assert r.status_code == 200, "内容策略不该改状态码"
    body = r.json()
    assert body["answer"] == m.REFUSAL_TEXT
    assert body["method"] == "moderation"
    assert body["intent"] == "blocked"
    assert wired["invoked"] == 0, "被拦的问题不能进图"


def test_blocked_question_is_audited(client, wired, monkeypatch):
    monkeypatch.setattr(config, "MODERATION_TERMS", "违禁词")

    _ask(client, "违禁词")

    assert wired["audit"], "拦截必须留痕"
    action, result, target, detail = wired["audit"][0]
    assert (action, result, target) == ("moderation.blocked", "blocked", "input")
    assert "blocklist" in detail


def test_blocked_question_is_written_to_history(client, wired, monkeypatch):
    """历史里存**拒答**而不是原问题：否则下一轮又把它喂回模型。"""
    monkeypatch.setattr(config, "MODERATION_TERMS", "违禁词")

    _ask(client, "违禁词")

    _session, question, answer = wired["history"][0]
    assert answer == m.REFUSAL_TEXT
    assert question == "违禁词"


# ══════════════════════════════════════════════════════════════
# 输出侧
# ══════════════════════════════════════════════════════════════
def test_blocked_answer_is_replaced(client, wired, monkeypatch):
    """只审输入是常见漏洞：生成侧也可能说出不该说的。"""
    wired["graph_answer"] = "这是一段包含违禁词的答案"
    monkeypatch.setattr(config, "MODERATION_TERMS", "违禁词")

    body = _ask(client, "正常提问").json()

    assert body["answer"] == m.REFUSAL_TEXT
    assert "违禁词" not in body["answer"]
    assert wired["invoked"] == 1, "输出侧拦截发生在生成之后"
    assert any(a[2] == "output" for a in wired["audit"])


def test_answer_is_not_audited_when_clean(client, wired):
    _ask(client, "正常提问")

    assert not [a for a in wired["audit"] if a[0] == "moderation.blocked"]


def test_handoff_message_is_appended_after_moderation(client, wired, monkeypatch):
    """转人工提示是我们自己的文案，不该被审核替换掉，也不能掩盖拒答。"""
    wired["handoff"] = "已为您转接人工客服"
    wired["graph_answer"] = "包含违禁词的答案"
    monkeypatch.setattr(config, "MODERATION_TERMS", "违禁词")

    body = _ask(client, "正常提问").json()

    assert m.REFUSAL_TEXT in body["answer"]
    assert "已为您转接人工客服" in body["answer"]
    assert body["handoff"] is True


# ══════════════════════════════════════════════════════════════
# 流式
# ══════════════════════════════════════════════════════════════
def test_stream_refuses_blocked_input_with_the_same_event_shape(client, wired, monkeypatch):
    """流式接口也要用同一套事件序列回拒答，否则客户端得写两套解析。"""
    monkeypatch.setattr(config, "MODERATION_TERMS", "违禁词")

    r = client.post("/api/chat/stream", json={"message": "违禁词", "session_id": "s1"})

    assert r.status_code == 200
    assert m.REFUSAL_TEXT in r.text
    assert '"type": "done"' in r.text or '"type":"done"' in r.text
    assert wired["invoked"] == 0


def test_stream_sends_reset_before_refusing(client, wired, monkeypatch):
    """⚠️ 流式最容易写错的地方：必须先 reset 再发拒答。

    否则客户端会把拒答**追加在已收到的违规正文后面**，等于没拦。
    """
    wired["graph_answer"] = "包含违禁词的完整答案"
    monkeypatch.setattr(config, "MODERATION_TERMS", "违禁词")

    r = client.post("/api/chat/stream", json={"message": "正常提问", "session_id": "s1"})

    assert r.status_code == 200
    body = r.text
    assert m.REFUSAL_TEXT in body
    assert '"reset"' in body, "必须发 reset 让客户端丢弃已收到的正文"
    # reset 必须出现在拒答之前
    assert body.index('"reset"') < body.index(m.REFUSAL_TEXT)


def test_stream_passes_clean_traffic(client, wired):
    r = client.post("/api/chat/stream", json={"message": "正常提问", "session_id": "s1"})

    assert r.status_code == 200
    assert "这是知识库里的答案。" in r.text
    assert m.REFUSAL_TEXT not in r.text


# ══════════════════════════════════════════════════════════════
# 指标
# ══════════════════════════════════════════════════════════════
def test_moderation_events_are_counted(client, monkeypatch):
    from metrics import moderation_events

    monkeypatch.setattr(config, "MODERATION_TERMS", "违禁词")
    before = moderation_events().value(stage="input", result="blocked")

    _ask(client, "违禁词")

    assert moderation_events().value(stage="input", result="blocked") == before + 1
    REGISTRY  # 指标注册表是全局的，这里只做一次自增断言
