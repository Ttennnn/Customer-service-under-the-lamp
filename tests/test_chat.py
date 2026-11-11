from app.services.chat import stream_chat
from app.services.session import SessionStore


class FakeChunk:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def stream(self, messages):
        yield FakeChunk("你")
        yield FakeChunk("好")


def test_stream_chat_yields_tokens_and_persists_history():
    store = SessionStore()
    tokens = list(stream_chat(FakeLLM(), store, "SYS", "s1", "在吗", max_tokens=100))
    assert tokens == ["你", "好"]
    history = store.get("s1")
    assert history[0].content == "在吗" and history[1].content == "你好"
