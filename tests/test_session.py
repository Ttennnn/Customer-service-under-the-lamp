from langchain_core.messages import AIMessage, HumanMessage

from app.services.session import SessionStore


def test_get_missing_returns_empty():
    store = SessionStore()
    assert store.get("nope") == []


def test_append_then_get_roundtrip():
    store = SessionStore()
    store.append("s1", HumanMessage(content="hi"))
    assert len(store.get("s1")) == 1


def test_messages_kept_in_append_order():
    store = SessionStore()
    store.append("s1", HumanMessage(content="first"))
    store.append("s1", AIMessage(content="second"))
    assert [m.content for m in store.get("s1")] == ["first", "second"]


def test_sessions_are_isolated():
    store = SessionStore()
    store.append("s1", HumanMessage(content="a"))
    store.append("s2", HumanMessage(content="b"))
    assert [m.content for m in store.get("s1")] == ["a"]
    assert [m.content for m in store.get("s2")] == ["b"]


def test_evicts_oldest_session_at_capacity():
    store = SessionStore(max_sessions=2)
    store.append("s1", HumanMessage(content="a"))
    store.append("s2", HumanMessage(content="b"))
    store.append("s3", HumanMessage(content="c"))
    assert store.get("s1") == []
    assert [m.content for m in store.get("s2")] == ["b"]
    assert [m.content for m in store.get("s3")] == ["c"]


def test_appending_to_existing_session_does_not_evict():
    store = SessionStore(max_sessions=2)
    store.append("s1", HumanMessage(content="a"))
    store.append("s2", HumanMessage(content="b"))
    store.append("s1", HumanMessage(content="more"))
    assert [m.content for m in store.get("s1")] == ["a", "more"]
    assert [m.content for m in store.get("s2")] == ["b"]
