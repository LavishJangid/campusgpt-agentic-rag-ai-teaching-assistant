"""Tests for Redis-backed conversation memory."""

from backend.memory import ConversationMemory


def test_memory_add_and_retrieve():
    mem = ConversationMemory(max_messages=10)
    mem.add_exchange("session-abc", "Hello", "Hi there!")
    history = mem.get_history("session-abc")
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_memory_clear_session():
    mem = ConversationMemory(max_messages=10)
    mem.add_message("s1", "user", "test")
    mem.clear_session("s1")
    assert mem.get_history("s1") == []


def test_memory_export():
    mem = ConversationMemory(max_messages=10)
    mem.add_exchange("export-session", "Q?", "A!")
    export = mem.export_chat("export-session")
    assert "export-session" in export
    assert "[USER]" in export
    assert "[ASSISTANT]" in export
