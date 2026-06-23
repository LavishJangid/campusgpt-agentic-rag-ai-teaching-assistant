"""Tests for LangGraph agent (mocked)."""

from unittest.mock import MagicMock, patch


def test_langgraph_agent_run():
    import sys
    import types
    from unittest.mock import MagicMock

    # Mock langgraph if not installed
    if "langgraph.graph" not in sys.modules:
        lg_graph = types.ModuleType("langgraph.graph")
        lg_graph.END = "END"

        class _SG:
            def __init__(self, _):
                self._nodes = {}

            def add_node(self, name, fn):
                self._nodes[name] = fn

            def set_entry_point(self, _):
                pass

            def add_edge(self, *a):
                pass

            def compile(self):
                g = self

                class _Compiled:
                    def invoke(self, state):
                        for fn in g._nodes.values():
                            state = fn(state)
                        return state

                return _Compiled()

        lg_graph.StateGraph = _SG
        sys.modules["langgraph"] = types.ModuleType("langgraph")
        sys.modules["langgraph.graph"] = lg_graph

    mock_engine = MagicMock()
    mock_engine._retrieve_with_rerank.return_value = [
        {
            "source": "notes.pdf",
            "page_number": 5,
            "similarity_score": 0.9,
            "text": "Machine learning is a subset of AI.",
            "metadata": {"subject": "ML", "semester": "6"},
        }
    ]
    mock_engine._build_context.return_value = "ML context"
    mock_engine._format_chat_history.return_value = "No previous conversation."
    mock_engine._calculate_confidence.return_value = 0.85
    mock_engine._generate_follow_ups.return_value = ["What is deep learning?"]

    mock_response = MagicMock()
    mock_response.content = "Machine learning enables computers to learn from data."
    mock_engine.llm.invoke.return_value = mock_response

    from backend.rag.agent import LangGraphAgent

    agent = LangGraphAgent(mock_engine)
    result = agent.run(question="What is machine learning?", chat_history=[])

    assert "answer" in result
    assert result["context_used"] is True
    assert len(result["sources"]) == 1
