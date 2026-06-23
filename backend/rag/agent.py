"""LangGraph agent orchestrating Planner → Retriever → Memory → Generator."""

from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph
from loguru import logger

from backend.rag.citations import format_sources_list
from backend.rag.prompts import CHAT_PROMPT_TEMPLATE, SYSTEM_PROMPT


class AgentState(TypedDict):
    question: str
    chat_history: list
    subject: str
    semester: str
    unit: str
    topic: str
    document_type: str
    course: str
    plan: str
    retrieved_docs: list
    context: str
    answer: str
    sources: list
    follow_up_questions: list
    confidence_score: float


class LangGraphAgent:
    """Multi-step RAG agent using LangGraph."""

    def __init__(self, rag_engine):
        self.engine = rag_engine
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("planner", self._planner)
        workflow.add_node("retriever", self._retriever)
        workflow.add_node("memory", self._memory)
        workflow.add_node("generator", self._generator)
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "retriever")
        workflow.add_edge("retriever", "memory")
        workflow.add_edge("memory", "generator")
        workflow.add_edge("generator", END)
        return workflow.compile()

    def run(self, **kwargs) -> dict:
        initial: AgentState = {
            "question": kwargs.get("question", ""),
            "chat_history": kwargs.get("chat_history") or [],
            "subject": kwargs.get("subject", ""),
            "semester": kwargs.get("semester", ""),
            "unit": kwargs.get("unit", ""),
            "topic": kwargs.get("topic", ""),
            "document_type": kwargs.get("document_type", ""),
            "course": kwargs.get("course", ""),
            "plan": "",
            "retrieved_docs": [],
            "context": "",
            "answer": "",
            "sources": [],
            "follow_up_questions": [],
            "confidence_score": 0.0,
        }
        result = self.graph.invoke(initial)
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "confidence_score": result["confidence_score"],
            "follow_up_questions": result.get("follow_up_questions", []),
            "context_used": len(result["retrieved_docs"]) > 0,
            "num_sources": len(result["sources"]),
            "plan": result.get("plan", ""),
        }

    def _planner(self, state: AgentState) -> AgentState:
        q = state["question"]
        filters = []
        if state["subject"]:
            filters.append(f"subject={state['subject']}")
        if state["topic"]:
            filters.append(f"topic={state['topic']}")
        plan = (
            f"1. Search knowledge base for: '{q[:120]}'\n"
            f"2. Apply filters: {', '.join(filters) or 'none'}\n"
            f"3. Rerank top chunks and generate grounded answer"
        )
        state["plan"] = plan
        logger.info(f"Agent plan: {plan[:80]}...")
        return state

    def _retriever(self, state: AgentState) -> AgentState:
        docs = self.engine._retrieve_with_rerank(
            query=state["question"],
            subject=state["subject"],
            semester=state["semester"],
            unit=state["unit"],
            topic=state["topic"],
            document_type=state["document_type"],
            course=state["course"],
        )
        state["retrieved_docs"] = docs
        state["context"] = self.engine._build_context(docs)
        state["sources"] = format_sources_list(docs)
        state["confidence_score"] = self.engine._calculate_confidence(docs)
        return state

    def _memory(self, state: AgentState) -> AgentState:
        # Memory node enriches context with recent conversation
        history = state.get("chat_history") or []
        if history:
            recent = self.engine._format_chat_history(history[-6:])
            state["context"] = f"Recent conversation:\n{recent}\n\n---\n\n{state['context']}"
        return state

    def _generator(self, state: AgentState) -> AgentState:
        from langchain_core.messages import HumanMessage, SystemMessage

        prompt = CHAT_PROMPT_TEMPLATE.format(
            context=state["context"],
            chat_history=self.engine._format_chat_history(state["chat_history"]),
            question=state["question"],
        )
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        response = self.engine.llm.invoke(messages)
        state["answer"] = response.content
        state["follow_up_questions"] = self.engine._generate_follow_ups(
            state["question"], state["answer"]
        )
        return state
