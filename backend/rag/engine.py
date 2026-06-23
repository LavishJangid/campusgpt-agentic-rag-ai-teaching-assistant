"""RAG Engine - Core retrieval-augmented generation pipeline."""

import time
from typing import Optional
from loguru import logger

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

from backend.config import get_settings
from backend.ingestion.embeddings import EmbeddingGenerator
from backend.vectorstore.chroma_store import ChromaStore
from backend.rag.prompts import (
    SYSTEM_PROMPT,
    CHAT_PROMPT_TEMPLATE,
    EXAM_PREP_PROMPT,
    QUIZ_GENERATOR_PROMPT,
    VIVA_QUESTIONS_PROMPT,
    ASSIGNMENT_HELPER_PROMPT,
    IMPORTANT_QUESTIONS_PROMPT,
)
from backend.rag.reranker import CrossEncoderReranker
from backend.rag.citations import format_sources_list


class RAGEngine:
    """
    RAG Engine orchestrating the full retrieval-augmented generation pipeline.

    Pipeline: Question → Embedding → Retriever → Top-K → Context → LLM → Answer
    """

    def __init__(self):
        self.settings = get_settings()

        # Initialize components
        self.embedding_generator = EmbeddingGenerator(
            model_name=self.settings.embedding_model
        )
        self.vector_store = ChromaStore(
            persist_dir=self.settings.chroma_persist_dir,
            collection_name=self.settings.chroma_collection_name,
            embedding_generator=self.embedding_generator,
        )

        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=self.settings.llm_model,
            google_api_key=self.settings.gemini_api_key,
            temperature=self.settings.llm_temperature,
            max_output_tokens=self.settings.llm_max_tokens,
            convert_system_message_to_human=True,
        )

        self.reranker = CrossEncoderReranker()
        self._agent = None

        # Metrics
        self.metrics = {
            "total_queries": 0,
            "total_response_time": 0,
            "avg_response_time": 0,
            "topics_searched": {},
        }

        logger.info("RAG Engine initialized")

    @property
    def agent(self):
        if self._agent is None:
            from backend.rag.agent import LangGraphAgent

            self._agent = LangGraphAgent(self)
        return self._agent

    def _retrieve_with_rerank(
        self,
        query: str,
        subject: str = "",
        semester: str = "",
        unit: str = "",
        topic: str = "",
        document_type: str = "",
        course: str = "",
        use_mmr: bool = False,
    ) -> list[dict]:
        """Embedding search → top N → CrossEncoder rerank → top K."""
        fetch_k = self.settings.retrieval_fetch_k
        top_k = self.settings.rerank_top_k

        if use_mmr:
            candidates = self.vector_store.mmr_search(query=query, top_k=fetch_k)
        else:
            candidates = self.vector_store.hybrid_search(
                query=query,
                top_k=fetch_k,
                subject=subject,
                semester=semester,
                unit=unit,
                topic=topic,
                document_type=document_type,
                course=course,
            )

        return self.reranker.rerank(query, candidates, top_k=top_k)

    def chat(
        self,
        question: str,
        chat_history: Optional[list[dict]] = None,
        subject: str = "",
        semester: str = "",
        unit: str = "",
        topic: str = "",
        document_type: str = "",
        course: str = "",
        top_k: int = 5,
        use_mmr: bool = False,
    ) -> dict:
        """
        Process a student question through the full RAG pipeline.

        Returns:
            dict with answer, sources, confidence, and metadata.
        """
        start_time = time.time()
        self.metrics["total_queries"] += 1

        try:
            if self.settings.use_langgraph_agent and not use_mmr:
                agent_result = self.agent.run(
                    question=question,
                    chat_history=chat_history,
                    subject=subject,
                    semester=semester,
                    unit=unit,
                    topic=topic,
                    document_type=document_type,
                    course=course,
                )
                elapsed = round(time.time() - start_time, 2)
                self.metrics["total_response_time"] += elapsed
                self.metrics["avg_response_time"] = round(
                    self.metrics["total_response_time"] / self.metrics["total_queries"], 2
                )
                topic_key = topic or subject or "general"
                self.metrics["topics_searched"][topic_key] = (
                    self.metrics["topics_searched"].get(topic_key, 0) + 1
                )
                return {
                    **agent_result,
                    "response_time": elapsed,
                }

            # Step 1: Retrieve with reranking
            logger.info(f"Retrieving context for: '{question[:80]}...'")

            retrieved_docs = self._retrieve_with_rerank(
                query=question,
                subject=subject,
                semester=semester,
                unit=unit,
                topic=topic,
                document_type=document_type,
                course=course,
                use_mmr=use_mmr,
            )

            # Step 2: Build context
            context = self._build_context(retrieved_docs)

            # Step 3: Format chat history
            history_str = self._format_chat_history(chat_history)

            # Step 4: Generate answer
            prompt = CHAT_PROMPT_TEMPLATE.format(
                context=context,
                chat_history=history_str,
                question=question,
            )

            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]

            response = self.llm.invoke(messages)
            answer = response.content

            # Step 5: Calculate confidence score
            confidence = self._calculate_confidence(retrieved_docs)

            # Step 6: Generate suggested follow-ups
            follow_ups = self._generate_follow_ups(question, answer)

            # Track metrics
            elapsed = round(time.time() - start_time, 2)
            self.metrics["total_response_time"] += elapsed
            self.metrics["avg_response_time"] = round(
                self.metrics["total_response_time"] / self.metrics["total_queries"], 2
            )

            # Track topics
            topic_key = topic or subject or "general"
            self.metrics["topics_searched"][topic_key] = (
                self.metrics["topics_searched"].get(topic_key, 0) + 1
            )

            result = {
                "answer": answer,
                "sources": format_sources_list(retrieved_docs),
                "confidence_score": confidence,
                "response_time": elapsed,
                "follow_up_questions": follow_ups,
                "context_used": len(retrieved_docs) > 0,
                "num_sources": len(retrieved_docs),
            }

            logger.info(
                f"Chat response generated | time={elapsed}s | "
                f"sources={len(retrieved_docs)} | confidence={confidence}"
            )
            return result

        except Exception as e:
            elapsed = round(time.time() - start_time, 2)
            logger.error(f"Chat failed: {str(e)}")
            return {
                "answer": f"I apologize, but I encountered an error processing your question. "
                          f"Please try again. Error: {str(e)}",
                "sources": [],
                "confidence_score": 0,
                "response_time": elapsed,
                "follow_up_questions": [],
                "context_used": False,
                "num_sources": 0,
                "error": str(e),
            }

    def exam_preparation(
        self, subject: str, topic: str = "", unit: str = ""
    ) -> dict:
        """Generate exam preparation material."""
        start_time = time.time()

        query = f"exam preparation {subject} {topic} {unit} important concepts formulas"
        docs = self.vector_store.hybrid_search(
            query=query, top_k=10, subject=subject, topic=topic, unit=unit
        )
        context = self._build_context(docs)

        prompt = EXAM_PREP_PROMPT.format(
            context=context, subject=subject, topic=topic or unit or "all topics"
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)

        return {
            "content": response.content,
            "subject": subject,
            "topic": topic,
            "sources": [d["source"] for d in docs],
            "response_time": round(time.time() - start_time, 2),
        }

    def generate_quiz(
        self,
        subject: str,
        topic: str = "",
        difficulty: str = "medium",
        num_questions: int = 10,
    ) -> dict:
        """Generate a quiz based on course material."""
        start_time = time.time()

        query = f"{subject} {topic} quiz questions concepts"
        docs = self.vector_store.hybrid_search(
            query=query, top_k=8, subject=subject, topic=topic
        )
        context = self._build_context(docs)

        prompt = QUIZ_GENERATOR_PROMPT.format(
            context=context,
            subject=subject,
            topic=topic or "general",
            difficulty=difficulty,
            num_questions=num_questions,
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)

        return {
            "quiz": response.content,
            "subject": subject,
            "topic": topic,
            "difficulty": difficulty,
            "num_questions": num_questions,
            "sources": [d["source"] for d in docs],
            "response_time": round(time.time() - start_time, 2),
        }

    def generate_viva_questions(
        self, subject: str, topic: str = "", num_questions: int = 15
    ) -> dict:
        """Generate viva voce questions."""
        start_time = time.time()

        query = f"{subject} {topic} concepts definitions important"
        docs = self.vector_store.hybrid_search(
            query=query, top_k=8, subject=subject, topic=topic
        )
        context = self._build_context(docs)

        prompt = VIVA_QUESTIONS_PROMPT.format(
            context=context,
            subject=subject,
            topic=topic or "general",
            num_questions=num_questions,
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)

        return {
            "questions": response.content,
            "subject": subject,
            "topic": topic,
            "num_questions": num_questions,
            "sources": [d["source"] for d in docs],
            "response_time": round(time.time() - start_time, 2),
        }

    def assignment_help(self, question: str, subject: str = "") -> dict:
        """Help with assignment questions."""
        start_time = time.time()

        docs = self.vector_store.hybrid_search(
            query=question, top_k=8, subject=subject
        )
        context = self._build_context(docs)

        prompt = ASSIGNMENT_HELPER_PROMPT.format(
            context=context, question=question
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)

        return {
            "answer": response.content,
            "question": question,
            "sources": [
                {"source": d["source"], "page": d["page_number"]}
                for d in docs
            ],
            "response_time": round(time.time() - start_time, 2),
        }

    def find_important_questions(
        self, subject: str, unit: str = ""
    ) -> dict:
        """Find important questions for a subject/unit."""
        start_time = time.time()

        query = f"{subject} {unit} important questions previous year exam"
        docs = self.vector_store.hybrid_search(
            query=query,
            top_k=10,
            subject=subject,
            unit=unit,
            document_type="question_paper",
        )

        # Also search notes
        note_docs = self.vector_store.hybrid_search(
            query=query, top_k=5, subject=subject, unit=unit
        )
        docs.extend(note_docs)

        context = self._build_context(docs)

        prompt = IMPORTANT_QUESTIONS_PROMPT.format(
            context=context, subject=subject, unit=unit or "all units"
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)

        return {
            "questions": response.content,
            "subject": subject,
            "unit": unit,
            "sources": list(set(d["source"] for d in docs)),
            "response_time": round(time.time() - start_time, 2),
        }

    def search_by_topic(self, topic: str, top_k: int = 10) -> list[dict]:
        """Search for content related to a specific topic."""
        results = self.vector_store.similarity_search(query=topic, top_k=top_k)
        return results

    def search_by_semester(self, semester: str, query: str = "", top_k: int = 10) -> list[dict]:
        """Search for content from a specific semester."""
        return self.vector_store.hybrid_search(
            query=query or f"semester {semester} topics", top_k=top_k, semester=semester
        )

    def search_by_subject(self, subject: str, query: str = "", top_k: int = 10) -> list[dict]:
        """Search for content from a specific subject."""
        return self.vector_store.hybrid_search(
            query=query or f"{subject} key concepts", top_k=top_k, subject=subject
        )

    def get_metrics(self) -> dict:
        """Return current engine metrics."""
        return {
            **self.metrics,
            "vector_store_stats": self.vector_store.get_collection_stats(),
        }

    def _build_context(self, documents: list[dict]) -> str:
        """Build context string from retrieved documents."""
        if not documents:
            return "No relevant context found in the knowledge base."

        context_parts = []
        for i, doc in enumerate(documents):
            source = doc.get("source", "unknown")
            page = doc.get("page_number", "N/A")
            score = doc.get("similarity_score", 0)
            text = doc.get("text", "")

            context_parts.append(
                f"[Source {i+1}: {source} | Page: {page} | Relevance: {score:.2f}]\n{text}"
            )

        return "\n\n---\n\n".join(context_parts)

    def _format_chat_history(self, chat_history: Optional[list[dict]]) -> str:
        """Format chat history for context."""
        if not chat_history:
            return "No previous conversation."

        formatted = []
        # Only use last 5 exchanges to keep context manageable
        recent = chat_history[-10:]
        for msg in recent:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted.append(f"{role.capitalize()}: {content}")

        return "\n".join(formatted)

    def _calculate_confidence(self, documents: list[dict]) -> float:
        """Calculate confidence score based on retrieval quality."""
        if not documents:
            return 0.0

        scores = [doc.get("similarity_score", 0) for doc in documents]
        if not scores:
            return 0.0

        # Weighted average: top result counts more
        weights = [1.0 / (i + 1) for i in range(len(scores))]
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        total_weight = sum(weights)

        confidence = round(weighted_sum / total_weight, 2)
        return min(max(confidence, 0.0), 1.0)

    def _generate_follow_ups(self, question: str, answer: str) -> list[str]:
        """Generate suggested follow-up questions."""
        try:
            prompt = (
                f"Based on this Q&A, suggest 3 brief follow-up questions a student might ask.\n\n"
                f"Question: {question}\nAnswer: {answer[:500]}\n\n"
                f"Return ONLY the 3 questions, one per line, no numbering."
            )
            response = self.llm.invoke([HumanMessage(content=prompt)])
            follow_ups = [
                q.strip().lstrip("0123456789.-) ")
                for q in response.content.strip().split("\n")
                if q.strip() and len(q.strip()) > 10
            ]
            return follow_ups[:3]
        except Exception:
            return []
