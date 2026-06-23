"""RAGAS evaluation for RAG pipeline quality."""

import time
from typing import Optional

from loguru import logger

from backend.config import get_settings


class RAGASEvaluator:
    """Evaluate RAG responses using RAGAS metrics."""

    def __init__(self):
        self.settings = get_settings()
        self._available = not self.settings.ragas_disabled

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: Optional[str] = None,
    ) -> dict:
        start = time.time()
        if not self._available:
            return self._heuristic_eval(question, answer, contexts, start)

        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import answer_relevancy, context_precision, faithfulness

            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
            }
            if ground_truth:
                data["ground_truth"] = [ground_truth]

            dataset = Dataset.from_dict(data)
            metrics = [faithfulness, context_precision, answer_relevancy]
            result = evaluate(dataset, metrics=metrics)

            scores = result.to_pandas().iloc[0].to_dict()
            latency = round(time.time() - start, 3)
            return {
                "faithfulness": round(float(scores.get("faithfulness", 0) or 0), 4),
                "context_precision": round(float(scores.get("context_precision", 0) or 0), 4),
                "answer_relevancy": round(float(scores.get("answer_relevancy", 0) or 0), 4),
                "latency_seconds": latency,
                "method": "ragas",
            }
        except Exception as e:
            logger.warning(f"RAGAS evaluation failed, using heuristic: {e}")
            return self._heuristic_eval(question, answer, contexts, start)

    def _heuristic_eval(self, question: str, answer: str, contexts: list[str], start: float) -> dict:
        ctx_text = " ".join(contexts).lower()
        ans_lower = answer.lower()
        q_words = set(question.lower().split())
        overlap = len(q_words & set(ans_lower.split())) / max(len(q_words), 1)
        ctx_overlap = sum(1 for w in q_words if w in ctx_text) / max(len(q_words), 1)
        return {
            "faithfulness": round(min(ctx_overlap + 0.2, 1.0), 4),
            "context_precision": round(min(ctx_overlap, 1.0), 4),
            "answer_relevancy": round(min(overlap + 0.1, 1.0), 4),
            "latency_seconds": round(time.time() - start, 3),
            "method": "heuristic",
        }
