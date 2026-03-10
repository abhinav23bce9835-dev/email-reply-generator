"""End-to-end evaluation pipeline for the RAG email reply system."""
import logging
from typing import Any

from evaluation.metrics import compute_bert_score, compute_bleu, compute_rouge

logger = logging.getLogger(__name__)


class SystemEvaluator:
    """Evaluates the quality of generated email replies using multiple metrics."""

    def evaluate_single(self, reference: str, generated: str) -> dict[str, Any]:
        """
        Evaluate a single generated reply against a reference.

        Args:
            reference: Ground truth reference reply.
            generated: Generated reply to evaluate.

        Returns:
            Dictionary containing all metric scores.
        """
        bleu = compute_bleu(reference, generated)
        rouge = compute_rouge(reference, generated)
        bert = compute_bert_score(reference, generated)

        return {
            "bleu": bleu,
            "rouge1": rouge["rouge1"],
            "rouge2": rouge["rouge2"],
            "rougeL": rouge["rougeL"],
            "bert_precision": bert["precision"],
            "bert_recall": bert["recall"],
            "bert_f1": bert["f1"],
        }

    def evaluate_batch(self, test_set: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Evaluate a batch of test samples and compute summary statistics.

        Args:
            test_set: List of dicts with 'reference' and 'generated' keys.

        Returns:
            Summary statistics dict with mean, min, max for each metric.
        """
        if not test_set:
            return {}

        all_scores: dict[str, list[float]] = {
            "bleu": [], "rouge1": [], "rouge2": [], "rougeL": [],
            "bert_precision": [], "bert_recall": [], "bert_f1": [],
        }

        for sample in test_set:
            ref = sample.get("reference", "")
            gen = sample.get("generated", "")
            scores = self.evaluate_single(ref, gen)
            for metric, value in scores.items():
                all_scores[metric].append(value)

        summary = {}
        for metric, values in all_scores.items():
            if values:
                summary[metric] = {
                    "mean": round(sum(values) / len(values), 4),
                    "min": round(min(values), 4),
                    "max": round(max(values), 4),
                }

        logger.info("Batch evaluation complete on %d samples", len(test_set))
        return summary

    def compare_rag_vs_baseline(
        self,
        test_set: list[dict[str, Any]],
        rag_pipeline: Any,
        baseline_pipeline: Any,
    ) -> dict[str, Any]:
        """
        Compare RAG pipeline against a baseline pipeline on a test set.

        Args:
            test_set: List of test samples with 'email_request' and 'reference' keys.
            rag_pipeline: RAGEmailPipeline instance.
            baseline_pipeline: Baseline pipeline (e.g., MockLLM without retrieval).

        Returns:
            Comparison dict with scores for both pipelines.
        """
        rag_samples = []
        baseline_samples = []

        for sample in test_set:
            reference = sample.get("reference", "")
            rag_gen = sample.get("rag_generated", "")
            baseline_gen = sample.get("baseline_generated", "")

            if reference and rag_gen:
                rag_samples.append({"reference": reference, "generated": rag_gen})
            if reference and baseline_gen:
                baseline_samples.append({"reference": reference, "generated": baseline_gen})

        return {
            "rag": self.evaluate_batch(rag_samples),
            "baseline": self.evaluate_batch(baseline_samples),
            "sample_count": len(test_set),
        }
