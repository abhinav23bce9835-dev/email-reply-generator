"""Evaluation metrics: BLEU, ROUGE, and BERTScore."""
import logging
from typing import Any

logger = logging.getLogger(__name__)


def compute_bleu(reference: str, hypothesis: str) -> float:
    """
    Compute BLEU score between a reference and hypothesis string.

    Args:
        reference: Ground truth reference text.
        hypothesis: Generated hypothesis text.

    Returns:
        BLEU score as a float between 0.0 and 1.0.
    """
    if not reference or not hypothesis:
        return 0.0

    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()

        if not ref_tokens or not hyp_tokens:
            return 0.0

        smoothing = SmoothingFunction().method1
        score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)
        return round(float(score), 4)
    except ImportError:
        logger.warning("nltk not available for BLEU computation")
        return _simple_bleu(reference, hypothesis)


def _simple_bleu(reference: str, hypothesis: str) -> float:
    """Simplified BLEU approximation without NLTK."""
    ref_tokens = set(reference.lower().split())
    hyp_tokens = hypothesis.lower().split()
    if not hyp_tokens:
        return 0.0
    overlap = sum(1 for t in hyp_tokens if t in ref_tokens)
    return round(overlap / len(hyp_tokens), 4)


def compute_rouge(reference: str, hypothesis: str) -> dict[str, float]:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L scores.

    Args:
        reference: Ground truth reference text.
        hypothesis: Generated hypothesis text.

    Returns:
        Dictionary with rouge1, rouge2, rougeL scores (F1 values).
    """
    if not reference or not hypothesis:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        return {
            "rouge1": round(scores["rouge1"].fmeasure, 4),
            "rouge2": round(scores["rouge2"].fmeasure, 4),
            "rougeL": round(scores["rougeL"].fmeasure, 4),
        }
    except ImportError:
        logger.warning("rouge-score not available, using simple ROUGE approximation")
        return _simple_rouge(reference, hypothesis)


def _simple_rouge(reference: str, hypothesis: str) -> dict[str, float]:
    """Simplified ROUGE-1 approximation without rouge-score package."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not ref_tokens or not hyp_tokens:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    ref_set = set(ref_tokens)
    hyp_set = set(hyp_tokens)
    overlap = len(ref_set & hyp_set)
    precision = overlap / len(hyp_set) if hyp_set else 0.0
    recall = overlap / len(ref_set) if ref_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"rouge1": round(f1, 4), "rouge2": 0.0, "rougeL": round(f1 * 0.9, 4)}


def compute_bert_score(reference: str, hypothesis: str) -> dict[str, float]:
    """
    Compute BERTScore (precision, recall, F1) between reference and hypothesis.

    Args:
        reference: Ground truth reference text.
        hypothesis: Generated hypothesis text.

    Returns:
        Dictionary with precision, recall, and f1 BERTScore values.
    """
    if not reference or not hypothesis:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    try:
        from bert_score import score as bert_score_fn

        P, R, F1 = bert_score_fn(
            [hypothesis],
            [reference],
            lang="en",
            verbose=False,
        )
        return {
            "precision": round(float(P[0]), 4),
            "recall": round(float(R[0]), 4),
            "f1": round(float(F1[0]), 4),
        }
    except ImportError:
        logger.warning("bert-score not available, returning approximation")
        # Fallback: use simple token overlap as approximation
        bleu = compute_bleu(reference, hypothesis)
        return {"precision": bleu, "recall": bleu, "f1": bleu}
    except Exception as exc:
        logger.error("BERTScore computation failed: %s", exc)
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
