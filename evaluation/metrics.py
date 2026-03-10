"""
Evaluation metrics for measuring email reply generation quality.
Implements BLEU, ROUGE, and BERTScore metrics.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _safe_texts(
    reference: Optional[str], hypothesis: Optional[str]
) -> Tuple[str, str]:
    """Return sanitised (reference, hypothesis) strings."""
    return (reference or "").strip(), (hypothesis or "").strip()


def compute_bleu(reference: str, hypothesis: str) -> float:
    """
    Compute the BLEU score between a reference and hypothesis string.

    Args:
        reference: Ground-truth reply text.
        hypothesis: Generated reply text.

    Returns:
        BLEU score as a float in [0, 1].
    """
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    ref, hyp = _safe_texts(reference, hypothesis)
    if not ref or not hyp:
        return 0.0

    try:
        nltk.download("punkt_tab", quiet=True)
        nltk.download("punkt", quiet=True)
        ref_tokens = nltk.word_tokenize(ref.lower())
        hyp_tokens = nltk.word_tokenize(hyp.lower())
        smoothing = SmoothingFunction().method1
        score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)
        return float(score)
    except Exception as exc:
        logger.error("BLEU computation error: %s", exc)
        return 0.0


def compute_rouge(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L scores.

    Args:
        reference: Ground-truth reply text.
        hypothesis: Generated reply text.

    Returns:
        Dict with keys 'rouge1', 'rouge2', 'rougeL' (F-measure floats).
    """
    from rouge_score import rouge_scorer

    ref, hyp = _safe_texts(reference, hypothesis)
    if not ref or not hyp:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    try:
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        scores = scorer.score(ref, hyp)
        return {
            "rouge1": round(scores["rouge1"].fmeasure, 4),
            "rouge2": round(scores["rouge2"].fmeasure, 4),
            "rougeL": round(scores["rougeL"].fmeasure, 4),
        }
    except Exception as exc:
        logger.error("ROUGE computation error: %s", exc)
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def compute_bert_score(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Compute BERTScore precision, recall, and F1.

    Args:
        reference: Ground-truth reply text.
        hypothesis: Generated reply text.

    Returns:
        Dict with keys 'precision', 'recall', 'f1'.
    """
    from bert_score import score as bert_score_fn

    ref, hyp = _safe_texts(reference, hypothesis)
    if not ref or not hyp:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    try:
        P, R, F1 = bert_score_fn(
            [hyp], [ref], lang="en", verbose=False
        )
        return {
            "precision": round(float(P[0]), 4),
            "recall": round(float(R[0]), 4),
            "f1": round(float(F1[0]), 4),
        }
    except Exception as exc:
        logger.error("BERTScore computation error: %s", exc)
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def evaluate_single(reference: str, generated: str) -> Dict[str, Any]:
    """
    Compute all evaluation metrics for a single reference/generated pair.

    Args:
        reference: Ground-truth reply.
        generated: Generated reply.

    Returns:
        Dict with bleu, rouge, and bert_score sub-dicts.
    """
    return {
        "bleu": compute_bleu(reference, generated),
        "rouge": compute_rouge(reference, generated),
        "bert_score": compute_bert_score(reference, generated),
    }


def evaluate_batch(test_pairs: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Compute average evaluation metrics over multiple reference/generated pairs.

    Args:
        test_pairs: List of dicts with 'reference' and 'generated' keys.

    Returns:
        Dict with average bleu, rouge, and bert_score values.
    """
    if not test_pairs:
        return {"bleu": 0.0, "rouge": {}, "bert_score": {}}

    bleu_scores: List[float] = []
    rouge1_scores: List[float] = []
    rouge2_scores: List[float] = []
    rougeL_scores: List[float] = []
    bert_p: List[float] = []
    bert_r: List[float] = []
    bert_f1: List[float] = []

    for pair in test_pairs:
        ref = pair.get("reference", "")
        gen = pair.get("generated", "")
        result = evaluate_single(ref, gen)
        bleu_scores.append(result["bleu"])
        rouge1_scores.append(result["rouge"]["rouge1"])
        rouge2_scores.append(result["rouge"]["rouge2"])
        rougeL_scores.append(result["rouge"]["rougeL"])
        bert_p.append(result["bert_score"]["precision"])
        bert_r.append(result["bert_score"]["recall"])
        bert_f1.append(result["bert_score"]["f1"])

    def avg(lst: List[float]) -> float:
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    return {
        "num_pairs": len(test_pairs),
        "bleu": avg(bleu_scores),
        "rouge": {
            "rouge1": avg(rouge1_scores),
            "rouge2": avg(rouge2_scores),
            "rougeL": avg(rougeL_scores),
        },
        "bert_score": {
            "precision": avg(bert_p),
            "recall": avg(bert_r),
            "f1": avg(bert_f1),
        },
    }


def run_evaluation(dataset_path: str = "dataset/sample_emails.json") -> Dict[str, Any]:
    """
    Load the dataset and evaluate the mock RAG replies against reference replies.

    Args:
        dataset_path: Path to the sample emails JSON file.

    Returns:
        Aggregated evaluation metrics dict.
    """
    import sys
    import os

    # Allow running from any working directory
    if not os.path.exists(dataset_path):
        alt_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), dataset_path
        )
        if os.path.exists(alt_path):
            dataset_path = alt_path

    try:
        with open(dataset_path, "r", encoding="utf-8") as fh:
            emails = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Failed to load dataset: %s", exc)
        return {}

    # Build test pairs: reference reply vs. a simple mock (first sentence of body)
    test_pairs: List[Dict[str, str]] = []
    for email in emails:
        reference = email.get("reply", "")
        # Use the email body as the "generated" stand-in for evaluation demo
        generated = email.get("body", "")
        if reference and generated:
            test_pairs.append({"reference": reference, "generated": generated})

    logger.info("Evaluating %d email pairs …", len(test_pairs))
    return evaluate_batch(test_pairs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_evaluation()
    print("\n" + "=" * 60)
    print("  EMAIL REPLY EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Pairs evaluated : {results.get('num_pairs', 0)}")
    print(f"  BLEU            : {results.get('bleu', 0):.4f}")
    rouge = results.get("rouge", {})
    print(f"  ROUGE-1         : {rouge.get('rouge1', 0):.4f}")
    print(f"  ROUGE-2         : {rouge.get('rouge2', 0):.4f}")
    print(f"  ROUGE-L         : {rouge.get('rougeL', 0):.4f}")
    bs = results.get("bert_score", {})
    print(f"  BERTScore-P     : {bs.get('precision', 0):.4f}")
    print(f"  BERTScore-R     : {bs.get('recall', 0):.4f}")
    print(f"  BERTScore-F1    : {bs.get('f1', 0):.4f}")
    print("=" * 60)
