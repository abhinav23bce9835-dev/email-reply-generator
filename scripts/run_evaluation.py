#!/usr/bin/env python3
"""Script to run evaluation metrics on the RAG pipeline."""
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def run_evaluation() -> None:
    """Run evaluation on the sample email dataset."""
    from backend.config import get_settings
    from backend.models import EmailRequest
    from dataset.loader import DatasetLoader
    from evaluation.evaluator import SystemEvaluator
    from rag_pipeline.pipeline import create_pipeline

    settings = get_settings()
    logger.info("Running evaluation...")

    # Load test data
    sample_path = Path(__file__).parent.parent / "dataset" / "sample_emails.json"
    loader = DatasetLoader()
    emails = loader.load_json(str(sample_path))
    test_emails = [e for e in emails if e.get("reply")][:10]  # Use first 10 with replies

    if not test_emails:
        logger.error("No test emails with reference replies found")
        return

    # Create pipeline
    pipeline = create_pipeline(settings)

    # Evaluate
    evaluator = SystemEvaluator()
    test_samples = []

    for email in test_emails:
        request = EmailRequest(
            subject=email["subject"],
            body=email["body"],
            sender=email.get("sender"),
            tone="formal",
        )
        try:
            result = await pipeline.generate_reply(request)
            test_samples.append({
                "reference": email["reply"],
                "generated": result.reply_text,
                "intent": result.intent,
                "email_id": email.get("id", "unknown"),
            })
        except Exception as exc:
            logger.warning("Failed to generate reply for email: %s", exc)

    if test_samples:
        summary = evaluator.evaluate_batch(test_samples)
        print("\n=== Evaluation Results ===")
        print(f"Evaluated {len(test_samples)} email pairs\n")
        for metric, stats in summary.items():
            print(f"{metric:20s}: mean={stats['mean']:.4f}, min={stats['min']:.4f}, max={stats['max']:.4f}")
    else:
        logger.warning("No samples to evaluate")


def main() -> None:
    asyncio.run(run_evaluation())


if __name__ == "__main__":
    main()
