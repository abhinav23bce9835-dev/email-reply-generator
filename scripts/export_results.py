#!/usr/bin/env python3
"""Script to export evaluation results to CSV."""
import asyncio
import csv
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def export_results() -> None:
    """Run evaluation and export results to CSV."""
    from backend.config import get_settings
    from backend.models import EmailRequest
    from dataset.loader import DatasetLoader
    from evaluation.evaluator import SystemEvaluator
    from evaluation.metrics import compute_bleu, compute_rouge
    from rag_pipeline.pipeline import create_pipeline

    settings = get_settings()
    sample_path = Path(__file__).parent.parent / "dataset" / "sample_emails.json"
    loader = DatasetLoader()
    emails = loader.load_json(str(sample_path))
    test_emails = [e for e in emails if e.get("reply")]

    pipeline = create_pipeline(settings)
    evaluator = SystemEvaluator()

    output_path = Path("results") / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    output_path.parent.mkdir(exist_ok=True)

    rows = []
    for email in test_emails:
        request = EmailRequest(
            subject=email["subject"],
            body=email["body"],
            sender=email.get("sender"),
            tone="formal",
        )
        try:
            result = await pipeline.generate_reply(request)
            scores = evaluator.evaluate_single(email["reply"], result.reply_text)
            rows.append({
                "email_id": email.get("id", ""),
                "subject": email["subject"],
                "intent": result.intent,
                "confidence": result.confidence_score,
                "num_contexts": len(result.retrieved_contexts),
                **scores,
            })
        except Exception as exc:
            logger.warning("Failed for email %s: %s", email.get("id"), exc)

    if rows:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Results exported to: %s", output_path)
        print(f"\nExported {len(rows)} rows to: {output_path}")
    else:
        logger.warning("No results to export")


def main() -> None:
    asyncio.run(export_results())


if __name__ == "__main__":
    main()
