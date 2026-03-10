#!/usr/bin/env python3
"""Script to ingest email dataset into the vector store."""
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Load sample emails and ingest them into the vector store."""
    from backend.config import get_settings
    from dataset.loader import DatasetLoader
    from dataset.preprocessor import EmailPreprocessor
    from rag_pipeline.pipeline import create_pipeline

    settings = get_settings()
    logger.info("Starting data ingestion...")
    logger.info("Vector DB: %s", settings.vector_db_type)
    logger.info("Embedding model: %s", settings.embedding_model)

    # Load sample emails
    sample_path = Path(__file__).parent.parent / "dataset" / "sample_emails.json"
    loader = DatasetLoader()
    emails = loader.load_json(str(sample_path))
    logger.info("Loaded %d emails from sample dataset", len(emails))

    # Create pipeline (initializes vector store)
    pipeline = create_pipeline(settings)
    preprocessor = EmailPreprocessor(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    documents = []
    metadatas = []
    ids = []

    for idx, email in enumerate(emails):
        text_parts = [
            f"Subject: {email.get('subject', '')}",
            f"Body: {email.get('body', '')}",
        ]
        if email.get("reply"):
            text_parts.append(f"Reply: {email['reply']}")

        raw_text = "\n".join(text_parts)
        cleaned = preprocessor.clean_email(raw_text)
        chunks = preprocessor.chunk_text(cleaned)

        for chunk_idx, chunk in enumerate(chunks):
            doc_id = f"sample_{idx}_{chunk_idx}"
            documents.append(chunk)
            metadatas.append({
                "subject": email.get("subject", ""),
                "sender": email.get("sender", ""),
                "recipient": email.get("recipient", ""),
                "intent": email.get("intent", "unknown"),
                "domain": email.get("domain", "unknown"),
                "source_type": "sample",
                "email_id": email.get("id", str(idx)),
            })
            ids.append(doc_id)

    if documents:
        pipeline.retriever.vector_store.add_documents(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )
        logger.info(
            "Successfully ingested %d chunks from %d emails into vector store",
            len(documents),
            len(emails),
        )
    else:
        logger.warning("No documents to ingest")

    # Show stats
    stats = pipeline.retriever.vector_store.get_collection_stats()
    logger.info("Vector store stats: %s", stats)


if __name__ == "__main__":
    main()
