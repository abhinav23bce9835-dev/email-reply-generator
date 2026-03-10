"""
Dataset loading and preprocessing utilities for email data.
"""

import json
import logging
import os
import re
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Loads, cleans, and prepares email datasets for the vector store."""

    @staticmethod
    def load_sample_emails(
        filepath: str = "dataset/sample_emails.json",
    ) -> List[Dict[str, Any]]:
        """
        Load sample emails from a JSON file.

        Args:
            filepath: Path to the JSON file (relative or absolute).

        Returns:
            List of email dictionaries.
        """
        if not os.path.exists(filepath):
            logger.error("Dataset file not found: %s", filepath)
            return []
        try:
            with open(filepath, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            logger.info("Loaded %d emails from %s", len(data), filepath)
            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load dataset: %s", exc)
            return []

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Strip HTML tags and normalise whitespace.

        Args:
            text: Raw input text.

        Returns:
            Cleaned text string.
        """
        if not text:
            return ""
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Normalise whitespace
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def preprocess_email(email_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean and normalise an email dictionary.

        Args:
            email_dict: Raw email dictionary.

        Returns:
            Preprocessed email dictionary.
        """
        processed = email_dict.copy()
        for field in ("subject", "body", "reply"):
            if field in processed:
                processed[field] = DatasetLoader.clean_text(str(processed.get(field, "")))
        # Ensure required fields exist
        processed.setdefault("sender", "unknown@example.com")
        processed.setdefault("recipient", "unknown@example.com")
        processed.setdefault("intent", "other")
        processed.setdefault("domain", "general")
        return processed

    @staticmethod
    def prepare_for_vectorstore(
        emails: List[Dict[str, Any]],
    ) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
        """
        Prepare emails for insertion into ChromaDB.

        Args:
            emails: List of email dictionaries.

        Returns:
            Tuple of (documents, metadatas, ids) ready for ChromaDB.
        """
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []

        for idx, email in enumerate(emails):
            processed = DatasetLoader.preprocess_email(email)

            # Build the document text that will be embedded
            doc_text = (
                f"Subject: {processed.get('subject', '')}\n"
                f"Body: {processed.get('body', '')}"
            )
            documents.append(doc_text)

            # Metadata stored alongside the vector
            meta = {
                "subject": processed.get("subject", ""),
                "sender": processed.get("sender", ""),
                "recipient": processed.get("recipient", ""),
                "intent": processed.get("intent", "other"),
                "domain": processed.get("domain", "general"),
                "reply": processed.get("reply", ""),
                "date": str(processed.get("date", "")),
            }
            metadatas.append(meta)

            # Use the email's own ID if available, otherwise generate one
            email_id = str(processed.get("id", f"email_{idx:04d}"))
            ids.append(email_id)

        logger.info("Prepared %d emails for vector store", len(documents))
        return documents, metadatas, ids
