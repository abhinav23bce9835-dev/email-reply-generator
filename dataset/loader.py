"""Dataset loading utilities for various email formats."""
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = {"subject", "body", "sender", "recipient"}


class DatasetLoader:
    """Loads email datasets from various sources."""

    def load_json(self, filepath: str) -> list[dict[str, Any]]:
        """
        Load an email dataset from a JSON file.

        Args:
            filepath: Path to the JSON file containing email records.

        Returns:
            List of email dictionaries.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the JSON format is invalid.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("JSON dataset must be a list of email objects")

        valid = [email for email in data if self.validate_email(email)]
        logger.info("Loaded %d/%d valid emails from '%s'", len(valid), len(data), filepath)
        return valid

    def load_enron(self, directory: str) -> list[dict[str, Any]]:
        """
        Load emails from the Enron dataset maildir format.

        Args:
            directory: Path to the Enron maildir directory.

        Returns:
            List of parsed email dictionaries.
        """
        emails = []
        mail_path = Path(directory)
        if not mail_path.exists():
            logger.warning("Enron dataset directory not found: %s", directory)
            return []

        for mail_file in mail_path.rglob("*."):
            if mail_file.is_file():
                try:
                    email = self._parse_enron_file(mail_file)
                    if email and self.validate_email(email):
                        emails.append(email)
                except Exception as exc:
                    logger.debug("Failed to parse %s: %s", mail_file, exc)

        logger.info("Loaded %d emails from Enron dataset at '%s'", len(emails), directory)
        return emails

    def _parse_enron_file(self, filepath: Path) -> dict[str, Any]:
        """Parse a single Enron maildir file."""
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Simple header parsing
        lines = content.split("\n")
        headers: dict[str, str] = {}
        body_start = 0

        for i, line in enumerate(lines):
            if line.strip() == "":
                body_start = i + 1
                break
            if ":" in line:
                key, _, value = line.partition(":")
                headers[key.strip().lower()] = value.strip()

        body = "\n".join(lines[body_start:]).strip()
        return {
            "subject": headers.get("subject", "(No Subject)"),
            "body": body,
            "sender": headers.get("from", "unknown@enron.com"),
            "recipient": headers.get("to", "unknown@enron.com"),
            "date": headers.get("date", ""),
            "intent": "other",
        }

    def validate_email(self, email_dict: dict[str, Any]) -> bool:
        """
        Validate that an email dictionary has all required fields.

        Args:
            email_dict: Email dictionary to validate.

        Returns:
            True if valid, False otherwise.
        """
        for field in REQUIRED_FIELDS:
            if field not in email_dict or not email_dict[field]:
                logger.debug("Email missing required field: '%s'", field)
                return False
        return True
