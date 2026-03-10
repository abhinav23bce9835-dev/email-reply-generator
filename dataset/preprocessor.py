"""Email preprocessing: cleaning, chunking, thread extraction."""
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Patterns for signature detection
SIGNATURE_PATTERNS = [
    re.compile(r"--\s*\n", re.MULTILINE),
    re.compile(r"^(Best regards?|Sincerely|Thanks?|Cheers|Regards?)[,.]?\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"Sent from my (iPhone|iPad|Android|BlackBerry)", re.IGNORECASE),
    re.compile(r"^_{3,}\s*$", re.MULTILINE),
]

# Pattern for HTML tags
HTML_PATTERN = re.compile(r"<[^>]+>")

# Pattern for quoted reply text
QUOTED_TEXT_PATTERNS = [
    re.compile(r"^>.*$", re.MULTILINE),
    re.compile(r"On .+ wrote:", re.IGNORECASE),
    re.compile(r"-----Original Message-----", re.IGNORECASE),
    re.compile(r"From:.*\nSent:.*\nTo:.*\nSubject:", re.IGNORECASE | re.MULTILINE),
]


class EmailPreprocessor:
    """Preprocesses raw email text for embedding and storage."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:
        """
        Initialize the preprocessor.

        Args:
            chunk_size: Target size for text chunks in characters.
            chunk_overlap: Character overlap between consecutive chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def clean_email(self, text: str) -> str:
        """
        Clean raw email text by removing HTML, signatures, and normalizing whitespace.

        Args:
            text: Raw email text.

        Returns:
            Cleaned email text string.
        """
        if not text:
            return ""

        # Remove HTML tags
        text = HTML_PATTERN.sub(" ", text)

        # Remove quoted reply text
        text = self.remove_quoted_text(text)

        # Remove email signatures
        for pattern in SIGNATURE_PATTERNS:
            match = pattern.search(text)
            if match:
                text = text[: match.start()]

        # Normalize whitespace
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        return text

    def extract_headers(self, raw_email: str) -> dict[str, str]:
        """
        Extract standard email headers from raw email text.

        Args:
            raw_email: Raw email text possibly including header lines.

        Returns:
            Dictionary with keys: from, to, subject, date.
        """
        headers: dict[str, str] = {"from": "", "to": "", "subject": "", "date": ""}
        header_patterns = {
            "from": re.compile(r"^From:\s*(.+)$", re.MULTILINE | re.IGNORECASE),
            "to": re.compile(r"^To:\s*(.+)$", re.MULTILINE | re.IGNORECASE),
            "subject": re.compile(r"^Subject:\s*(.+)$", re.MULTILINE | re.IGNORECASE),
            "date": re.compile(r"^Date:\s*(.+)$", re.MULTILINE | re.IGNORECASE),
        }
        for key, pattern in header_patterns.items():
            match = pattern.search(raw_email)
            if match:
                headers[key] = match.group(1).strip()
        return headers

    def split_thread(self, email_text: str) -> list[str]:
        """
        Split an email thread into individual messages.

        Args:
            email_text: Email text potentially containing forwarded/replied chains.

        Returns:
            List of individual email texts in the thread.
        """
        # Split on common thread delimiters
        delimiter_patterns = [
            r"-----Original Message-----",
            r"On .+ wrote:",
            r"From:.*\nSent:.*\n",
        ]
        combined = "|".join(delimiter_patterns)
        parts = re.split(combined, email_text, flags=re.IGNORECASE)
        return [p.strip() for p in parts if p.strip()]

    def chunk_text(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> list[str]:
        """
        Split text into overlapping chunks for embedding.

        Args:
            text: Input text to chunk.
            chunk_size: Target chunk size in characters.
            overlap: Character overlap between chunks.

        Returns:
            List of text chunk strings.
        """
        chunk_size = chunk_size or self.chunk_size
        overlap = overlap if overlap is not None else self.chunk_overlap

        if not text or len(text) <= chunk_size:
            return [text] if text.strip() else []

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                break_point = text.rfind(". ", start, end)
                if break_point == -1:
                    break_point = text.rfind("\n", start, end)
                if break_point != -1 and break_point > start:
                    end = break_point + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap
            if start >= len(text):
                break

        return chunks

    def remove_quoted_text(self, text: str) -> str:
        """
        Remove quoted reply text from an email body.

        Args:
            text: Email body text.

        Returns:
            Text with quoted portions removed.
        """
        for pattern in QUOTED_TEXT_PATTERNS:
            match = pattern.search(text)
            if match:
                text = text[: match.start()]
        # Remove lines starting with >
        lines = [line for line in text.split("\n") if not line.strip().startswith(">")]
        return "\n".join(lines)
