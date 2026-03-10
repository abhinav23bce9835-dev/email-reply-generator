"""Response post-processing: tone adjustment, formatting, PII detection."""
import logging
import re

logger = logging.getLogger(__name__)

# Basic PII detection patterns
PII_PATTERNS = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b"),
    "phone": re.compile(r"\b(?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b"),
}

TONE_GREETINGS = {
    "formal": ["Dear", "Good morning", "Good afternoon", "Good day"],
    "friendly": ["Hi", "Hello", "Hey"],
    "concise": ["Hello", "Hi"],
}

TONE_CLOSINGS = {
    "formal": ["Sincerely,", "Best regards,", "Yours faithfully,", "Kind regards,"],
    "friendly": ["Best,", "Thanks,", "Warm regards,", "Cheers,"],
    "concise": ["Regards,", "Thanks,"],
}


class ResponsePostProcessor:
    """Post-processes LLM-generated email replies for quality and safety."""

    def format_email(
        self,
        raw_response: str,
        tone: str = "formal",
        sender_name: str = "Support Team",
        max_length: int = 1500,
    ) -> str:
        """
        Format and clean the LLM-generated email reply.

        Args:
            raw_response: Raw text from the LLM.
            tone: Desired tone for adjustments.
            sender_name: Name to use in the signature.
            max_length: Maximum character length for the reply.

        Returns:
            Cleaned and formatted email reply string.
        """
        if not raw_response or not raw_response.strip():
            return self._generate_fallback_reply(tone, sender_name)

        text = raw_response.strip()

        # Ensure greeting
        text = self._ensure_greeting(text, tone)

        # Ensure closing signature
        text = self._ensure_closing(text, tone, sender_name)

        # Trim excessive length
        if len(text) > max_length:
            text = self._trim_to_length(text, max_length)

        # PII check (log warning, do not remove — just warn)
        pii_detected = self._detect_pii(text)
        if pii_detected:
            logger.warning(
                "Potential PII detected in reply (%s). Review before sending.",
                ", ".join(pii_detected),
            )

        return text

    def _ensure_greeting(self, text: str, tone: str) -> str:
        """Add a greeting if the text doesn't start with one."""
        greetings = TONE_GREETINGS.get(tone, TONE_GREETINGS["formal"])
        first_line = text.split("\n")[0].strip()

        has_greeting = any(
            first_line.lower().startswith(g.lower()) for g in greetings
        ) or any(
            g.lower() in first_line.lower() for g in ["dear", "hi", "hello", "hey", "good"]
        )

        if not has_greeting:
            default_greeting = greetings[0]
            text = f"{default_greeting},\n\n{text}"

        return text

    def _ensure_closing(self, text: str, tone: str, sender_name: str) -> str:
        """Add a closing signature if the text doesn't have one."""
        closings = TONE_CLOSINGS.get(tone, TONE_CLOSINGS["formal"])
        text_lower = text.lower()

        has_closing = any(c.lower().rstrip(",") in text_lower for c in closings) or any(
            kw in text_lower
            for kw in ["sincerely", "regards", "best", "thanks", "cheers", "yours"]
        )

        if not has_closing:
            default_closing = closings[0]
            text = f"{text}\n\n{default_closing}\n{sender_name}"

        return text

    def _trim_to_length(self, text: str, max_length: int) -> str:
        """Trim text to maximum length while preserving structure."""
        if len(text) <= max_length:
            return text

        # Try to cut at a sentence boundary
        truncated = text[:max_length]
        last_period = truncated.rfind(".")
        if last_period > max_length * 0.7:
            truncated = truncated[: last_period + 1]

        # Append closing if not present
        if not any(
            kw in truncated.lower()
            for kw in ["sincerely", "regards", "best", "thanks"]
        ):
            truncated += "\n\nBest regards,\nSupport Team"

        return truncated

    def _detect_pii(self, text: str) -> list[str]:
        """Detect potential PII patterns in the text."""
        detected = []
        for pii_type, pattern in PII_PATTERNS.items():
            if pattern.search(text):
                detected.append(pii_type)
        return detected

    def _generate_fallback_reply(self, tone: str, sender_name: str) -> str:
        """Generate a minimal fallback reply when LLM returns empty output."""
        greeting = TONE_GREETINGS.get(tone, ["Hello"])[0]
        closing = TONE_CLOSINGS.get(tone, ["Regards,"])[0]
        return (
            f"{greeting},\n\n"
            "Thank you for your email. We have received your message and will "
            "respond shortly with the relevant information.\n\n"
            f"{closing}\n{sender_name}"
        )
