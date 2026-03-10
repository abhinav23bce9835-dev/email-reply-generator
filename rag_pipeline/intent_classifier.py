"""Keyword-based email intent classifier and entity extractor."""
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Intent keywords mapping
INTENT_KEYWORDS: dict[str, list[str]] = {
    "inquiry": [
        "what", "how", "when", "where", "why", "could you", "can you", "would you",
        "please tell", "i would like to know", "i want to know", "information about",
        "question about", "asking about", "wondering",
    ],
    "complaint": [
        "problem", "issue", "error", "bug", "not working", "broken", "failed",
        "disappointed", "frustrated", "unhappy", "poor", "terrible", "worst",
        "complaint", "unsatisfied", "defective", "refund",
    ],
    "request": [
        "please", "request", "need", "require", "would like", "asking for",
        "provide", "send me", "give me", "help me", "assist", "support",
        "i need", "can you please",
    ],
    "follow_up": [
        "following up", "follow up", "checking in", "any update", "status",
        "progress", "heard back", "still waiting", "reminder", "as discussed",
        "per our conversation", "previously",
    ],
    "introduction": [
        "introducing", "my name is", "i am", "pleased to meet", "new to",
        "joining", "hello", "hi there", "reaching out",
    ],
    "thank_you": [
        "thank you", "thanks", "grateful", "appreciate", "many thanks",
        "much appreciated", "thankful",
    ],
    "scheduling": [
        "meeting", "schedule", "appointment", "calendar", "available",
        "availability", "time slot", "book", "reschedule", "call",
        "conference", "discuss",
    ],
}


class IntentClassifier:
    """Classifies email intent using keyword-based matching."""

    def classify(self, email_text: str) -> str:
        """
        Classify the intent of an email based on keyword matching.

        Args:
            email_text: The email body and/or subject text.

        Returns:
            Intent label string (one of: inquiry, complaint, request,
            follow_up, introduction, thank_you, scheduling, other).
        """
        text_lower = email_text.lower()
        scores: dict[str, int] = {}

        for intent, keywords in INTENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            scores[intent] = score

        best_intent = max(scores, key=lambda k: scores[k])
        if scores[best_intent] == 0:
            return "other"

        logger.debug("Intent scores: %s -> '%s'", scores, best_intent)
        return best_intent

    def extract_entities(self, email_text: str) -> dict[str, Any]:
        """
        Extract named entities and key information from email text.

        Args:
            email_text: The email body/subject text.

        Returns:
            Dictionary with extracted names, dates, topics, and products.
        """
        entities: dict[str, Any] = {
            "names": [],
            "dates": [],
            "topics": [],
            "products": [],
            "emails": [],
        }

        # Extract email addresses
        email_pattern = r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
        entities["emails"] = re.findall(email_pattern, email_text)

        # Extract dates (basic pattern)
        date_patterns = [
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
            r"\b(?:January|February|March|April|May|June|July|August|September|"
            r"October|November|December)\s+\d{1,2},?\s+\d{4}\b",
            r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
            r"\b(?:today|tomorrow|yesterday|next week|last week)\b",
        ]
        for pattern in date_patterns:
            found = re.findall(pattern, email_text, re.IGNORECASE)
            entities["dates"].extend(found)

        # Extract capitalized words as potential named entities (names/products)
        cap_words = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", email_text)
        # Filter common words
        stopwords = {
            "Dear", "Hello", "Hi", "Thanks", "Thank", "Best", "Regards",
            "Sincerely", "Please", "The", "This", "That", "Your", "Our",
            "From", "Subject", "Body",
        }
        filtered = [w for w in cap_words if w not in stopwords]
        entities["names"] = list(set(filtered[:5]))

        return entities
