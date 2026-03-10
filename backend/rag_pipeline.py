"""
RAG (Retrieval-Augmented Generation) pipeline for email reply generation.
Orchestrates intent classification, retrieval, prompt building, and generation.
"""

import logging
import time
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent keyword mapping
# ---------------------------------------------------------------------------
_INTENT_KEYWORDS: Dict[str, List[str]] = {
    "inquiry": [
        "question", "asking", "wondering", "curious", "information",
        "details", "how does", "can you explain", "what is", "inquire",
        "pricing", "cost", "price", "quote", "learn more",
    ],
    "complaint": [
        "problem", "issue", "complaint", "broken", "not working",
        "disappointed", "frustrated", "error", "bug", "fail",
        "terrible", "awful", "unacceptable", "poor", "worst",
    ],
    "request": [
        "please", "could you", "would you", "request", "asking for",
        "need", "require", "provide", "send me", "attach", "submit",
        "can you", "would it be possible", "help",
    ],
    "follow_up": [
        "follow up", "following up", "checking in", "status", "update",
        "progress", "any news", "wanted to check", "reminder", "previous",
        "last email", "as discussed",
    ],
    "thank_you": [
        "thank you", "thanks", "appreciate", "grateful", "gratitude",
        "thankful", "many thanks", "much appreciated",
    ],
    "scheduling": [
        "meeting", "schedule", "calendar", "availability", "available",
        "slot", "time", "appointment", "call", "conference", "zoom",
        "teams", "when can", "book",
    ],
    "introduction": [
        "introduce", "introduction", "new to", "joining", "onboard",
        "hello", "hi there", "reach out", "connect", "pleased to meet",
        "my name is", "i am",
    ],
}

# ---------------------------------------------------------------------------
# Mock reply templates
# ---------------------------------------------------------------------------
_REPLY_TEMPLATES: Dict[str, str] = {
    "inquiry": (
        "Thank you for your inquiry. I appreciate you reaching out to us.\n\n"
        "I would be happy to provide you with the information you requested. "
        "Based on the details you have shared, I want to ensure you receive "
        "the most accurate and helpful response possible.\n\n"
        "Please find the relevant details below:\n"
        "- Our team is fully equipped to address your specific needs.\n"
        "- We offer comprehensive solutions tailored to your requirements.\n"
        "- You can expect a detailed proposal within 2 business days.\n\n"
        "Should you have any further questions, please do not hesitate to contact us."
    ),
    "complaint": (
        "I sincerely apologise for the inconvenience you have experienced. "
        "Your feedback is extremely valuable to us, and I understand your frustration.\n\n"
        "We take all complaints very seriously and want to resolve this matter "
        "as quickly as possible. I have escalated your case to our technical team "
        "and we are working on a solution.\n\n"
        "Here is what we will do to resolve this:\n"
        "1. Investigate the root cause of the issue immediately.\n"
        "2. Implement a fix and test thoroughly.\n"
        "3. Follow up with you within 24 hours with a resolution.\n\n"
        "We appreciate your patience and assure you this will not happen again."
    ),
    "request": (
        "Thank you for your request. I have received your message and will "
        "process it promptly.\n\n"
        "I understand the urgency of your request and want to assure you that "
        "we are prioritising it accordingly. Our team will review the details "
        "and prepare everything you need.\n\n"
        "Expected timeline:\n"
        "- Initial review: Within 1 business day\n"
        "- Processing: 2–3 business days\n"
        "- Delivery: By end of week\n\n"
        "I will keep you updated on the progress and notify you once everything "
        "is ready. Please let me know if you have any additional requirements."
    ),
    "follow_up": (
        "Thank you for following up on this matter. I appreciate your patience "
        "while we work on your request.\n\n"
        "Here is a current status update:\n"
        "- The task is currently in progress and proceeding as planned.\n"
        "- We have completed the initial phase and are now moving to the next stage.\n"
        "- We anticipate having a complete update ready by the end of this week.\n\n"
        "I will send you a detailed report once everything has been finalised. "
        "Thank you for your continued understanding and support."
    ),
    "thank_you": (
        "Thank you so much for your kind words — they truly mean a lot to our team!\n\n"
        "It was a genuine pleasure working with you, and we are thrilled to hear "
        "that you are satisfied with the outcome. Our goal is always to deliver "
        "the highest quality of service, and your positive feedback confirms we "
        "are on the right track.\n\n"
        "We look forward to continuing our collaboration and supporting you "
        "in future endeavours. Please do not hesitate to reach out whenever "
        "we can be of assistance."
    ),
    "scheduling": (
        "Thank you for reaching out about scheduling a meeting. I would be "
        "delighted to connect with you.\n\n"
        "Based on my availability, I would like to propose the following time slots:\n"
        "- Option 1: Tuesday at 10:00 AM – 11:00 AM\n"
        "- Option 2: Wednesday at 2:00 PM – 3:00 PM\n"
        "- Option 3: Thursday at 9:00 AM – 10:00 AM\n\n"
        "Please let me know which time works best for you, or suggest an "
        "alternative if none of the above are convenient. I will send a "
        "calendar invitation once we confirm the time."
    ),
    "introduction": (
        "Welcome! It is wonderful to meet you, and we are very excited to "
        "have you on board.\n\n"
        "We have prepared everything to ensure a smooth start for you. "
        "Our team is here to support you every step of the way, and I encourage "
        "you to reach out with any questions or concerns you may have.\n\n"
        "Here are a few things to help you get started:\n"
        "- Review the onboarding documentation in the shared drive.\n"
        "- Schedule introductory calls with key team members.\n"
        "- Join our team communication channels.\n\n"
        "Looking forward to working together and achieving great things as a team!"
    ),
    "other": (
        "Thank you for your email. I have carefully reviewed your message and "
        "appreciate you taking the time to write to us.\n\n"
        "I will give this matter the attention it deserves and respond with a "
        "comprehensive reply shortly. Our team is dedicated to providing "
        "timely and helpful responses.\n\n"
        "If this is urgent, please feel free to call us directly. "
        "Otherwise, you can expect a detailed response within 1–2 business days."
    ),
}

# Tone-specific greetings and closings
_GREETINGS: Dict[str, str] = {
    "formal": "Dear {sender},",
    "friendly": "Hi {sender},",
    "professional": "Hello {sender},",
}
_CLOSINGS: Dict[str, str] = {
    "formal": "Yours sincerely,\nThe Support Team",
    "friendly": "Cheers,\nThe Support Team",
    "professional": "Best regards,\nThe Support Team",
}


class RAGPipeline:
    """Orchestrates the full RAG pipeline for email reply generation."""

    def __init__(self, vector_store: Any, embedding_generator: Any) -> None:
        """
        Initialise the RAG pipeline.

        Args:
            vector_store: A ChromaVectorStore instance.
            embedding_generator: An EmbeddingGenerator instance.
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.top_k = 5
        logger.info("RAGPipeline initialised")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate_reply(
        self,
        subject: str,
        body: str,
        sender: str = "User",
        tone: str = "formal",
    ) -> Dict[str, Any]:
        """
        Generate a reply for an incoming email using the RAG pipeline.

        Args:
            subject: Email subject line.
            body: Email body text.
            sender: Sender name or email address.
            tone: Desired tone ('formal', 'friendly', 'professional').

        Returns:
            Dict containing reply_text, confidence_score, intent,
            retrieved_contexts, and metadata.
        """
        start_time = time.time()

        # 1. Classify intent
        intent = self.classify_intent(f"{subject} {body}")
        logger.info("Classified intent: %s", intent)

        # 2. Build search query
        search_query = f"{subject} {body}"

        # 3. Retrieve similar emails
        contexts = self.vector_store.similarity_search(search_query, top_k=self.top_k)
        logger.info("Retrieved %d context documents", len(contexts))

        # 4. Build augmented prompt
        prompt = self.build_prompt(subject, body, contexts, intent, tone)

        # 5. Generate reply
        raw_reply = self.mock_generate(prompt, intent, tone, subject)

        # 6. Adjust tone
        reply_text = self.adjust_tone(raw_reply, tone, sender)

        # 7. Calculate confidence
        confidence = self.calculate_confidence(contexts)

        elapsed = time.time() - start_time

        return {
            "reply_text": reply_text,
            "confidence_score": confidence,
            "intent": intent,
            "retrieved_contexts": [
                {
                    "content": ctx["content"],
                    "score": ctx["score"],
                    "metadata": ctx["metadata"],
                }
                for ctx in contexts
            ],
            "metadata": {
                "processing_time_seconds": round(elapsed, 3),
                "model": "mock-llm-v1",
                "embedding_model": "all-MiniLM-L6-v2",
                "tone": tone,
                "num_contexts_retrieved": len(contexts),
            },
        }

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def classify_intent(self, text: str) -> str:
        """
        Classify the intent of an email using keyword matching.

        Args:
            text: Combined subject + body text.

        Returns:
            Intent label string.
        """
        text_lower = text.lower()
        scores: Dict[str, int] = {intent: 0 for intent in _INTENT_KEYWORDS}
        for intent, keywords in _INTENT_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    scores[intent] += 1
        best_intent = max(scores, key=lambda k: scores[k])
        if scores[best_intent] == 0:
            return "other"
        return best_intent

    def build_prompt(
        self,
        email_subject: str,
        email_body: str,
        contexts: List[Dict[str, Any]],
        intent: str,
        tone: str,
    ) -> str:
        """
        Construct the augmented prompt for the LLM.

        Args:
            email_subject: Subject of the incoming email.
            email_body: Body of the incoming email.
            contexts: Retrieved context documents.
            intent: Classified intent.
            tone: Desired reply tone.

        Returns:
            Formatted prompt string.
        """
        context_text = ""
        for i, ctx in enumerate(contexts, 1):
            meta = ctx.get("metadata", {})
            context_text += (
                f"\n--- Context {i} (score: {ctx['score']:.3f}) ---\n"
                f"Subject: {meta.get('subject', 'N/A')}\n"
                f"{ctx['content']}\n"
            )

        if not context_text:
            context_text = "No similar past emails retrieved."

        prompt = (
            "SYSTEM:\n"
            "You are a professional email assistant. Generate a helpful, "
            f"context-aware reply in a {tone} tone.\n\n"
            "RETRIEVED CONTEXT (similar past emails):\n"
            f"{context_text}\n\n"
            "INCOMING EMAIL:\n"
            f"Subject: {email_subject}\n"
            f"Body: {email_body}\n\n"
            f"INTENT: {intent}\n\n"
            "TASK: Generate a professional email reply."
        )
        return prompt

    def mock_generate(
        self, prompt: str, intent: str, tone: str, subject: str
    ) -> str:
        """
        Template-based mock LLM reply generator.

        Args:
            prompt: The full augmented prompt (not used by mock, but accepted
                    for API compatibility).
            intent: Classified intent label.
            tone: Desired reply tone.
            subject: Original email subject (for context).

        Returns:
            Generated reply text (without greeting/closing — those are added
            by adjust_tone).
        """
        template = _REPLY_TEMPLATES.get(intent, _REPLY_TEMPLATES["other"])
        return template

    def adjust_tone(self, text: str, tone: str, sender: str = "User") -> str:
        """
        Prepend a greeting and append a closing matched to the desired tone.

        Args:
            text: Core reply body text.
            tone: Desired tone ('formal', 'friendly', 'professional').
            sender: Sender name or email to personalise the greeting.

        Returns:
            Full email reply with greeting and closing.
        """
        tone = tone.lower() if tone else "formal"
        if tone not in _GREETINGS:
            tone = "formal"

        # Use first part of email address as name, capitalised
        display_name = sender.split("@")[0].replace(".", " ").title() if sender else "User"

        greeting = _GREETINGS[tone].format(sender=display_name)
        closing = _CLOSINGS[tone]

        return f"{greeting}\n\n{text}\n\n{closing}"

    def calculate_confidence(self, contexts: List[Dict[str, Any]]) -> float:
        """
        Calculate a confidence score from retrieval similarity scores.

        Args:
            contexts: List of retrieved context dicts with 'score' keys.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        if not contexts:
            return 0.3  # Low confidence when no context is retrieved
        scores = [max(0.0, min(1.0, ctx.get("score", 0.0))) for ctx in contexts]
        # Weighted average: top result counts more
        weight_sum = 0.0
        weighted_score = 0.0
        for i, score in enumerate(scores):
            weight = 1.0 / (i + 1)
            weighted_score += score * weight
            weight_sum += weight
        return round(weighted_score / weight_sum, 3) if weight_sum > 0 else 0.3
