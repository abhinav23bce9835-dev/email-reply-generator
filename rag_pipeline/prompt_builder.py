"""Prompt engineering and template construction for email reply generation."""
import logging
from typing import Any

logger = logging.getLogger(__name__)

TONE_INSTRUCTIONS = {
    "formal": (
        "Use a formal and professional tone. Begin with 'Dear [Name],' "
        "and close with 'Sincerely,' or 'Best regards,'."
    ),
    "friendly": (
        "Use a warm and friendly tone. Begin with 'Hi [Name],' "
        "and close with 'Thanks,' or 'Best,'."
    ),
    "concise": (
        "Be brief and to the point. Aim for 3-5 sentences maximum. "
        "Skip pleasantries where possible."
    ),
}

SYSTEM_PROMPT = """You are a professional email assistant specializing in generating
context-aware, accurate, and helpful email replies.

CRITICAL RULES:
1. Only use information from the provided context documents.
2. If the context does not contain relevant information, acknowledge the email politely
   and state that you will follow up with more details.
3. Never fabricate facts, dates, prices, or policies not mentioned in the context.
4. Always maintain a professional and respectful tone appropriate to the request.
5. Structure the reply with a greeting, body paragraphs, and a closing signature.

TONE: {tone_instruction}
"""


class PromptBuilder:
    """Constructs augmented prompts for LLM-based email reply generation."""

    def build_prompt(
        self,
        incoming_email: dict[str, Any],
        retrieved_contexts: list[tuple[str, float, dict[str, Any]]],
        intent: str,
        tone: str = "formal",
    ) -> str:
        """
        Build a complete prompt for the LLM combining system instructions,
        retrieved context, and the incoming email.

        Args:
            incoming_email: Dict with 'subject', 'body', and optionally 'sender'.
            retrieved_contexts: List of (text, score, metadata) tuples from retriever.
            intent: Classified intent of the incoming email.
            tone: Desired tone for the reply (formal/friendly/concise).

        Returns:
            Complete prompt string ready for LLM generation.
        """
        tone_instruction = TONE_INSTRUCTIONS.get(tone, TONE_INSTRUCTIONS["formal"])
        system_section = SYSTEM_PROMPT.format(tone_instruction=tone_instruction)

        # Build context section
        context_section = self._build_context_section(retrieved_contexts)

        # Build incoming email section
        email_section = self._build_email_section(incoming_email)

        # Build task section
        task_section = self._build_task_section(intent, tone)

        prompt = "\n\n".join([
            system_section,
            context_section,
            email_section,
            task_section,
        ])

        logger.debug("Built prompt with %d context documents", len(retrieved_contexts))
        return prompt

    def _build_context_section(
        self, contexts: list[tuple[str, float, dict[str, Any]]]
    ) -> str:
        """Build the context injection section of the prompt."""
        if not contexts:
            return "CONTEXT DOCUMENTS:\nNo relevant context found in the knowledge base."

        context_parts = ["CONTEXT DOCUMENTS:"]
        for i, (doc, score, meta) in enumerate(contexts, 1):
            source = meta.get("source_type", "unknown") if meta else "unknown"
            subject = meta.get("subject", "") if meta else ""
            context_parts.append(
                f"[Document {i}] (relevance: {score:.2f}, source: {source}"
                + (f", subject: {subject}" if subject else "")
                + f")\n{doc}"
            )

        return "\n\n".join(context_parts)

    def _build_email_section(self, email: dict[str, Any]) -> str:
        """Build the incoming email section of the prompt."""
        parts = ["INCOMING EMAIL:"]
        if email.get("subject"):
            parts.append(f"Subject: {email['subject']}")
        if email.get("sender"):
            parts.append(f"From: {email['sender']}")
        parts.append(f"\n{email.get('body', '')}")

        if email.get("thread_history"):
            parts.append("\nTHREAD HISTORY:")
            for prev_email in email["thread_history"][:3]:
                parts.append(f"---\n{prev_email}")

        return "\n".join(parts)

    def _build_task_section(self, intent: str, tone: str) -> str:
        """Build the task instruction section."""
        return (
            f"TASK:\nGenerate a professional email reply to the above email.\n"
            f"The sender's intent appears to be: {intent}\n"
            f"Tone required: {tone}\n\n"
            f"Reply only with the email text (greeting through signature). "
            f"Do not include any preamble like 'Here is the reply:'. "
            f"Start directly with the greeting."
        )
