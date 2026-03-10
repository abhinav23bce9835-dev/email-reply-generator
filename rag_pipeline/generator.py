"""LLM response generation with support for OpenAI, Ollama, and MockLLM."""
import logging
import random
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)

MOCK_REPLY_TEMPLATES = {
    "inquiry": (
        "{greeting}\n\n"
        "Thank you for your inquiry. I appreciate you reaching out to us.\n\n"
        "Based on the information available, I would be happy to assist you with your "
        "question. Our team has extensive experience in this area and we can provide "
        "you with the detailed information you need.\n\n"
        "Please feel free to ask any follow-up questions you may have. We are committed "
        "to providing you with the best possible support.\n\n"
        "{closing}"
    ),
    "complaint": (
        "{greeting}\n\n"
        "Thank you for bringing this matter to our attention. I sincerely apologize for "
        "the inconvenience you have experienced.\n\n"
        "We take all feedback seriously and are committed to resolving this issue "
        "promptly. Our team will investigate the matter and get back to you with a "
        "solution within 24-48 hours.\n\n"
        "We value your business and appreciate your patience as we work to address "
        "this situation.\n\n"
        "{closing}"
    ),
    "request": (
        "{greeting}\n\n"
        "Thank you for your request. I have reviewed the details you provided and "
        "I am pleased to confirm that we can assist you.\n\n"
        "We will process your request and ensure everything is handled according to "
        "your specifications. You can expect to hear from us shortly with an update "
        "on the progress.\n\n"
        "Please do not hesitate to contact us if you need any clarification.\n\n"
        "{closing}"
    ),
    "follow_up": (
        "{greeting}\n\n"
        "Thank you for following up on this matter. I apologize for any delay in "
        "our response.\n\n"
        "I am currently reviewing the status of your inquiry and will provide you with "
        "a comprehensive update shortly. We have been working diligently to address "
        "all pending items.\n\n"
        "I will personally ensure that this receives the attention it deserves.\n\n"
        "{closing}"
    ),
    "scheduling": (
        "{greeting}\n\n"
        "Thank you for reaching out regarding scheduling. I would be delighted to "
        "arrange a meeting at a convenient time.\n\n"
        "Please let me know your availability and preferred meeting format (in-person, "
        "phone, or video call). I will do my best to accommodate your schedule.\n\n"
        "Looking forward to our discussion.\n\n"
        "{closing}"
    ),
    "thank_you": (
        "{greeting}\n\n"
        "Thank you so much for your kind words. It is truly a pleasure to work with "
        "you and I am glad that we were able to assist you effectively.\n\n"
        "We strive to provide the best possible service, and feedback like yours "
        "motivates us to continue improving. Please do not hesitate to reach out "
        "if there is anything else we can do for you.\n\n"
        "{closing}"
    ),
    "introduction": (
        "{greeting}\n\n"
        "Thank you for reaching out and for the introduction. It is a pleasure to "
        "connect with you.\n\n"
        "I look forward to learning more about your background and exploring potential "
        "ways we can work together. Please feel free to share more about your goals "
        "and how we might be able to assist you.\n\n"
        "{closing}"
    ),
    "other": (
        "{greeting}\n\n"
        "Thank you for your email. I have received your message and will review it "
        "carefully.\n\n"
        "A member of our team will respond to you shortly with the relevant "
        "information. We appreciate your patience.\n\n"
        "{closing}"
    ),
}

GREETINGS = {
    "formal": "Dear Valued Customer,",
    "friendly": "Hi there,",
    "concise": "Hello,",
}

CLOSINGS = {
    "formal": "Sincerely,\nThe Support Team",
    "friendly": "Best wishes,\nThe Team",
    "concise": "Regards,\nSupport",
}


class BaseLLMGenerator(ABC):
    """Abstract base class for LLM generators."""

    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 500) -> str:
        """
        Generate a response from the given prompt.

        Args:
            prompt: The complete prompt string.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated text string.
        """


class MockLLMGenerator(BaseLLMGenerator):
    """Template-based mock LLM for testing without external API calls."""

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 500) -> str:
        """
        Generate a template-based mock reply.

        Args:
            prompt: The prompt (used to extract intent and tone hints).
            temperature: Unused in mock generator.
            max_tokens: Unused in mock generator.

        Returns:
            Template-based email reply string.
        """
        # Extract intent hint from prompt
        intent = "other"
        for possible_intent in MOCK_REPLY_TEMPLATES:
            if f"intent appears to be: {possible_intent}" in prompt:
                intent = possible_intent
                break

        # Extract tone hint from prompt
        tone = "formal"
        for possible_tone in ["formal", "friendly", "concise"]:
            if f"Tone required: {possible_tone}" in prompt:
                tone = possible_tone
                break

        template = MOCK_REPLY_TEMPLATES.get(intent, MOCK_REPLY_TEMPLATES["other"])
        greeting = GREETINGS.get(tone, GREETINGS["formal"])
        closing = CLOSINGS.get(tone, CLOSINGS["formal"])

        reply = template.format(greeting=greeting, closing=closing)
        logger.debug("MockLLM generated reply for intent='%s', tone='%s'", intent, tone)
        return reply


class OpenAILLMGenerator(BaseLLMGenerator):
    """LLM generator using OpenAI API (GPT-3.5/GPT-4)."""

    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo") -> None:
        """
        Initialize OpenAI generator.

        Args:
            api_key: OpenAI API key.
            model_name: Model name to use.
        """
        try:
            import openai

            self.client = openai.OpenAI(api_key=api_key)
            self.model_name = model_name
            logger.info("OpenAI generator initialized with model: %s", model_name)
        except ImportError as exc:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            ) from exc

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 500) -> str:
        """
        Generate a reply using OpenAI API.

        Args:
            prompt: The complete prompt string.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated text from the LLM.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""


class OllamaLLMGenerator(BaseLLMGenerator):
    """LLM generator using local Ollama models."""

    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "llama2") -> None:
        """
        Initialize Ollama generator.

        Args:
            base_url: Base URL of the Ollama server.
            model_name: Ollama model name to use.
        """
        import httpx

        self.client = httpx.Client(base_url=base_url, timeout=120.0)
        self.model_name = model_name
        logger.info("Ollama generator initialized: url=%s, model=%s", base_url, model_name)

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 500) -> str:
        """
        Generate a reply using the local Ollama API.

        Args:
            prompt: The complete prompt string.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated text from the Ollama model.
        """
        response = self.client.post(
            "/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
        )
        response.raise_for_status()
        return response.json().get("response", "")


def create_generator(settings: Any) -> BaseLLMGenerator:
    """
    Factory function to create the appropriate LLM generator.

    Args:
        settings: Application settings object.

    Returns:
        An instance of the appropriate BaseLLMGenerator subclass.
    """
    provider = settings.llm_provider.lower()
    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        return OpenAILLMGenerator(
            api_key=settings.openai_api_key,
            model_name=settings.model_name,
        )
    elif provider == "ollama":
        return OllamaLLMGenerator(
            base_url=settings.ollama_base_url,
            model_name=settings.model_name,
        )
    else:  # mock (default)
        logger.info("Using MockLLM generator (no API keys required)")
        return MockLLMGenerator()
