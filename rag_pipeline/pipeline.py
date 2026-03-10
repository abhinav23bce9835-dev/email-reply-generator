"""Main RAG orchestration pipeline for email reply generation."""
import logging
import time
from typing import Any

from backend.models import EmailReply, EmailRequest, RetrievedContext

logger = logging.getLogger(__name__)


class RAGEmailPipeline:
    """
    Main RAG-based email reply generation pipeline.

    Orchestrates: intent classification → entity extraction → context
    retrieval → prompt building → LLM generation → post-processing.
    """

    def __init__(
        self,
        retriever: Any,
        prompt_builder: Any,
        generator: Any,
        postprocessor: Any,
        intent_classifier: Any,
        settings: Any,
    ) -> None:
        """
        Initialize the RAG pipeline with all component dependencies.

        Args:
            retriever: EmailRetriever instance.
            prompt_builder: PromptBuilder instance.
            generator: BaseLLMGenerator instance.
            postprocessor: ResponsePostProcessor instance.
            intent_classifier: IntentClassifier instance.
            settings: Application settings.
        """
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.generator = generator
        self.postprocessor = postprocessor
        self.intent_classifier = intent_classifier
        self.settings = settings
        logger.info("RAGEmailPipeline initialized")

    async def generate_reply(self, email_request: EmailRequest) -> EmailReply:
        """
        Generate a context-aware email reply using the full RAG pipeline.

        Steps:
            1. Classify email intent
            2. Extract key entities
            3. Retrieve relevant context from vector DB
            4. Build augmented prompt
            5. Generate response via LLM
            6. Post-process (tone, formatting, signature)
            7. Return structured EmailReply

        Args:
            email_request: The incoming email request.

        Returns:
            EmailReply with reply text, confidence score, intent, and contexts.
        """
        start_time = time.time()

        # 1. Classify intent
        email_text = f"{email_request.subject} {email_request.body}"
        intent = self.intent_classifier.classify(email_text)
        logger.info("Step 1 — Intent classified: '%s'", intent)

        # 2. Extract entities
        entities = self.intent_classifier.extract_entities(email_text)
        logger.info("Step 2 — Entities extracted: %s", list(entities.keys()))

        # 3. Retrieve relevant context
        query = f"{email_request.subject} {email_request.body}"
        retrieved_docs = self.retriever.retrieve(
            query=query,
            top_k=self.settings.top_k_results,
        )
        logger.info("Step 3 — Retrieved %d context documents", len(retrieved_docs))

        # 4. Build augmented prompt
        email_dict = {
            "subject": email_request.subject,
            "body": email_request.body,
            "sender": email_request.sender or "",
            "thread_history": email_request.thread_history,
        }
        prompt = self.prompt_builder.build_prompt(
            incoming_email=email_dict,
            retrieved_contexts=retrieved_docs,
            intent=intent,
            tone=email_request.tone or self.settings.default_tone,
        )
        logger.info("Step 4 — Prompt built (%d chars)", len(prompt))

        # 5. Generate response via LLM
        raw_response = self.generator.generate(
            prompt=prompt,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
        )
        logger.info("Step 5 — LLM response generated (%d chars)", len(raw_response))

        # 6. Post-process
        tone = email_request.tone or self.settings.default_tone
        formatted_reply = self.postprocessor.format_email(
            raw_response=raw_response,
            tone=tone,
        )
        logger.info("Step 6 — Response post-processed")

        # 7. Build response object
        elapsed_ms = int((time.time() - start_time) * 1000)
        confidence_score = self._compute_confidence(retrieved_docs, intent)

        contexts = [
            RetrievedContext(
                content=doc,
                similarity_score=round(score, 4),
                source=meta.get("source_type") if meta else None,
                metadata=meta,
            )
            for doc, score, meta in retrieved_docs
        ]

        return EmailReply(
            reply_text=formatted_reply,
            confidence_score=confidence_score,
            intent=intent,
            retrieved_contexts=contexts,
            metadata={
                "entities": entities,
                "tone": tone,
                "elapsed_ms": elapsed_ms,
                "num_contexts": len(retrieved_docs),
                "llm_provider": self.settings.llm_provider,
                "model": self.settings.model_name,
            },
        )

    def _compute_confidence(
        self,
        retrieved_docs: list[tuple[str, float, Any]],
        intent: str,
    ) -> float:
        """
        Compute a confidence score based on retrieval quality and intent.

        Args:
            retrieved_docs: List of (doc, score, meta) tuples.
            intent: Classified intent string.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        if not retrieved_docs:
            return 0.3 if intent != "other" else 0.2

        avg_score = sum(score for _, score, _ in retrieved_docs) / len(retrieved_docs)
        intent_bonus = 0.1 if intent != "other" else 0.0
        confidence = min(1.0, avg_score + intent_bonus)
        return round(confidence, 4)


def create_pipeline(settings: Any) -> RAGEmailPipeline:
    """
    Factory function to create a fully configured RAG pipeline.

    Args:
        settings: Application settings object.

    Returns:
        Configured RAGEmailPipeline instance.
    """
    from rag_pipeline.generator import create_generator
    from rag_pipeline.intent_classifier import IntentClassifier
    from rag_pipeline.postprocessor import ResponsePostProcessor
    from rag_pipeline.prompt_builder import PromptBuilder
    from rag_pipeline.retriever import EmailRetriever
    from vector_db.embeddings import EmbeddingGenerator
    from vector_db.store_factory import create_vector_store

    logger.info("Creating RAG pipeline...")

    embedding_gen = EmbeddingGenerator(model_name=settings.embedding_model)
    vector_store = create_vector_store(settings, embedding_gen)
    retriever = EmailRetriever(vector_store=vector_store)
    prompt_builder = PromptBuilder()
    generator = create_generator(settings)
    postprocessor = ResponsePostProcessor()
    intent_classifier = IntentClassifier()

    return RAGEmailPipeline(
        retriever=retriever,
        prompt_builder=prompt_builder,
        generator=generator,
        postprocessor=postprocessor,
        intent_classifier=intent_classifier,
        settings=settings,
    )
