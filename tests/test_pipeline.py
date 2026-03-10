"""Tests for the RAG pipeline with mock components."""
import asyncio
from unittest.mock import MagicMock

import pytest

from backend.models import EmailRequest


def make_email_request(**kwargs):
    """Helper to create EmailRequest objects."""
    defaults = {
        "subject": "Test subject",
        "body": "Test email body requesting help.",
        "sender": "test@example.com",
        "tone": "formal",
    }
    defaults.update(kwargs)
    return EmailRequest(**defaults)


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.llm_provider = "mock"
    settings.model_name = "mock"
    settings.top_k_results = 3
    settings.temperature = 0.7
    settings.max_tokens = 200
    settings.default_tone = "formal"
    return settings


@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    retriever.retrieve.return_value = [
        ("Relevant context about the topic", 0.85, {"source_type": "test"}),
        ("Another relevant context", 0.70, {"source_type": "test"}),
    ]
    return retriever


@pytest.fixture
def mock_intent_classifier():
    classifier = MagicMock()
    classifier.classify.return_value = "request"
    classifier.extract_entities.return_value = {
        "names": ["John"],
        "dates": [],
        "topics": [],
        "products": [],
        "emails": [],
    }
    return classifier


@pytest.fixture
def mock_prompt_builder():
    builder = MagicMock()
    builder.build_prompt.return_value = "SYSTEM: ...\nCONTEXT: ...\nUSER: ...\nTASK: ..."
    return builder


@pytest.fixture
def mock_generator():
    gen = MagicMock()
    gen.generate.return_value = (
        "Dear Customer,\n\nThank you for your request.\n\nBest regards,\nSupport"
    )
    return gen


@pytest.fixture
def mock_postprocessor():
    pp = MagicMock()
    pp.format_email.return_value = (
        "Dear Customer,\n\nThank you for your request.\n\nBest regards,\nSupport"
    )
    return pp


@pytest.fixture
def pipeline(
    mock_retriever,
    mock_prompt_builder,
    mock_generator,
    mock_postprocessor,
    mock_intent_classifier,
    mock_settings,
):
    from rag_pipeline.pipeline import RAGEmailPipeline

    return RAGEmailPipeline(
        retriever=mock_retriever,
        prompt_builder=mock_prompt_builder,
        generator=mock_generator,
        postprocessor=mock_postprocessor,
        intent_classifier=mock_intent_classifier,
        settings=mock_settings,
    )


def test_pipeline_generate_reply_returns_email_reply(pipeline):
    """Test that pipeline returns a valid EmailReply object."""
    from backend.models import EmailReply

    email_request = make_email_request()
    result = asyncio.get_event_loop().run_until_complete(
        pipeline.generate_reply(email_request)
    )
    assert isinstance(result, EmailReply)


def test_pipeline_classifies_intent(pipeline, mock_intent_classifier):
    """Test that pipeline classifies email intent."""
    email_request = make_email_request()
    result = asyncio.get_event_loop().run_until_complete(
        pipeline.generate_reply(email_request)
    )
    mock_intent_classifier.classify.assert_called_once()
    assert result.intent == "request"


def test_pipeline_retrieves_context(pipeline, mock_retriever):
    """Test that pipeline calls retriever with correct parameters."""
    email_request = make_email_request()
    asyncio.get_event_loop().run_until_complete(pipeline.generate_reply(email_request))
    mock_retriever.retrieve.assert_called_once()


def test_pipeline_builds_prompt(pipeline, mock_prompt_builder):
    """Test that pipeline builds augmented prompt."""
    email_request = make_email_request()
    asyncio.get_event_loop().run_until_complete(pipeline.generate_reply(email_request))
    mock_prompt_builder.build_prompt.assert_called_once()


def test_pipeline_generates_with_llm(pipeline, mock_generator):
    """Test that pipeline calls the LLM generator."""
    email_request = make_email_request()
    asyncio.get_event_loop().run_until_complete(pipeline.generate_reply(email_request))
    mock_generator.generate.assert_called_once()


def test_pipeline_includes_retrieved_contexts(pipeline):
    """Test that returned reply includes retrieved contexts."""
    email_request = make_email_request()
    result = asyncio.get_event_loop().run_until_complete(
        pipeline.generate_reply(email_request)
    )
    assert len(result.retrieved_contexts) == 2
    for ctx in result.retrieved_contexts:
        assert ctx.similarity_score >= 0.0


def test_pipeline_confidence_score_range(pipeline):
    """Test that confidence score is within valid range."""
    email_request = make_email_request()
    result = asyncio.get_event_loop().run_until_complete(
        pipeline.generate_reply(email_request)
    )
    assert 0.0 <= result.confidence_score <= 1.0


def test_pipeline_metadata_contains_intent(pipeline):
    """Test that result metadata contains expected fields."""
    email_request = make_email_request()
    result = asyncio.get_event_loop().run_until_complete(
        pipeline.generate_reply(email_request)
    )
    assert "tone" in result.metadata
    assert "elapsed_ms" in result.metadata
    assert "entities" in result.metadata


def test_pipeline_uses_custom_tone(pipeline, mock_prompt_builder):
    """Test that custom tone is passed through the pipeline."""
    email_request = make_email_request(tone="friendly")
    result = asyncio.get_event_loop().run_until_complete(
        pipeline.generate_reply(email_request)
    )
    # Check tone was used
    call_kwargs = mock_prompt_builder.build_prompt.call_args[1]
    assert call_kwargs.get("tone") == "friendly" or result.metadata.get("tone") == "friendly"
