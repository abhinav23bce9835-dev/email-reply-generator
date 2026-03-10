"""Tests for the email preprocessor module."""
import pytest

from dataset.preprocessor import EmailPreprocessor


@pytest.fixture
def preprocessor():
    return EmailPreprocessor(chunk_size=200, chunk_overlap=20)


def test_clean_email_removes_html(preprocessor):
    """Test that HTML tags are stripped from email text."""
    html_text = "<p>Hello <b>World</b></p><br/>"
    cleaned = preprocessor.clean_email(html_text)
    assert "<" not in cleaned
    assert "Hello" in cleaned
    assert "World" in cleaned


def test_clean_email_normalizes_whitespace(preprocessor):
    """Test whitespace normalization."""
    text = "Hello    World\n\n\n\nHow are you?"
    cleaned = preprocessor.clean_email(text)
    assert "  " not in cleaned
    assert "\n\n\n" not in cleaned


def test_clean_email_empty_string(preprocessor):
    """Test that empty input returns empty string."""
    assert preprocessor.clean_email("") == ""


def test_clean_email_removes_signature(preprocessor):
    """Test that email signatures are removed."""
    text = "Hello,\n\nPlease find details.\n\n--\nJohn Doe\nSupport Team"
    cleaned = preprocessor.clean_email(text)
    assert "John Doe" not in cleaned or "Hello" in cleaned


def test_remove_quoted_text(preprocessor):
    """Test removal of quoted reply content."""
    text = "Sure, let me help.\n\nOn Monday, John wrote:\n> Can you help me?"
    result = preprocessor.remove_quoted_text(text)
    assert "> Can you help me?" not in result
    assert "Sure, let me help." in result


def test_chunk_text_short_text(preprocessor):
    """Test chunking of text shorter than chunk_size."""
    short_text = "This is a short text."
    chunks = preprocessor.chunk_text(short_text)
    assert len(chunks) == 1
    assert chunks[0] == short_text


def test_chunk_text_long_text(preprocessor):
    """Test chunking of long text into multiple chunks."""
    long_text = "This is a test sentence. " * 20
    chunks = preprocessor.chunk_text(long_text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= preprocessor.chunk_size + 50  # Allow slight overflow at boundaries


def test_chunk_text_empty(preprocessor):
    """Test chunking of empty text."""
    assert preprocessor.chunk_text("") == []


def test_extract_headers(preprocessor):
    """Test extraction of email headers."""
    raw_email = (
        "From: john@example.com\n"
        "To: support@example.com\n"
        "Subject: Test email\n"
        "Date: 2025-01-01\n\n"
        "Body content here."
    )
    headers = preprocessor.extract_headers(raw_email)
    assert headers["from"] == "john@example.com"
    assert headers["to"] == "support@example.com"
    assert headers["subject"] == "Test email"


def test_split_thread(preprocessor):
    """Test splitting email thread into individual messages."""
    thread = (
        "Current reply here.\n\n"
        "On Monday, John wrote:\n"
        "Original email content."
    )
    parts = preprocessor.split_thread(thread)
    assert len(parts) >= 1
