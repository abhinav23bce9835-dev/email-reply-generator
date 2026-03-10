"""Pydantic data models and schemas for the Email Reply Generator API."""
from typing import Any, Optional

from pydantic import BaseModel, Field


class RetrievedContext(BaseModel):
    """A single retrieved context document with similarity score."""

    content: str = Field(..., description="The retrieved document content")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity score")
    source: Optional[str] = Field(default=None, description="Source identifier of the document")
    metadata: Optional[dict[str, Any]] = Field(default=None, description="Additional metadata")


class EmailRequest(BaseModel):
    """Request model for email reply generation."""

    subject: str = Field(..., description="Email subject line")
    body: str = Field(..., description="Email body text")
    sender: Optional[str] = Field(default=None, description="Sender email address")
    recipient: Optional[str] = Field(default=None, description="Recipient email address")
    thread_history: Optional[list[str]] = Field(
        default=None, description="Previous emails in the thread"
    )
    tone: str = Field(default="formal", description="Reply tone: formal, friendly, or concise")


class EmailReply(BaseModel):
    """Response model containing the generated email reply."""

    reply_text: str = Field(..., description="The generated email reply text")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in the reply")
    intent: str = Field(..., description="Classified intent of the incoming email")
    retrieved_contexts: list[RetrievedContext] = Field(
        default_factory=list, description="Context documents used for generation"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the reply"
    )


class EmailDocument(BaseModel):
    """A single email document for the knowledge base."""

    subject: str = Field(..., description="Email subject")
    body: str = Field(..., description="Email body text")
    sender: str = Field(..., description="Sender email address")
    recipient: str = Field(..., description="Recipient email address")
    date: Optional[str] = Field(default=None, description="Date of the email")
    intent: Optional[str] = Field(default=None, description="Email intent label")
    reply: Optional[str] = Field(default=None, description="Expected or actual reply text")


class DatasetUploadRequest(BaseModel):
    """Request model for uploading an email dataset."""

    emails: list[EmailDocument] = Field(..., description="List of email documents to ingest")
    source_type: str = Field(default="manual", description="Source type: manual, enron, synthetic")


class DatasetUploadResponse(BaseModel):
    """Response model for dataset upload."""

    processed_count: int = Field(..., description="Number of documents successfully processed")
    failed_count: int = Field(default=0, description="Number of documents that failed processing")
    message: str = Field(..., description="Status message")


class SearchRequest(BaseModel):
    """Request model for context search."""

    query: str = Field(..., description="Search query text")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    filters: Optional[dict[str, Any]] = Field(default=None, description="Metadata filters")


class SearchResponse(BaseModel):
    """Response model for context search."""

    results: list[RetrievedContext] = Field(..., description="Retrieved context documents")
    query: str = Field(..., description="The original search query")
    total_results: int = Field(..., description="Number of results returned")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(default="1.0.0", description="API version")
    vector_db: str = Field(..., description="Vector DB type")
    llm_provider: str = Field(..., description="LLM provider type")
