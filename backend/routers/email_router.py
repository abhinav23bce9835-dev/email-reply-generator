"""Email reply generation router."""
import logging

from fastapi import APIRouter, HTTPException, Request

from backend.models import EmailReply, EmailRequest

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/generate_reply", response_model=EmailReply)
async def generate_reply(request: Request, email_request: EmailRequest) -> EmailReply:
    """
    Generate a context-aware email reply using the RAG pipeline.

    Args:
        request: FastAPI request object containing app state.
        email_request: The incoming email data.

    Returns:
        EmailReply with generated text, confidence score, and retrieved contexts.
    """
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    logger.info(
        "Generating reply for email: subject='%s', tone='%s'",
        email_request.subject[:50],
        email_request.tone,
    )

    try:
        reply = await pipeline.generate_reply(email_request)
        logger.info(
            "Reply generated: intent='%s', confidence=%.2f",
            reply.intent,
            reply.confidence_score,
        )
        return reply
    except Exception as exc:
        logger.error("Failed to generate reply: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to generate email reply: {str(exc)}"
        ) from exc
