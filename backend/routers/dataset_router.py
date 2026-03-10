"""Dataset upload and ingestion router."""
import logging

from fastapi import APIRouter, HTTPException, Request

from backend.models import DatasetUploadRequest, DatasetUploadResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/upload_dataset", response_model=DatasetUploadResponse)
async def upload_dataset(
    request: Request, dataset_request: DatasetUploadRequest
) -> DatasetUploadResponse:
    """
    Upload and ingest an email dataset into the vector store.

    Preprocesses emails, generates embeddings, and stores them in the
    configured vector database.

    Args:
        request: FastAPI request object containing app state.
        dataset_request: Dataset upload request with email documents.

    Returns:
        DatasetUploadResponse with counts of processed/failed documents.
    """
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    if not dataset_request.emails:
        raise HTTPException(status_code=400, detail="No emails provided in the request")

    logger.info(
        "Ingesting dataset: %d emails, source_type='%s'",
        len(dataset_request.emails),
        dataset_request.source_type,
    )

    processed_count = 0
    failed_count = 0

    from dataset.preprocessor import EmailPreprocessor

    preprocessor = EmailPreprocessor()

    documents = []
    metadatas = []
    ids = []

    for idx, email_doc in enumerate(dataset_request.emails):
        try:
            # Build document text combining subject, body, and reply
            text_parts = [f"Subject: {email_doc.subject}", f"Body: {email_doc.body}"]
            if email_doc.reply:
                text_parts.append(f"Reply: {email_doc.reply}")

            raw_text = "\n".join(text_parts)
            cleaned_text = preprocessor.clean_email(raw_text)

            chunks = preprocessor.chunk_text(cleaned_text)
            for chunk_idx, chunk in enumerate(chunks):
                doc_id = f"{dataset_request.source_type}_{idx}_{chunk_idx}"
                documents.append(chunk)
                metadatas.append(
                    {
                        "subject": email_doc.subject,
                        "sender": email_doc.sender,
                        "recipient": email_doc.recipient,
                        "intent": email_doc.intent or "unknown",
                        "source_type": dataset_request.source_type,
                        "date": email_doc.date or "",
                    }
                )
                ids.append(doc_id)

            processed_count += 1
        except Exception as exc:
            logger.warning("Failed to process email %d: %s", idx, exc)
            failed_count += 1

    if documents:
        pipeline.retriever.vector_store.add_documents(
            documents=documents, metadatas=metadatas, ids=ids
        )
        logger.info("Stored %d document chunks from %d emails", len(documents), processed_count)

    return DatasetUploadResponse(
        processed_count=processed_count,
        failed_count=failed_count,
        message=f"Successfully processed {processed_count} emails ({len(documents)} chunks stored)",
    )
