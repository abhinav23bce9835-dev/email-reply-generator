# RAG-Based Generative Email Reply Generator

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **production-ready, research-grade** Generative Email Reply Generator using **Retrieval-Augmented Generation (RAG)** with Large Language Models. The system automatically generates context-aware, professional email replies by retrieving relevant past emails/knowledge documents and combining them with an LLM.

---

## Problem Statement

Manual email drafting is time-consuming and inconsistent. Pure LLM approaches suffer from hallucination—generating plausible-sounding but factually incorrect responses. This system solves both problems by grounding LLM responses in retrieved real past emails and knowledge documents.

---

## Why RAG > Simple LLM Prompting

| Aspect | LLM-Only | RAG-Enhanced |
|--------|----------|--------------|
| Factual accuracy | Hallucinates | Grounded in retrieved context |
| Domain adaptation | Generic | Uses company-specific emails |
| Updatable knowledge | Requires retraining | Add docs to vector DB |
| Auditability | Black box | Shows retrieved sources |
| Consistency | Varies | Leverages past approved replies |

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     Email Reply Generator                        │
│                                                                  │
│  ┌──────────┐   ┌─────────────┐   ┌──────────────────────────┐  │
│  │  Email   │──▶│ Preprocessor│──▶│   Intent Classifier      │  │
│  │  Input   │   │ (clean/     │   │   (inquiry/complaint/    │  │
│  │  (API /  │   │  chunk)     │   │    request/follow_up)    │  │
│  │   UI)    │   └─────────────┘   └──────────┬───────────────┘  │
│  └──────────┘                                │                   │
│                                              ▼                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    Embedding Generator                       │ │
│  │          (sentence-transformers/all-MiniLM-L6-v2)           │ │
│  └──────────────────────────┬────────────────────────────────── │ │
│                             │                                    │ │
│         ┌───────────────────▼──────────────────────┐            │ │
│         │              Vector Database              │            │ │
│         │   ┌──────────────┐  ┌──────────────────┐ │            │ │
│         │   │  ChromaDB    │  │      FAISS        │ │            │ │
│         │   │  (default)   │  │   (alternative)   │ │            │ │
│         │   └──────────────┘  └──────────────────┘ │            │ │
│         └───────────────────┬──────────────────────┘            │ │
│                             │                                    │ │
│                             ▼                                    │ │
│         ┌───────────────────────────────────────────┐           │ │
│         │              Retriever                    │           │ │
│         │  (top-k similarity search + filtering)    │           │ │
│         └───────────────────┬───────────────────────┘           │ │
│                             │                                    │ │
│                             ▼                                    │ │
│         ┌───────────────────────────────────────────┐           │ │
│         │           Prompt Builder                  │           │ │
│         │  (system prompt + context + user email)   │           │ │
│         └───────────────────┬───────────────────────┘           │ │
│                             │                                    │ │
│                             ▼                                    │ │
│         ┌───────────────────────────────────────────┐           │ │
│         │              LLM Generator                │           │ │
│         │  OpenAI GPT / Ollama / MockLLM (default)  │           │ │
│         └───────────────────┬───────────────────────┘           │ │
│                             │                                    │ │
│                             ▼                                    │ │
│         ┌───────────────────────────────────────────┐           │ │
│         │           Post Processor                  │           │ │
│         │  (tone, greeting, signature, PII check)   │           │ │
│         └───────────────────┬───────────────────────┘           │ │
│                             │                                    │ │
│                             ▼                                    │ │
│                  ┌──────────────────┐                           │ │
│                  │  Email Reply     │                           │ │
│                  │  (JSON Response) │                           │ │
│                  └──────────────────┘                           │ │
└──────────────────────────────────────────────────────────────────┘
```

---

## Real-World Applications

- **Customer Support Automation**: Auto-draft replies to support tickets
- **Enterprise Email Management**: Handle high-volume internal communications
- **Helpdesk Automation**: Respond to IT/HR queries using past knowledge
- **Sales Follow-ups**: Generate personalized follow-up emails
- **Scheduling Assistant**: Handle meeting requests with calendar-aware responses

---

## Installation

### Prerequisites

- Python 3.11+
- pip

### Quick Start (No API Keys Required)

```bash
# Clone the repository
git clone https://github.com/abhinav23bce9835-dev/email-reply-generator.git
cd email-reply-generator

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env if needed (defaults work out-of-the-box)

# Ingest sample data
python scripts/ingest_data.py

# Start the backend
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# In a separate terminal, start the frontend
streamlit run frontend/streamlit_app.py
```

### Docker Deployment

```bash
# Build and start all services
docker-compose up --build

# Services:
# Backend API:  http://localhost:8000
# Frontend UI:  http://localhost:8501
# ChromaDB:     http://localhost:8010
```

---

## API Usage

### Health Check
```bash
curl http://localhost:8000/api/v1/health
```

### Generate Email Reply
```bash
curl -X POST http://localhost:8000/api/v1/generate_reply \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "Product refund request",
    "body": "Hi, I purchased your software last week but it is not working as expected. I would like a refund.",
    "sender": "customer@example.com",
    "tone": "formal"
  }'
```

### Upload Email Dataset
```bash
curl -X POST http://localhost:8000/api/v1/upload_dataset \
  -H "Content-Type: application/json" \
  -d '{
    "emails": [
      {
        "subject": "Support ticket",
        "body": "My account is locked.",
        "sender": "user@example.com",
        "recipient": "support@company.com",
        "intent": "complaint",
        "reply": "We have unlocked your account. Please try again."
      }
    ],
    "source_type": "manual"
  }'
```

### Search Context
```bash
curl -X POST http://localhost:8000/api/v1/search_context \
  -H "Content-Type: application/json" \
  -d '{
    "query": "refund policy for software products",
    "top_k": 5
  }'
```

---

## Configuration

Copy `.env.example` to `.env` and configure:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `mock` | `openai`, `ollama`, or `mock` |
| `OPENAI_API_KEY` | - | Required if using OpenAI |
| `MODEL_NAME` | `gpt-3.5-turbo` | LLM model name |
| `VECTOR_DB_TYPE` | `chroma` | `chroma` or `faiss` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `TOP_K_RESULTS` | `5` | Number of retrieved contexts |
| `DEFAULT_TONE` | `formal` | `formal`, `friendly`, or `concise` |

---

## Evaluation Metrics

The system supports automated quality evaluation:

- **BLEU Score**: Measures n-gram overlap between generated and reference replies
- **ROUGE-1/2/L**: Recall-oriented unigram/bigram/LCS overlap
- **BERTScore**: Semantic similarity using BERT embeddings

```bash
python scripts/run_evaluation.py
```

---

## Project Structure

```
email-reply-generator/
├── README.md
├── requirements.txt
├── setup.py
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── models.py
│   ├── routers/
│   │   ├── email_router.py
│   │   ├── dataset_router.py
│   │   └── search_router.py
│   └── middleware/
│       └── error_handler.py
├── rag_pipeline/
│   ├── pipeline.py
│   ├── retriever.py
│   ├── prompt_builder.py
│   ├── generator.py
│   ├── postprocessor.py
│   └── intent_classifier.py
├── vector_db/
│   ├── embeddings.py
│   ├── chroma_store.py
│   ├── faiss_store.py
│   └── store_factory.py
├── dataset/
│   ├── preprocessor.py
│   ├── loader.py
│   ├── synthetic_generator.py
│   └── sample_emails.json
├── evaluation/
│   ├── metrics.py
│   ├── evaluator.py
│   └── benchmark.py
├── frontend/
│   ├── streamlit_app.py
│   └── static/style.css
├── research_paper/
│   └── paper_outline.md
├── tests/
│   ├── test_pipeline.py
│   ├── test_retriever.py
│   ├── test_vector_store.py
│   ├── test_preprocessor.py
│   └── test_api.py
└── scripts/
    ├── ingest_data.py
    ├── run_evaluation.py
    └── export_results.py
```

---

## Research Publication

This system is designed for IEEE/Springer conference submission. See `research_paper/paper_outline.md` for the full paper outline.

**Cite as:**
```bibtex
@inproceedings{rag_email_reply_2025,
  title     = {RAG-Enhanced Email Reply Generation: A Retrieval-Augmented Approach
               for Context-Aware Professional Communication},
  author    = {Author, A. and Author, B.},
  booktitle = {Proceedings of the IEEE International Conference on Natural Language Processing},
  year      = {2025},
  pages     = {1--8},
  doi       = {10.1109/ICNLP.2025.000001}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
