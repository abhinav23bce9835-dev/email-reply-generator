# RAG-Enhanced Email Reply Generator

A production-ready **Generative Email Reply Generator** using **Retrieval-Augmented Generation (RAG)** with Large Language Models. The system automatically generates context-aware, professional email replies by retrieving relevant past emails and combining them with an LLM.

---

## Problem Statement

Manual email drafting is slow, error-prone, and inconsistent. Pure LLM approaches hallucinate facts and lack grounding in real organizational context. **RAG solves this** by:

- **Grounded responses**: Retrieves real past emails before generating
- **Updatable knowledge**: Add new emails without retraining
- **Less hallucination**: LLM is constrained by retrieved context
- **Consistent tone**: Post-processing enforces style guidelines

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAG Email Reply Generator                     │
│                                                                 │
│  ┌──────────┐    ┌─────────────┐    ┌──────────────────────┐  │
│  │  Email   │───▶│ Preprocess  │───▶│  Embedding Generator │  │
│  │  Input   │    │  & Intent   │    │  (Sentence-BERT)     │  │
│  └──────────┘    └─────────────┘    └──────────┬───────────┘  │
│                                                 │               │
│                                                 ▼               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   ChromaDB Vector Store                   │  │
│  │          (Persistent, Semantic Similarity Search)         │  │
│  └──────────────────────────┬───────────────────────────────┘  │
│                              │  Top-K Similar Emails            │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Prompt Builder                         │  │
│  │   [System] + [Retrieved Context] + [Incoming Email]      │  │
│  └──────────────────────────┬───────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              LLM (Mock / OpenAI / Ollama)                 │  │
│  │              Intent-aware Template Generation             │  │
│  └──────────────────────────┬───────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  Post-Processor                           │  │
│  │           Tone Adjustment + Formatting                    │  │
│  └──────────────────────────┬───────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│                    ┌─────────────────┐                         │
│                    │  Email Reply    │                          │
│                    └─────────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Option 1: pip install

```bash
# Clone repository
git clone https://github.com/abhinav23bce9835-dev/email-reply-generator.git
cd email-reply-generator

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env

# Start backend
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Start frontend (new terminal)
streamlit run frontend/app.py
```

### Option 2: Docker

```bash
docker-compose -f docker/docker-compose.yml up --build
```

---

## API Usage

### Health Check

```bash
curl http://localhost:8000/
```

### Generate Reply

```bash
curl -X POST http://localhost:8000/api/v1/generate_reply \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "Question about pricing",
    "body": "Hi, I would like to know more about your enterprise pricing plans.",
    "sender": "client@example.com",
    "tone": "formal"
  }'
```

### Search Context

```bash
curl -X POST http://localhost:8000/api/v1/search_context \
  -H "Content-Type: application/json" \
  -d '{"query": "pricing inquiry", "top_k": 3}'
```

### Upload Dataset

```bash
curl -X POST http://localhost:8000/api/v1/upload_dataset \
  -H "Content-Type: application/json" \
  -d '{
    "emails": [
      {
        "subject": "Meeting request",
        "body": "Can we schedule a call?",
        "sender": "boss@company.com",
        "recipient": "team@company.com",
        "intent": "scheduling"
      }
    ]
  }'
```

---

## Streamlit UI

Navigate to `http://localhost:8501` after starting the frontend.

1. Set your **Backend URL** in the sidebar (default: `http://localhost:8000`)
2. Choose a **Tone**: Formal / Friendly / Professional
3. Adjust **Top-K** retrieval results (1–10)
4. Enter your email **Subject**, **Body**, and optional **Sender**
5. Click **Generate Reply**
6. View the generated reply, intent badge, confidence score, and retrieved context

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| BLEU | N-gram overlap between generated and reference reply |
| ROUGE-1/2/L | Recall-oriented overlap (unigram, bigram, longest sequence) |
| BERTScore | Semantic similarity using BERT embeddings |

Run evaluation:

```bash
python evaluation/metrics.py
```

---

## Project Structure

```
email-reply-generator/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── backend/
│   ├── __init__.py
│   ├── main.py              # FastAPI entry point
│   ├── rag_pipeline.py      # RAG orchestration
│   ├── embeddings.py        # Sentence Transformers
│   ├── vector_store.py      # ChromaDB
│   └── dataset_loader.py    # Data loading
├── frontend/
│   └── app.py               # Streamlit UI
├── dataset/
│   └── sample_emails.json   # 30+ sample emails
├── evaluation/
│   ├── __init__.py
│   └── metrics.py           # BLEU, ROUGE, BERTScore
├── tests/
│   ├── __init__.py
│   └── test_api.py          # FastAPI tests
└── docker/
    ├── Dockerfile
    └── docker-compose.yml
```

---

## Environment Variables

See `.env.example` for all configuration options. Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `mock` | LLM backend (`mock`, `openai`, `ollama`) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence Transformers model |
| `TOP_K_RESULTS` | `5` | Number of retrieved contexts |
| `DEFAULT_TONE` | `formal` | Default reply tone |
| `CHROMA_PERSIST_DIR` | `./chroma_data` | Vector DB storage path |
