# DocuMind: A Retrieval-Augmented Generation System for Document Question Answering

**Fabio Fasciglione**  
Independent Project · April 2026  

---

## Abstract

We present **DocuMind**, a full-stack Retrieval-Augmented Generation (RAG) system designed for open-domain question answering over user-provided documents. The system combines dense vector retrieval with a large language model (LLM) backend to produce accurate, source-grounded answers from heterogeneous document collections (PDF, DOCX, TXT). DocuMind is containerised via Docker and deployable on commodity cloud platforms with no specialised infrastructure. Empirical evaluation on a representative document corpus demonstrates that source attribution is preserved end-to-end, reducing hallucination risk inherent to purely generative approaches. The full codebase is available as a private repository; this document serves as a technical reference and reproducibility guide.

---

## 1. Introduction

Large language models excel at language understanding but are fundamentally limited by their training cutoff and the absence of grounding in private or domain-specific corpora. Retrieval-Augmented Generation (RAG) [Lewis et al., 2020] addresses this limitation by conditioning generation on passages retrieved at inference time, enabling the model to answer questions about documents it has never seen during training.

DocuMind instantiates this paradigm as a practical, self-contained application. Given a user query *q* and a document collection *D*, the system retrieves the *k* most semantically relevant passages from *D* and conditions a generative model to produce an answer *a* with explicit provenance. The primary contributions of this work are:

1. An end-to-end RAG pipeline integrating open-source embedding models with a hosted LLM inference API.
2. A modular service architecture that allows independent substitution of the vector store (ChromaDB / Pinecone) and embedding backend (sentence-transformers / Jina AI).
3. A lightweight web interface exposing document upload, conversational Q&A, and real-time system statistics.
4. Full containerisation and one-command deployment on Railway and Render.

---

## 2. System Architecture

### 2.1 Overview

```
┌─────────────────────┐         REST / HTTP         ┌──────────────────────────────┐
│   Streamlit UI      │ ◄──────────────────────────► │   FastAPI Backend            │
│   (frontend/)       │                              │                              │
└─────────────────────┘                              │  ┌────────────────────────┐  │
                                                     │  │  Document Processor    │  │
                                                     │  │  PDF · DOCX · TXT      │  │
                                                     │  └───────────┬────────────┘  │
                                                     │              │ chunks        │
                                                     │  ┌───────────▼────────────┐  │
                                                     │  │  Embedding Service     │  │
                                                     │  │  sentence-transformers │  │
                                                     │  └───────────┬────────────┘  │
                                                     │              │ vectors       │
                                                     │  ┌───────────▼────────────┐  │
                                                     │  │  Vector Store          │  │
                                                     │  │  ChromaDB / Pinecone   │  │
                                                     │  └───────────┬────────────┘  │
                                                     │              │ top-k chunks  │
                                                     │  ┌───────────▼────────────┐  │
                                                     │  │  LLM Service           │  │
                                                     │  │  Groq · llama-3.1-8b   │  │
                                                     │  └────────────────────────┘  │
                                                     └──────────────────────────────┘
```

### 2.2 Component Stack

| Layer | Technology | Notes |
|---|---|---|
| Frontend | Streamlit | Single-page UI; HTTP client via `requests` |
| API server | FastAPI (Python 3.11) | Async; OpenAPI spec auto-generated at `/docs` |
| Document parsing | `pdfplumber`, `python-docx` | Format-agnostic ingestion pipeline |
| Embeddings | `sentence-transformers` (`paraphrase-MiniLM-L3-v2`) | 384-dim dense vectors; Jina AI optional |
| Vector store | ChromaDB (default) · Pinecone (optional) | Persistent; cosine similarity search |
| LLM inference | Groq API (`llama-3.1-8b-instant`) | Sub-second latency; interchangeable |
| Containerisation | Docker · Docker Compose | Multi-service; production-ready images |
| Deployment | Railway · Render | PaaS; environment-variable configuration |

---

## 3. Methodology

### 3.1 Document Ingestion

Uploaded files are parsed into raw text by format-specific extractors. The text is then split into fixed-length chunks with configurable overlap:

```
chunk_size    = 512 tokens  (default)
chunk_overlap = 50  tokens  (default)
```

Overlapping windows preserve cross-sentence context at chunk boundaries, mitigating the retrieval penalty of hard splits [Zhu et al., 2023].

### 3.2 Embedding and Indexing

Each chunk *cᵢ* is encoded by the embedding model *φ* into a dense vector **vᵢ** ∈ ℝ³⁸⁴:

```
vᵢ = φ(cᵢ)
```

Vectors are stored in ChromaDB with the original text and document metadata as payload. Indexing is performed at upload time; subsequent queries incur no re-computation cost.

### 3.3 Retrieval

At query time, the user query *q* is encoded by the same model φ to produce **v_q**. The *k* nearest chunks are retrieved via cosine similarity:

```
top-k = argtop-k { cos(v_q, vᵢ) }   k = 5 (default)
```

### 3.4 Generation

Retrieved chunks are concatenated into a structured prompt:

```
System: You are a document assistant. Answer using only the provided context.
        Always cite the source passage.
Context: [chunk_1] ... [chunk_k]
Question: {user_query}
Answer:
```

The prompt is forwarded to the Groq inference API. The response, together with the raw source passages, is returned to the frontend.

---

## 4. Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── main.py                      # FastAPI endpoints (upload, query, stats)
│   │   ├── config.py                    # Pydantic settings; env-var binding
│   │   ├── models.py                    # Request / response schemas
│   │   └── services/
│   │       ├── document_processor.py    # Format-agnostic text extraction
│   │       ├── embeddings.py            # Embedding model wrapper
│   │       ├── vector_store.py          # ChromaDB interface
│   │       ├── vector_store_pinecone.py # Pinecone alternative
│   │       └── llm.py                   # LLM abstraction layer
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── app.py                           # Streamlit application
│   ├── Dockerfile
│   └── requirements.txt
└── docker-compose.yml
```

---

## 5. Reproducibility

### 5.1 Prerequisites

- Docker ≥ 24.0 **or** Python 3.11
- Groq API key ([free tier](https://console.groq.com))

### 5.2 Local Deployment

```bash
# 1. Clone the repository (private)
git clone <repository-url>
cd projectRag

# 2. Configure environment
echo "GROQ_API_KEY=<your_key>" > backend/.env

# 3. Build and run
docker compose up --build
```

| Service | URL |
|---|---|
| Frontend | http://localhost:8501 |
| Backend API | http://localhost:8000/docs |

### 5.3 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | **Required.** Groq inference API key |
| `LLM_MODEL` | `llama-3.1-8b-instant` | Model identifier |
| `EMBEDDING_MODEL` | `paraphrase-MiniLM-L3-v2` | Sentence-transformers model |
| `VECTOR_STORE_TYPE` | `chromadb` | `chromadb` or `pinecone` |
| `CHUNK_SIZE` | `512` | Tokens per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between consecutive chunks |
| `MAX_FILE_SIZE_MB` | `10` | Upload size limit |

### 5.4 Cloud Deployment

Detailed guides are provided for both supported platforms:

- **Railway** (recommended — persistent services, internal networking): [`DEPLOY_RAILWAY.md`](DEPLOY_RAILWAY.md)
- **Render**: create two Docker Web Services pointing to `backend/` and `frontend/` respectively; set `GROQ_API_KEY` on the backend and `BACKEND_URL` on the frontend.

---

## 6. API Reference

The backend exposes a RESTful API documented via OpenAPI (Swagger UI at `/docs`).

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/upload` | Ingest a document file |
| `POST` | `/query` | Submit a natural language question |
| `GET` | `/stats` | Return indexing statistics |

---

## 7. Limitations and Future Work

The current implementation has the following known limitations:

- **Single-user, stateless sessions** — no authentication or conversation persistence across sessions.
- **Fixed chunk granularity** — dynamic chunking strategies (e.g., semantic segmentation) may improve retrieval precision.
- **English-centric embeddings** — multilingual documents may require a dedicated multilingual encoder.

Planned extensions include:

- [ ] User authentication and per-user document namespaces
- [ ] Persistent chat history and session management
- [ ] Support for Markdown, HTML, and EPUB formats
- [ ] Evaluation harness (RAGAS metrics: faithfulness, answer relevancy, context recall)
- [ ] Rate limiting and usage analytics

---

## 8. References

- Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020.
- Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP 2019.
- Zhu, Y. et al. (2023). *Large Language Models for Information Retrieval: A Survey*. arXiv:2308.07107.

---

## License

MIT — see [`LICENSE`](LICENSE) for details.
