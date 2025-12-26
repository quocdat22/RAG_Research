# 🔬 RAG - Research Assistant System

<div align="center">

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

**A powerful RAG (Retrieval-Augmented Generation) system designed for researchers to intelligently search and query their document collections.**

</div>

---

## 🎯 Overview

**RAG Native** is an advanced research assistant system that leverages state-of-the-art AI technologies to help researchers efficiently search, retrieve, and query information from their document collections (scientific papers, books, reports, etc.). 

Built with FastAPI and modern Python technologies, it combines vector embeddings, keyword search (BM25), and large language models to provide accurate, context-aware answers with proper source citations.

**Key Capabilities:**
- 📚 Multi-format document processing (PDF, DOCX, TXT) with smart chunking
- 🔍 Hybrid search combining semantic understanding and keyword matching
- 🎯 Reciprocal Rank Fusion (RRF) and optional Cohere re-ranking
- 💬 Conversational Q&A interface with streaming responses
- 📖 Automatic source citations with filename and page numbers
- 🚀 Production-ready FastAPI backend + Streamlit UI

---

## ✨ Features

**Document Processing:** Multi-format support (PDF/DOCX/TXT) • Smart token-based chunking (500-1000 tokens) • Metadata extraction • Optional LlamaParse integration

**Retrieval:** Vector search (OpenAI embeddings) • BM25 keyword search • Hybrid retrieval with RRF • Optional Cohere reranking • Configurable weights and top-k

**Generation:** Multiple GPT models (4o-mini/4o/4.1-mini) • RAG-optimized prompts • Citation-aware responses • Streaming support • Conversation history

**API:** FastAPI async REST API • Document management endpoints • Search & chat APIs • Health monitoring • CORS support • Structured logging

**Frontend:** Streamlit web UI • Interactive chat interface • Document library management • Advanced search dashboard • Conversation history

---

## 🛠️ Tech Stack

**Core:** Python 3.11+ • FastAPI • Uvicorn/Gunicorn • UV package manager

**AI/ML:** OpenAI (embeddings & GPT models) • Cohere Rerank • LlamaParse

**Storage:** ChromaDB (vector store) • Rank-BM25 (keyword search) • Supabase (conversations)

**Document Processing:** PyPDF • PyMuPDF • python-docx • tiktoken

**Frontend:** Streamlit • Requests

---

## 📦 Prerequisites

- **Python 3.11+**
- **UV package manager** ([Install](https://github.com/astral-sh/uv#installation))
- **OpenAI API key** or **GitHub Token**
- **Cohere API key** (optional, for re-ranking)
- **LlamaCloud API key** (optional, for advanced PDF parsing)

---

## 🚀 Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/RAG_Native.git
cd RAG_Native
```

### 2. Install UV Package Manager

```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Setup Environment

```bash
# Create and activate virtual environment
uv venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# Install dependencies
uv pip install -e .
```

---

## ⚙️ Configuration

### 1. Create Environment File

```bash
cp .env.example .env
```

### 2. Configure API Keys

Edit `.env` and add your credentials:

```bash
# Required: OpenAI API Key (or GitHub Token)
OPENAI_API_KEY=sk-proj-...
# GITHUB_TOKEN=ghp_...  # Alternative

# Optional: Additional Services
COHERE_API_KEY=your-cohere-key
LLAMA_CLOUD_API_KEY=llx-...
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-key

# Environment
ENVIRONMENT=development
ALLOWED_ORIGINS=http://localhost:8501
```

### 3. Key Configuration Options

```bash
# Models
LLM_MODEL=openai/gpt-4o-mini
EMBEDDING_MODEL=openai/text-embedding-3-small

# Chunking
CHUNK_SIZE=800          # Tokens per chunk
CHUNK_OVERLAP=200       # Overlap between chunks

# Retrieval
RETRIEVAL_TOP_K=5              # Number of results
RETRIEVAL_VECTOR_WEIGHT=0.5    # Vector vs keyword balance
RETRIEVAL_USE_RERANKING=false  # Enable Cohere reranking
```

See [config/settings.py](config/settings.py) for all configuration options.

---

## 🏃 Running the Application

### Start Backend API

```bash
# Development
uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Production
uv run gunicorn src.api.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

API documentation: **http://localhost:8000/docs**

### Start Frontend UI

```bash
uv run streamlit run ui/app.py
```

Access UI: **http://localhost:8501**

---

## 📚 API Quick Reference

### Core Endpoints

```http
# Health Check
GET /health

# Upload Document
POST /documents/upload
Content-Type: multipart/form-data

# List Documents
GET /documents

# Delete Document
DELETE /documents/{filename}

# Search
POST /search
{
  "query": "your search query",
  "method": "hybrid",  # vector|keyword|hybrid
  "top_k": 5
}

# Chat
POST /chat
{
  "message": "your question",
  "conversation_id": "uuid",
  "stream": false
}

# Stream Chat
POST /chat/stream
```

**Full API documentation:** http://localhost:8000/docs (when running)

---

## 🏗️ Architecture

RAG Native uses a multi-stage pipeline:

1. **Ingestion:** Extract text → Chunk documents → Generate embeddings → Store in ChromaDB
2. **Retrieval:** Hybrid search (Vector + BM25) → RRF fusion → Optional reranking
3. **Generation:** Retrieve context → Construct prompt → Generate answer → Add citations
4. **Storage:** Conversation history in Supabase • Document metadata tracking

For detailed architecture, see [ARCHITECTURE.md](ARCHITECTURE.md) (if available).

---

## 🚀 Deployment

### Render (Backend)

```bash
# Build Command
uv sync --frozen --no-dev && uv cache prune --ci

# Start Command
uv run gunicorn src.api.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 120
```

### Streamlit Cloud (Frontend)

- Main file: `ui/app.py`
- Add secret: `API_BASE_URL=https://your-backend.onrender.com`

See [DEPLOY.md](DEPLOY.md) for detailed deployment instructions.

---

## 🤝 Contributing

Contributions welcome! Please fork the repo, create a feature branch, and submit a PR. Run tests with `uv run pytest`.

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ for researchers by researchers**

⭐ Star this repo if you find it useful!

</div>
