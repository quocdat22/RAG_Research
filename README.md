# 🔬 RAG Native - Research Assistant System

<div align="center">

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

**A powerful RAG (Retrieval-Augmented Generation) system designed for researchers to intelligently search and query their document collections.**

[Features](#-features) • [Demo](#-demo) • [Architecture](#-architecture) • [API Documentation](#-api-documentation) • [Deployment](#-deployment)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-features)
- [Tech Stack](#-tech-stack)
- [Demo](#-demo)
- [Prerequisites](#-prerequisites)
- [Configuration](#-configuration)
- [API Documentation](#-api-documentation)
- [Architecture](#-architecture)
- [Deployment](#-deployment)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

**RAG Native** is an advanced research assistant system that leverages state-of-the-art AI technologies to help researchers efficiently search, retrieve, and query information from their document collections (scientific papers, books, reports, etc.). 

Built with modern Python technologies and AI models, it combines the power of vector embeddings, keyword search, and large language models to provide accurate, context-aware answers with proper source citations.

### Why RAG Native?

- 📚 **Intelligent Document Processing**: Automatically extracts and chunks documents while preserving context
- 🔍 **Hybrid Search**: Combines semantic understanding (vector embeddings) with traditional keyword search (BM25)
- 🎯 **Accurate Retrieval**: Uses Reciprocal Rank Fusion (RRF) and optional re-ranking for optimal results
- 💬 **Conversational Interface**: Natural Q&A interface with streaming responses
- 📖 **Source Citations**: Automatic citation tracking with filename and page numbers
- 🚀 **Production Ready**: Built with FastAPI, includes logging, error handling, and deployment configs

---

## ✨ Features

### Document Processing
- ✅ **Multi-format Support**: PDF, DOCX, and TXT files
- ✅ **Smart Chunking**: Token-based chunking (500-1000 tokens) with configurable overlap
- ✅ **Metadata Extraction**: Automatic extraction of filename, page numbers, and timestamps
- ✅ **LlamaParse Integration**: Optional advanced PDF parsing with LlamaParse

### Retrieval & Search
- ✅ **Vector Search**: Semantic search using OpenAI embeddings (text-embedding-3-small)
- ✅ **Keyword Search**: BM25 algorithm for traditional keyword matching
- ✅ **Hybrid Retrieval**: Combines vector and keyword search with RRF (Reciprocal Rank Fusion)
- ✅ **Re-ranking**: Optional Cohere reranker for improved relevance
- ✅ **Configurable Parameters**: Adjustable weights, top-k results, and search strategies

### Generation & Chat
- ✅ **Multiple LLM Options**: GPT-4o-mini (default), GPT-4o (heavy), GPT-4.1-mini (light)
- ✅ **RAG-Optimized Prompts**: Specialized prompt templates for accurate responses
- ✅ **Citation-Aware**: Responses include source attributions with page numbers
- ✅ **Streaming Support**: Real-time response streaming for better UX
- ✅ **Conversation Management**: Persistent conversation history with Supabase integration

### API & Backend
- ✅ **FastAPI Framework**: High-performance async REST API
- ✅ **Comprehensive Endpoints**: Document management, search, chat, conversation history
- ✅ **Health Monitoring**: Health check endpoints for production monitoring
- ✅ **CORS Support**: Configurable CORS for frontend integration
- ✅ **Structured Logging**: Comprehensive logging to files and console
- ✅ **Error Handling**: Graceful error handling with detailed messages

### Frontend
- ✅ **Streamlit UI**: Modern, responsive web interface
- ✅ **Chat Interface**: Interactive Q&A with message history
- ✅ **Document Library**: Upload and manage document collections
- ✅ **Search Dashboard**: Advanced search with configurable parameters
- ✅ **Source Display**: View retrieved chunks and citations
- ✅ **Conversation History**: Browse and resume past conversations

---

## 🛠️ Tech Stack

### Backend
- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) - High-performance async web framework
- **Language**: Python 3.11+
- **Package Manager**: [UV](https://github.com/astral-sh/uv) - Fast Python package installer

### AI & ML
- **Embeddings**: [OpenAI text-embedding-3-small](https://platform.openai.com/docs/models/embeddings) (1536 dimensions)
- **LLM**: [OpenAI GPT-4o-mini/GPT-4o](https://platform.openai.com/docs/models/gpt-4) - Chat completion models
- **Re-ranking**: [Cohere Rerank](https://cohere.com/rerank) (optional)
- **Document Parsing**: [LlamaParse](https://github.com/run-llama/llama_parse) (optional)

### Storage & Retrieval
- **Vector Database**: [ChromaDB](https://www.trychroma.com/) - Embedded vector store
- **Keyword Search**: [Rank-BM25](https://github.com/dorianbrown/rank_bm25) - BM25 algorithm implementation
- **Metadata Storage**: [Supabase](https://supabase.com/) - PostgreSQL for conversation history

### Document Processing
- **PDF**: [PyPDF](https://pypdf.readthedocs.io/), [PyMuPDF](https://pymupdf.readthedocs.io/)
- **DOCX**: [python-docx](https://python-docx.readthedocs.io/)
- **Tokenization**: [tiktoken](https://github.com/openai/tiktoken) - OpenAI's tokenizer

### Frontend
- **UI Framework**: [Streamlit](https://streamlit.io/) - Interactive web apps for ML/data science
- **HTTP Client**: [Requests](https://requests.readthedocs.io/)

### DevOps & Deployment
- **ASGI Server**: [Uvicorn](https://www.uvicorn.org/) (development), [Gunicorn](https://gunicorn.org/) (production)
- **Environment**: [python-dotenv](https://github.com/theskumar/python-dotenv) - Environment variable management
- **Testing**: [pytest](https://pytest.org/), [httpx](https://www.python-httpx.org/)

---

## 🎬 Demo

### Chat Interface
Ask questions and get answers with source citations:

```
Q: What are the main findings of the paper?
A: Based on the provided documents, the main findings include...
   
   Sources:
   📄 research_paper.pdf (Page 3)
   📄 research_paper.pdf (Page 7)
```

### Search Dashboard
Perform advanced searches with configurable parameters:
- Vector search (semantic)
- Keyword search (BM25)
- Hybrid search (combined)
- Adjustable top-k results
- Weight configuration

### Document Library
Upload and manage documents:
- Drag-and-drop file upload
- List all documents with metadata
- Delete documents from collection
- View document statistics

---

## 📦 Prerequisites

Before installation, ensure you have:

- **Python 3.11 or higher**
- **UV package manager** ([Installation guide](https://github.com/astral-sh/uv#installation))
- **OpenAI API key** or **GitHub Token** (for LLM access)
- **Cohere API key** (optional, for re-ranking)
- **LlamaCloud API key** (optional, for advanced PDF parsing)

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/RAG_Native.git
cd RAG_Native
```

### 2. Install UV Package Manager

If you haven't installed UV yet:

```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via pip
pip install uv
```

### 3. Create Virtual Environment and Install Dependencies

```bash
# Create virtual environment
uv venv

# Activate virtual environment
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# Install dependencies
uv pip install -e .
```

### 4. Install Development Dependencies (Optional)

```bash
uv pip install -e ".[dev]"
```

---

## ⚙️ Configuration

### 1. Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

### 2. Configure API Keys

Edit `.env` and add your API keys:

```bash
# Required: OpenAI API Key (or GitHub Token as alternative)
OPENAI_API_KEY=sk-proj-...
# GITHUB_TOKEN=ghp_...  # Alternative to OpenAI API

# Optional: Cohere API Key (for re-ranking)
COHERE_API_KEY=your-cohere-api-key

# Optional: LlamaCloud API Key (for advanced PDF parsing)
LLAMA_CLOUD_API_KEY=llx-...

# Optional: Supabase Configuration (for conversation history)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-anon-key

# Environment
ENVIRONMENT=development

# CORS (for production)
ALLOWED_ORIGINS=http://localhost:8501,https://your-frontend-url.com
```

### 3. Advanced Configuration

You can customize various settings via environment variables:

#### Embedding Settings
```bash
EMBEDDING_MODEL=openai/text-embedding-3-small
EMBEDDING_DIMENSION=1536
```

#### LLM Settings
```bash
LLM_MODEL=openai/gpt-4o-mini           # Default model
LLM_MODEL_LIGHT=openai/gpt-4.1-mini    # Light/fast model
LLM_MODEL_HEAVY=openai/gpt-4o          # Heavy/powerful model
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=2048
```

#### Chunking Settings
```bash
CHUNK_SIZE=800        # Chunk size in tokens (500-1000)
CHUNK_OVERLAP=200     # Overlap between chunks (0-500)
```

#### Retrieval Settings
```bash
RETRIEVAL_TOP_K=5                # Number of chunks to retrieve
RETRIEVAL_VECTOR_WEIGHT=0.5      # Vector search weight (0.0-1.0)
RETRIEVAL_KEYWORD_WEIGHT=0.5     # Keyword search weight (0.0-1.0)
RETRIEVAL_USE_RERANKING=false    # Enable/disable Cohere reranking
```

#### Storage Paths
```bash
DOCUMENTS_DIR=data/documents     # Document storage directory
CHROMA_DIR=data/chroma_db        # ChromaDB storage directory
LOG_DIR=logs                     # Log files directory
```

See [config/settings.py](config/settings.py) for all available configuration options.

---

## 📚 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-12-26T10:30:00",
  "chroma_status": "connected",
  "version": "0.1.0"
}
```

#### Upload Document
```http
POST /documents/upload
Content-Type: multipart/form-data
```

**Parameters:**
- `file` (file): Document file (PDF, DOCX, TXT)

**Response:**
```json
{
  "filename": "research_paper.pdf",
  "num_chunks": 24,
  "message": "Document uploaded successfully"
}
```

#### List Documents
```http
GET /documents
```

**Response:**
```json
{
  "documents": [
    {
      "filename": "research_paper.pdf",
      "chunk_count": 24,
      "upload_time": "2025-12-26T10:00:00"
    }
  ],
  "total": 1
}
```

#### Delete Document
```http
DELETE /documents/{filename}
```

#### Search
```http
POST /search
Content-Type: application/json
```

**Request Body:**
```json
{
  "query": "machine learning applications",
  "method": "hybrid",
  "top_k": 5,
  "vector_weight": 0.6,
  "keyword_weight": 0.4
}
```

**Response:**
```json
{
  "results": [
    {
      "content": "Machine learning has various applications...",
      "metadata": {
        "filename": "ml_paper.pdf",
        "page": 3,
        "chunk_index": 5
      },
      "score": 0.87
    }
  ]
}
```

#### Chat
```http
POST /chat
Content-Type: application/json
```

**Request Body:**
```json
{
  "message": "What is deep learning?",
  "conversation_id": "uuid-here",
  "stream": false
}
```

**Response:**
```json
{
  "answer": "Deep learning is a subset of machine learning...",
  "sources": [
    {
      "filename": "dl_book.pdf",
      "page": 12,
      "content": "Deep learning architectures..."
    }
  ],
  "conversation_id": "uuid-here"
}
```

#### Stream Chat
```http
POST /chat/stream
Content-Type: application/json
```

Streams response as Server-Sent Events (SSE).

For complete API documentation, visit: **http://localhost:8000/docs** (when running)

---

## 🏗️ Architecture

### System Overview

![System Architecture](/assets/system_architecture.png)

### RAG Pipeline

![RAG Pipeline](/assets/rag_pipeline.png)

### Key Components

1. **Document Ingestion**
   - Extracts text from PDF/DOCX/TXT
   - Chunks documents into manageable sizes
   - Generates embeddings
   - Stores in vector database

2. **Hybrid Retrieval**
   - **Vector Search**: Semantic similarity using cosine distance
   - **BM25 Search**: Keyword-based ranking
   - **RRF Fusion**: Combines both methods for optimal results
   - **Re-ranking**: Optional Cohere reranker for refinement

3. **RAG Generation**
   - Retrieves relevant context
   - Constructs prompt with context and query
   - Generates answer using LLM
   - Includes source citations

4. **Conversation Management**
   - Persists conversation history in Supabase
   - Maintains context across turns
   - Supports conversation resume and delete

---

## 🚀 Deployment

This project supports deployment to cloud platforms:

### Render (Backend)

1. **Create a new Web Service** on [Render](https://render.com)
2. **Configure Build Command:**
   ```bash
   uv sync --frozen --no-dev && uv cache prune --ci
   ```
3. **Configure Start Command:**
   ```bash
   uv run gunicorn src.api.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 120
   ```
4. **Add Environment Variables** in Render dashboard (see [DEPLOY.md](DEPLOY.md))

### Streamlit Cloud (Frontend)

1. **Deploy to [Streamlit Cloud](https://streamlit.io/cloud)**
2. **Set Main file:** `ui/app.py`
3. **Add Secrets** in Streamlit dashboard:
   ```toml
   API_BASE_URL = "https://your-backend.onrender.com"
   ```

For detailed deployment instructions, see [DEPLOY.md](DEPLOY.md).

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Install dev dependencies (`uv pip install -e ".[dev]"`)
4. Make your changes
5. Run tests (`uv run pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [LangChain](https://www.langchain.com/) for RAG inspiration
- [OpenAI](https://openai.com/) for embeddings and LLM APIs
- [ChromaDB](https://www.trychroma.com/) for the excellent vector database
- [FastAPI](https://fastapi.tiangolo.com/) for the amazing web framework
- [Streamlit](https://streamlit.io/) for the intuitive UI framework

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

<div align="center">

**Built with ❤️ for researchers by researchers**

⭐ Star this repo if you find it useful!

</div>
