# Deployment Guide

## Backend (Render)

### Cấu hình Render Web Service

1. **Root Directory**: `.` (root folder)
2. **Build Command**: 
   ```bash
   uv sync --frozen --no-dev && uv cache prune --ci
   ```
3. **Start Command**:
   ```bash
   uv run gunicorn src.api.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 120
   ```

### Environment Variables (Set trên Render Dashboard)

**Required:**
- `OPENAI_API_KEY` hoặc `GITHUB_TOKEN` (ít nhất 1 trong 2)
- `COHERE_API_KEY` (nếu dùng reranking)
- `LLAMA_CLOUD_API_KEY` (nếu dùng LlamaParse)

**Production Config:**
- `ENVIRONMENT=production`
- `ALLOWED_ORIGINS=https://your-frontend-url.streamlit.app` (thay bằng URL Streamlit thực tế, có thể nhiều domain cách nhau bằng dấu phẩy)

**Storage Paths (nếu dùng Persistent Disk):**
- `DOCUMENTS_DIR=/var/data/documents`
- `CHROMA_DIR=/var/data/chroma_db`
- `LOG_DIR=/var/data/logs`

### Persistent Disk (Optional)

Nếu muốn lưu data giữa các lần deploy:
- Tạo Disk với name: `rag-data`
- Mount path: `/var/data`
- Size: 10GB

---

## Frontend (Streamlit Cloud)

### Cấu hình

1. **Main file path**: `ui/app.py`
2. **Python version**: 3.11

### Secrets (Streamlit Cloud)

Vào Settings > Secrets và thêm:

```toml
API_BASE_URL = "https://your-backend-url.onrender.com"
```

### Files cần sửa

Trong `ui/app.py`, `ui/pages/1_Library.py`, `ui/pages/2_Search.py`, thay:

```python
API_BASE_URL = "http://localhost:8000"
```

Bằng:

```python
import os
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
```

---

## Test Local

### Windows (Gunicorn không hỗ trợ Windows)

```bash
# Chạy với uvicorn
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Hoặc chạy trực tiếp
uv run python src/api/main.py
```

### Linux/Mac

```bash
# Test với gunicorn (giống production)
uv run gunicorn src.api.main:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Hoặc dùng uvicorn
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Truy cập: http://localhost:8000/health

---

## Checklist trước khi Deploy

### Backend:
- [x] Gunicorn đã có trong dependencies
- [x] CORS config dùng environment variable
- [x] Uvicorn reload chỉ bật trong development
- [x] Settings hỗ trợ ALLOWED_ORIGINS
- [x] Set tất cả environment variables trên Render
- [x] (Optional) Setup Persistent Disk - supabase

### Frontend:
- [x] Sửa BACKEND_API_URL trong 3 files UI
- [x] Tạo ui/requirements.txt
- [x] Set BACKEND_API_URL secret trên Streamlit Cloud

### Post-Deploy:
- [x] Test health check: `https://your-api.onrender.com/health`
- [x] Cập nhật ALLOWED_ORIGINS với URL frontend thực tế
- [x] Upload documents qua UI
- [x] Test chat functionality
