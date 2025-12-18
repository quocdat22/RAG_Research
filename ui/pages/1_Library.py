import streamlit as st
import requests
import re
import fitz  # PyMuPDF
import io
from PIL import Image
from datetime import datetime

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="Document Library - RAG Native",
    page_icon="📚",
    layout="wide"
)

# Custom CSS for the "Zoom" effect, grid, and metadata cards
st.markdown("""
<style>
    .chunk-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        height: 200px;
        overflow: hidden;
        position: relative;
        background-color: white;
        transition: transform 0.2s, box-shadow 0.2s;
        cursor: pointer;
    }
    .chunk-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-color: #1E88E5;
    }
    .chunk-header {
        font-weight: bold;
        font-size: 0.8rem;
        color: #666;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
    }
    .chunk-content {
        font-size: 0.9rem;
        line-height: 1.4;
        color: #333;
        display: -webkit-box;
        -webkit-line-clamp: 6;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .metadata-badge {
        display: inline-block;
        background-color: #E3F2FD;
        color: #1976D2;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        margin: 4px;
        font-weight: 500;
    }
    .metadata-section {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .metadata-label {
        font-weight: 600;
        color: #666;
        font-size: 0.85rem;
        margin-bottom: 4px;
    }
    .metadata-value {
        color: #333;
        margin-bottom: 10px;
    }
    .upload-container {
        background-color: #f1f8ff;
        padding: 20px;
        border-radius: 10px;
        border: 1px dashed #1E88E5;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

def format_latex(text: str) -> str:
    if not text:
        return text
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)
    return text

def get_documents():
    try:
        response = requests.get(f"{API_BASE_URL}/documents")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching documents: {e}")
        return {"documents": [], "total": 0}

def get_document_chunks(doc_id):
    try:
        response = requests.get(f"{API_BASE_URL}/documents/{doc_id}/chunks")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching chunks: {e}")
        return None

def upload_document(file):
    files = {"file": (file.name, file.getvalue(), file.type)}
    try:
        response = requests.post(f"{API_BASE_URL}/documents/upload", files=files)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error uploading document: {e}")
        return None

def update_document_metadata(doc_id, metadata):
    try:
        response = requests.put(
            f"{API_BASE_URL}/documents/{doc_id}/metadata",
            json=metadata
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error updating metadata: {e}")
        return None

def delete_document(doc_id):
    try:
        response = requests.delete(f"{API_BASE_URL}/documents/{doc_id}")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error deleting document: {e}")
        return None

def get_pdf_preview(file_bytes, max_pages=5):
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        images = []
        num_pages = min(len(doc), max_pages)
        for i in range(num_pages):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=fitz.Matrix(1, 1))
            img_data = pix.tobytes("png")
            images.append(Image.open(io.BytesIO(img_data)))
        doc.close()
        return images
    except Exception as e:
        st.error(f"Error generating PDF preview: {e}")
        return []

@st.dialog("Chunk Detail", width="large")
def show_chunk_detail(chunk):
    st.markdown(f"### Chunk ID: `{chunk['chunk_id']}`")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Page", chunk['metadata'].get('page', 'N/A'))
    with col2:
        st.metric("Tokens", chunk['metadata'].get('token_count', 'N/A'))
    with col3:
        st.metric("Source", chunk['metadata'].get('filename', 'Unknown'))
    st.markdown("---")
    st.markdown("#### Content")
    st.markdown(format_latex(chunk['text']))
    st.markdown("---")
    st.json(chunk['metadata'])

@st.dialog("Edit Document Metadata", width="large")
def show_metadata_editor(doc):
    st.markdown(f"### Edit Metadata: {doc['filename']}")
    with st.form("metadata_form"):
        col1, col2 = st.columns(2)
        with col1:
            authors = st.text_input("Authors (comma-separated)", value=doc.get('authors') or "")
            year = st.text_input("Year", value=doc.get('year') or "")
            doi = st.text_input("DOI", value=doc.get('doi') or "")
            arxiv_id = st.text_input("arXiv ID", value=doc.get('arxiv_id') or "")
        with col2:
            venue = st.text_input("Venue (Conference/Journal)", value=doc.get('venue') or "")
            keywords = st.text_area("Keywords (comma-separated)", value=doc.get('keywords') or "", height=100)
        abstract = st.text_area("Abstract", value=doc.get('abstract') or "", height=150)
        col_save, col_cancel = st.columns([1, 4])
        with col_save:
            submitted = st.form_submit_button("💾 Save", use_container_width=True, type="primary")
        with col_cancel:
            cancel = st.form_submit_button("Cancel", use_container_width=True)
        if submitted:
            updates = {
                "authors": authors, "year": year, "doi": doi,
                "arxiv_id": arxiv_id, "venue": venue,
                "keywords": keywords, "abstract": abstract
            }
            result = update_document_metadata(doc['document_id'], updates)
            if result:
                st.success("Metadata updated successfully!")
                st.rerun()
        if cancel:
            st.rerun()

@st.dialog("⚠️ Confirm Delete", width="medium")
def show_delete_confirmation(doc):
    st.warning(f"Are you sure you want to delete **{doc['filename']}**?")
    st.markdown("This action cannot be undone. All chunks and metadata will be permanently deleted.")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🗑️ Yes, Delete", use_container_width=True, type="primary"):
            result = delete_document(doc['document_id'])
            if result:
                st.success(f"Deleted {doc['filename']}!")
                st.rerun()
    with col2:
        if st.button("Cancel", use_container_width=True):
            st.rerun()

def display_metadata(doc):
    st.markdown('<div class="metadata-section">', unsafe_allow_html=True)
    if doc.get('authors'):
        st.markdown(f"**Authors:** {doc['authors']}")
    row1 = st.columns([1, 1, 2])
    with row1[0]:
        if doc.get('year'): st.markdown(f"**Year:** {doc['year']}")
    with row1[1]:
        if doc.get('doi'): st.markdown(f"**DOI:** {doc['doi']}")
    with row1[2]:
        if doc.get('venue'): st.markdown(f"**Venue:** {doc['venue']}")
    if doc.get('keywords'):
        st.markdown("**Keywords:**")
        keywords = [k.strip() for k in doc['keywords'].split(',')]
        keyword_html = "".join([f'<span class="metadata-badge">{k}</span>' for k in keywords])
        st.markdown(keyword_html, unsafe_allow_html=True)
    if doc.get('abstract'):
        with st.expander("📄 Abstract"):
            st.markdown(doc['abstract'])
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    st.title("📚 Document Library")
    st.markdown("Upload and manage your research documents.")

    # 1. Upload Section
    if "lib_uploader_key" not in st.session_state:
        st.session_state.lib_uploader_key = 0

    with st.expander("📤 Upload New Document", expanded=False):
        st.markdown('<div class="upload-container">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose a file", type=["pdf", "docx", "txt"],
            help="Upload PDF, DOCX, or TXT files", key=f"lib_uploader_{st.session_state.lib_uploader_key}"
        )
        if uploaded_file:
            if uploaded_file.type == "application/pdf":
                st.markdown("#### Preview")
                preview_images = get_pdf_preview(uploaded_file.getvalue())
                if preview_images:
                    cols = st.columns(3)
                    for i, img in enumerate(preview_images[:3]):
                        with cols[i]: st.image(img, use_container_width=True, caption=f"Page {i+1}")
            
            if st.button("🚀 Process & Index Document", type="primary", use_container_width=True):
                with st.spinner("Processing document..."):
                    result = upload_document(uploaded_file)
                    if result:
                        st.success(f"Successfully indexed: {result['filename']}")
                        st.session_state.lib_uploader_key += 1
                        st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # 2. Document Selection & Management
    docs_data = get_documents()
    docs = docs_data.get("documents", [])
    
    if not docs:
        st.info("No documents in library. Use the upload section above to get started.")
        return

    st.subheader(f"🗂️ Manage Documents ({len(docs)})")
    doc_options = {f"{doc['filename']} ({doc['document_id'][:8]})": doc for doc in docs}
    selected_doc_name = st.selectbox("Select a document to inspect:", options=list(doc_options.keys()))
    selected_doc = doc_options[selected_doc_name]

    # Quick Actions
    col_info, col_edit, col_del = st.columns([3, 1, 1])
    with col_info:
        st.write(f"**Type:** {selected_doc['file_type'].upper()} | **Uploaded:** {selected_doc['upload_timestamp']}")
    with col_edit:
        if st.button("✏️ Edit Metadata", use_container_width=True):
            show_metadata_editor(selected_doc)
    with col_del:
        if st.button("🗑️ Delete", use_container_width=True, type="secondary"):
            show_delete_confirmation(selected_doc)

    st.markdown("---")
    
    # 3. View Metadata & Chunks
    tab_meta, tab_chunks = st.tabs(["📋 Metadata", "📄 Chunks"])
    
    with tab_meta:
        display_metadata(selected_doc)
        
    with tab_chunks:
        with st.spinner("Loading chunks..."):
            chunks_data = get_document_chunks(selected_doc['document_id'])
        
        if chunks_data and chunks_data.get("chunks"):
            chunks = chunks_data["chunks"]
            st.subheader(f"Content Clusters ({len(chunks)})")
            
            cols_per_row = 3
            for i in range(0, len(chunks), cols_per_row):
                row_chunks = chunks[i:i + cols_per_row]
                cols = st.columns(cols_per_row)
                for j, chunk in enumerate(row_chunks):
                    with cols[j]:
                        st.markdown(f'<div class="chunk-header"><span>Chunk {i+j+1}</span><span>Pg {chunk["metadata"].get("page", "N/A")}</span></div>', unsafe_allow_html=True)
                        preview_text = chunk['text'][:400] + "..." if len(chunk['text']) > 400 else chunk['text']
                        st.markdown(f'<div class="chunk-content">{preview_text}</div>', unsafe_allow_html=True)
                        if st.button("🔍 Zoom", key=f"btn_{chunk['chunk_id']}"):
                            show_chunk_detail(chunk)
                        st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
        else:
            st.warning("No chunks found.")

if __name__ == "__main__":
    main()
