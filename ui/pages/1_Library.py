import streamlit as st
import requests
import re
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
    .search-box {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
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

def search_documents(query=None, authors=None, year_min=None, year_max=None, keywords=None):
    try:
        payload = {}
        if query:
            payload["query"] = query
        if authors:
            payload["authors"] = authors
        if year_min:
            payload["year_min"] = year_min
        if year_max:
            payload["year_max"] = year_max
        if keywords:
            payload["keywords"] = keywords
        
        response = requests.post(f"{API_BASE_URL}/documents/search", json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error searching documents: {e}")
        return {"documents": [], "total": 0}

def get_document_chunks(doc_id):
    try:
        response = requests.get(f"{API_BASE_URL}/documents/{doc_id}/chunks")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching chunks: {e}")
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
    
    # Create form for editing
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
            # Prepare update payload
            updates = {}
            if authors: updates["authors"] = authors
            if year: updates["year"] = year
            if doi: updates["doi"] = doi
            if arxiv_id: updates["arxiv_id"] = arxiv_id
            if venue: updates["venue"] = venue
            if keywords: updates["keywords"] = keywords
            if abstract: updates["abstract"] = abstract
            
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
    """Display document metadata in a nice format"""
    st.markdown('<div class="metadata-section">', unsafe_allow_html=True)
    
    if doc.get('authors'):
        st.markdown(f"**Authors:** {doc['authors']}")
    
    row1 = st.columns([1, 1, 2])
    with row1[0]:
        if doc.get('year'):
            st.markdown(f"**Year:** {doc['year']}")
    with row1[1]:
        if doc.get('doi'):
            st.markdown(f"**DOI:** {doc['doi']}")
    with row1[2]:
        if doc.get('venue'):
            st.markdown(f"**Venue:** {doc['venue']}")
    
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
    st.markdown("Manage your documents with smart search and rich metadata.")
    
    # Smart Search Section
    st.markdown('<div class="search-box">', unsafe_allow_html=True)
    st.markdown("### 🔍 Smart Search")
    
    col_search1, col_search2 = st.columns([3, 1])
    with col_search1:
        search_query = st.text_input("Search documents", placeholder="Search by filename, authors, keywords, abstract, venue...")
    with col_search2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_button = st.button("🔎 Search", use_container_width=True, type="primary")
    
    # Advanced filters in expander
    with st.expander("⚙️ Advanced Filters"):
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            filter_authors = st.text_input("Filter by Authors")
        with col_f2:
            year_range = st.slider("Year Range", 1990, 2030, (2000, 2030))
        with col_f3:
            filter_keywords = st.text_input("Filter by Keywords")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Refresh button
    if st.button("🔄 Refresh Documents"):
        st.rerun()
    
    # Get documents based on search/filters
    if search_button or search_query:
        year_min, year_max = year_range
        docs_data = search_documents(
            query=search_query if search_query else None,
            authors=filter_authors if filter_authors else None,
            year_min=year_min,
            year_max=year_max,
            keywords=filter_keywords if filter_keywords else None
        )
        if search_query or filter_authors or filter_keywords:
            st.info(f"Found {docs_data['total']} matching documents")
    else:
        docs_data = get_documents()
    
    docs = docs_data.get("documents", [])
    
    if not docs:
        st.info("No documents found. Please upload documents in the Chat page or adjust your search filters.")
        return

    # Document selection
    doc_options = {f"{doc['filename']} ({doc['document_id'][:8]})": doc for doc in docs}
    selected_doc_name = st.selectbox("Select a document to inspect:", options=list(doc_options.keys()))
    selected_doc = doc_options[selected_doc_name]

    # Document Info and Actions
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
    
    # Display rich metadata
    st.subheader("📋 Document Metadata")
    display_metadata(selected_doc)
    
    st.markdown("---")
    
    # Load and display chunks
    with st.spinner("Loading chunks..."):
        chunks_data = get_document_chunks(selected_doc['document_id'])
    
    if chunks_data and chunks_data.get("chunks"):
        chunks = chunks_data["chunks"]
        st.subheader(f"📄 Content Chunks ({len(chunks)})")
        
        # Grid layout
        cols_per_row = 3
        for i in range(0, len(chunks), cols_per_row):
            row_chunks = chunks[i:i + cols_per_row]
            cols = st.columns(cols_per_row)
            
            for j, chunk in enumerate(row_chunks):
                with cols[j]:
                    st.markdown(f"""
                    <div class="chunk-header">
                        <span>Chunk {i+j+1}</span>
                        <span>Page {chunk['metadata'].get('page', 'N/A')}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Truncated preview
                    preview_text = chunk['text'][:500] + "..." if len(chunk['text']) > 500 else chunk['text']
                    st.markdown(f'<div class="chunk-content">{preview_text}</div>', unsafe_allow_html=True)
                    
                    chunk_key = f"chunk_{chunk['chunk_id']}"
                    if st.button("🔍 Zoom In", key=chunk_key):
                        show_chunk_detail(chunk)
                    
                    st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
    else:
        st.warning("No chunks found for this document.")

if __name__ == "__main__":
    main()
