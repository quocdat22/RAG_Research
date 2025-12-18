import streamlit as st
import requests
import re
from datetime import datetime

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="Smart Search - RAG Native",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .search-container {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
        border: 1px solid #e9ecef;
    }
    .result-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.8rem;
        border-left: 5px solid #1E88E5;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .result-score {
        background-color: #E3F2FD;
        color: #1E88E5;
        padding: 2px 10px;
        border-radius: 10px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .source-info {
        font-size: 0.85rem;
        color: #666;
        margin-top: 10px;
    }
    .search-tabs {
        margin-bottom: 20px;
    }
    .document-card {
        border: 1px solid #eee;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        transition: transform 0.2s;
    }
    .document-card:hover {
        border-color: #1E88E5;
        transform: scale(1.01);
    }
</style>
""", unsafe_allow_html=True)

def format_latex(text: str) -> str:
    if not text:
        return text
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)
    return text

def semantic_search(query, top_k=5, search_type="hybrid"):
    try:
        response = requests.post(
            f"{API_BASE_URL}/search",
            json={
                "query": query,
                "top_k": top_k,
                "search_type": search_type
            }
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error in content search: {e}")
        return None

def document_search(query=None, authors=None, year_min=None, year_max=None, keywords=None):
    try:
        payload = {}
        if query: payload["query"] = query
        if authors: payload["authors"] = authors
        if year_min: payload["year_min"] = year_min
        if year_max: payload["year_max"] = year_max
        if keywords: payload["keywords"] = keywords
        
        response = requests.post(f"{API_BASE_URL}/documents/search", json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error in document search: {e}")
        return None

def main():
    st.title("🔍 Smart Search")
    st.markdown("Search for specific content chunks or discover documents by metadata.")

    tab1, tab2 = st.tabs(["📄 Content Search", "📚 Document Discovery"])

    with tab1:
        st.markdown("### Search across all document contents")
        
        with st.container():
            col1, col2 = st.columns([4, 1])
            with col1:
                q = st.text_input("Enter your query (e.g., 'What is the attention mechanism?')", key="content_q")
            with col2:
                search_type = st.selectbox("Method", ["hybrid", "vector", "bm25"], key="method")
        
        top_k = st.slider("Number of results", 1, 20, 5)
        
        if q:
            with st.spinner("Searching..."):
                results = semantic_search(q, top_k, search_type)
                
                if results and results.get("results"):
                    st.success(f"Found {len(results['results'])} relevant chunks")
                    
                    for i, res in enumerate(results["results"]):
                        meta = res.get("metadata", {})
                        with st.container():
                            st.markdown(f"""
                            <div class="result-card">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                    <span style="font-weight: bold; color: #333;">Result #{i+1}</span>
                                    <span class="result-score">Score: {res['score']:.4f}</span>
                                </div>
                                <div style="font-size: 0.95rem; line-height: 1.6;">
                                    {format_latex(res['text'])}
                                </div>
                                <div class="source-info">
                                    📍 <b>{meta.get('filename', 'Unknown')}</b> | Page {meta.get('page', 'N/A')} | {meta.get('file_type', '').upper()}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("No content matches found.")

    with tab2:
        st.markdown("### Find documents by metadata")
        
        with st.container():
            col_d1, col_d2 = st.columns([3, 1])
            with col_d1:
                doc_q = st.text_input("Global search (filename, authors, keywords...)", key="doc_q")
            with col_d2:
                st.markdown("<br>", unsafe_allow_html=True)
                do_doc_search = st.button("🔎 Search Documents", use_container_width=True, type="primary")
        
        with st.expander("Advanced Metadata Filters"):
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                f_authors = st.text_input("Authors")
            with col_f2:
                f_year_range = st.slider("Year Range", 1990, 2030, (2000, 2030), key="year_slide")
            with col_f3:
                f_keywords = st.text_input("Keywords (comma-separated)")
        
        if do_doc_search or doc_q or f_authors or f_keywords:
            with st.spinner("Finding documents..."):
                y_min, y_max = f_year_range
                docs_data = document_search(
                    query=doc_q if doc_q else None,
                    authors=f_authors if f_authors else None,
                    year_min=y_min,
                    year_max=y_max,
                    keywords=f_keywords if f_keywords else None
                )
                
                if docs_data and docs_data.get("documents"):
                    st.info(f"Found {docs_data['total']} documents")
                    
                    for doc in docs_data["documents"]:
                        with st.container():
                            st.markdown(f"""
                            <div class="document-card">
                                <div style="display: flex; justify-content: space-between;">
                                    <span style="font-weight: bold; font-size: 1.1rem; color: #1E88E5;">{doc['filename']}</span>
                                    <span style="color: #666; font-size: 0.8rem;">{doc['upload_timestamp']}</span>
                                </div>
                                <div style="margin-top: 5px; font-size: 0.9rem;">
                                    <b>Authors:</b> {doc.get('authors') or 'N/A'} | <b>Year:</b> {doc.get('year') or 'N/A'}
                                </div>
                                <div style="margin-top: 5px; font-size: 0.85rem; color: #444;">
                                    <b>Venue:</b> {doc.get('venue') or 'N/A'}
                                </div>
                                <div style="margin-top: 8px;">
                                    {" ".join([f'<span style="background: #eee; padding: 2px 8px; border-radius: 10px; font-size: 0.75rem; margin-right: 5px;">{k.strip()}</span>' for k in (doc.get('keywords') or '').split(',') if k.strip()])}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            with st.expander("View Abstract"):
                                if doc.get('abstract'):
                                    st.write(doc['abstract'])
                                else:
                                    st.write("No abstract available.")
                else:
                    st.info("No documents found matching the criteria.")

if __name__ == "__main__":
    main()
