"""Streamlit frontend for RAG Native."""
import os
import requests
import streamlit as st
import re

# API Configuration
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "https://rag-native.onrender.com")

# Page config
st.set_page_config(
    page_title="RAG Native - Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Show toast if document was recently uploaded
if "upload_success" in st.session_state:
    st.toast(f"✅ Uploaded: {st.session_state.upload_success}", icon='📄')
    del st.session_state.upload_success

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .chunk-preview {
        background-color: #fafafa;
        padding: 0.8rem;
        border-left: 3px solid #1E88E5;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .conversation-item {
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.2rem 0;
        cursor: pointer;
    }
    .conversation-item:hover {
        background-color: #f0f2f6;
    }
    .conversation-active {
        background-color: #e3f2fd;
        border-left: 3px solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)


def format_latex(text: str) -> str:
    """
    Ensures LaTeX formulas are correctly formatted for Streamlit.
    Replaces \\[ ... \\] with $$ ... $$ and \\( ... \\) with $ ... $
    """
    if not text:
        return text
    
    # Replace \[ ... \] with $$ ... $$ for block formulas
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    
    # Replace \( ... \) with $ ... $ for inline formulas
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)
    
    # Handle potential raw [ ... ] math blocks if they contain math symbols
    # This specifically addresses the user's example style
    def replace_brackets(match):
        content = match.group(1)
        # Use more specific math indicators to avoid false positives
        math_indicators = ['\\', '_', '^', '=', '+', '*', '/', '{', '}', '\sum', '\int', '\frac', '\sqrt', '\alpha', '\beta', '\gamma']
        if any(indicator in content for indicator in math_indicators):
            return f"$$\n{content.strip()}\n$$"
        return match.group(0)
    
    # Match [ ... ] that are not part of source citations [Source: ...]
    text = re.sub(r'(?<!\[Source: )\[\s*(.*?)\s*\]', replace_brackets, text)
    
    return text


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "search_type" not in st.session_state:
        st.session_state.search_type = "hybrid"
    if "top_k" not in st.session_state:
        st.session_state.top_k = 5
    if "model_mode" not in st.session_state:
        st.session_state.model_mode = "light"
    if "current_conversation_id" not in st.session_state:
        st.session_state.current_conversation_id = None
    if "conversations" not in st.session_state:
        st.session_state.conversations = []
    if "show_all_conversations" not in st.session_state:
        st.session_state.show_all_conversations = False
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0


# API functions for conversations
def get_conversations(limit: int = 50):
    """Get list of all conversations."""
    try:
        response = requests.get(f"{BACKEND_API_URL}/conversations?limit={limit}")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching conversations: {e}")
        return {"conversations": [], "total": 0}


def create_conversation(title: str = None):
    """Create a new conversation."""
    try:
        data = {"title": title} if title else {}
        response = requests.post(f"{BACKEND_API_URL}/conversations", json=data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error creating conversation: {e}")
        return None


def get_conversation(conversation_id: str):
    """Get a conversation with its messages."""
    try:
        response = requests.get(f"{BACKEND_API_URL}/conversations/{conversation_id}")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching conversation: {e}")
        return None


def delete_conversation(conversation_id: str):
    """Delete a conversation."""
    try:
        response = requests.delete(f"{BACKEND_API_URL}/conversations/{conversation_id}")
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Error deleting conversation: {e}")
        return False






def get_documents():
    """Get list of all documents."""
    try:
        response = requests.get(f"{BACKEND_API_URL}/documents")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching documents: {e}")
        return {"documents": [], "total": 0}


def chat(query, top_k, search_type, model_mode, conversation_id=None):
    """Ask a question using RAG."""
    try:
        data = {
            "query": query,
            "top_k": top_k,
            "search_type": search_type,
            "model_mode": model_mode
        }
        if conversation_id:
            data["conversation_id"] = conversation_id
            
        response = requests.post(f"{BACKEND_API_URL}/chat", json=data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error in chat: {e}")
        return None


def load_conversation_messages(conversation_id: str):
    """Load messages from a conversation into session state."""
    conv = get_conversation(conversation_id)
    if conv and conv.get("messages"):
        st.session_state.messages = [
            {
                "role": msg["role"],
                "content": msg["content"],
                "sources": msg.get("sources", [])
            }
            for msg in conv["messages"]
        ]
    else:
        st.session_state.messages = []




def render_sidebar():
    """Render sidebar with document management and conversations."""
    with st.sidebar:
        st.markdown("## 💬 Conversations")
        
        # New conversation button
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("➕ New Conversation", use_container_width=True):
                conv = create_conversation()
                if conv:
                    st.session_state.current_conversation_id = conv["id"]
                    st.session_state.messages = []
                    st.rerun()
        
        # Refresh conversations
        with col2:
            if st.button("🔄", help="Refresh conversations"):
                st.rerun()
        
        # List conversations
        # Fetch 50 by default, or 1000 if showing all
        limit = 1000 if st.session_state.show_all_conversations else 50
        convs = get_conversations(limit=limit)
        if convs["total"] > 0:
            all_convs = convs["conversations"]
            display_convs = all_convs if st.session_state.show_all_conversations else all_convs[:3]
            
            for conv in display_convs:
                is_active = st.session_state.current_conversation_id == conv["id"]
                
                col1, col2 = st.columns([4, 1])
                with col1:
                    # Truncate long titles
                    display_title = conv["title"][:20] + "..." if len(conv["title"]) > 20 else conv["title"]
                    
                    if st.button(
                        f"{display_title}",
                        key=f"conv_{conv['id']}",
                        use_container_width=True,
                        type="primary" if is_active else "secondary"
                    ):
                        st.session_state.current_conversation_id = conv["id"]
                        load_conversation_messages(conv["id"])
                        st.rerun()
                
                with col2:
                    if st.button("🗑️", key=f"del_{conv['id']}", help="Delete conversation"):
                        if delete_conversation(conv["id"]):
                            if st.session_state.current_conversation_id == conv["id"]:
                                st.session_state.current_conversation_id = None
                                st.session_state.messages = []
                            st.rerun()
            
            # Load more button
            if not st.session_state.show_all_conversations and convs["total"] > 3:
                if st.button("🔽 Load More", use_container_width=True):
                    st.session_state.show_all_conversations = True
                    st.rerun()
            elif st.session_state.show_all_conversations and convs["total"] > 3:
                if st.button("🔼 Show Less", use_container_width=True):
                    st.session_state.show_all_conversations = False
                    st.rerun()
        else:
            st.info("No conversations yet. Start a new one!")
        
        st.markdown("---")
        
        st.markdown("---")
        
        # Settings
        st.markdown("### ⚙️ Settings")
        
        st.session_state.search_type = st.selectbox(
            "Search Method",
            ["hybrid", "vector", "bm25"],
            index=["hybrid", "vector", "bm25"].index(st.session_state.search_type),
            help="Vector: semantic similarity | BM25: keyword matching | Hybrid: both combined"
        )
        
        st.session_state.model_mode = st.radio(
            "Model",
            ["light", "full"],
            index=["light", "full"].index(st.session_state.model_mode),
            format_func=lambda x: "Light" if x == "light" else "Full",
            horizontal=True,
            help="Light: faster responses | Full: more comprehensive analysis"
        )


def render_main():
    """Render main chat interface."""
    st.markdown('<p class="main-title">🔬 Research Assistant</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Ask questions about your research documents</p>', unsafe_allow_html=True)
    
    # Show current conversation info
    if st.session_state.current_conversation_id:
        st.caption(f"📌 Active conversation: {st.session_state.current_conversation_id[:8]}...")
    else:
        st.info("💡 Start a new conversation from the sidebar to enable context memory and follow-up questions.")
    
    # Check if documents exist
    docs = get_documents()
    if docs["total"] == 0:
        st.warning("👈 Please upload documents in the sidebar to get started")
        
        # Show example queries
        st.markdown("### Example Questions You Can Ask:")
        st.markdown("""
        - What are the main findings in the papers about machine learning?
        - Compare the methodologies used in different studies
        - What datasets were used in the research?
        - Summarize the conclusions from the uploaded documents
        """)
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(format_latex(message["content"]))
            
            # Display sources if available
            if message.get("sources"):
                with st.expander("📖 Sources"):
                    for source in message["sources"]:
                        # Determine confidence color
                        conf_score = source.get('confidence_score', 0)
                        if conf_score >= 75:
                            conf_color = "🟢"
                        elif conf_score >= 50:
                            conf_color = "🟡"
                        else:
                            conf_color = "🔴"
                        
                        st.markdown(
                            f"**[{source.get('citation_index', '?')}]** {source['filename']}, page {source['page']} "
                            f"({source['file_type'].upper()}) {conf_color} **{conf_score:.1f}%**"
                        )
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to display
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat(
                    query=prompt,
                    top_k=st.session_state.top_k,
                    search_type=st.session_state.search_type,
                    model_mode=st.session_state.model_mode,
                    conversation_id=st.session_state.current_conversation_id
                )
                
                if response:
                    st.markdown(format_latex(response["answer"]))
                    
                    # Show sources
                    if response.get("sources"):
                        with st.expander("📖 Sources"):
                            for source in response["sources"]:
                                # Determine confidence color
                                conf_score = source.get('confidence_score', 0)
                                if conf_score >= 75:
                                    conf_color = "🟢"
                                elif conf_score >= 50:
                                    conf_color = "🟡"
                                else:
                                    conf_color = "🔴"
                                
                                st.markdown(
                                    f"**[{source.get('citation_index', '?')}]** {source['filename']}, page {source['page']} "
                                    f"({source['file_type'].upper()}) {conf_color} **{conf_score:.1f}%**"
                                )
                    
                    # Add to message history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response.get("sources", [])
                    })
                    
                    # Rerun to update sidebar title if it's the first message
                    if len(st.session_state.messages) <= 2:
                        st.rerun()
                else:
                    st.error("Failed to generate response")


def main():
    """Main application."""
    init_session_state()
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
