"""
RAG Chat Assistant - Main Streamlit Application
A showcase project demonstrating RAG (Retrieval-Augmented Generation) with LangChain and OpenAI.
"""

import streamlit as st
import os
from dotenv import load_dotenv

# Import our custom modules
from core.document_processor import DocumentProcessor
from core.rag_pipeline import RAGPipeline
from core.chat_handler import ChatHandler

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .source-chunk {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #667eea;
    }
    .stats-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'processor' not in st.session_state:
        st.session_state.processor = DocumentProcessor()
    
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = RAGPipeline()
    
    if 'chat_handler' not in st.session_state:
        st.session_state.chat_handler = ChatHandler()
    
    if 'document_loaded' not in st.session_state:
        st.session_state.document_loaded = False
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []


def main():
    """Main application function."""
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 RAG Chat Assistant</h1>
        <p>Upload a document and chat with it using AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for document upload and settings
    with st.sidebar:
        st.header("📄 Document Upload")
        
        # API Key check
        if not os.getenv("OPENAI_API_KEY"):
            st.error("⚠️ OpenAI API Key not found! Please set your OPENAI_API_KEY in the .env file.")
            st.stop()
        else:
            st.success("✅ OpenAI API Key loaded")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'docx'],
            help="Upload a PDF or DOCX file to chat with"
        )
        
        # Process uploaded file
        if uploaded_file is not None:
            with st.spinner("Processing document..."):
                text, chunks = st.session_state.processor.process_uploaded_file(uploaded_file)
                
                if text and chunks:
                    # Create vector store
                    success = st.session_state.rag_pipeline.create_vector_store(chunks)
                    
                    if success:
                        st.session_state.document_loaded = True
                        st.success(f"✅ Document processed successfully!")
                        
                        # Document statistics
                        stats = st.session_state.processor.get_document_stats(text, chunks)
                        
                        st.markdown("### 📊 Document Statistics")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Words", f"{stats['total_words']:,}")
                            st.metric("Characters", f"{stats['total_characters']:,}")
                        with col2:
                            st.metric("Chunks", stats['total_chunks'])
                            st.metric("Avg Chunk Size", stats['avg_chunk_size'])
        
        # Settings
        st.header("⚙️ Settings")
        show_sources = st.checkbox("Show source chunks", value=True)
        num_sources = st.slider("Number of source chunks", 1, 8, 4)
    
    # Main chat interface
    if st.session_state.document_loaded:
        st.header("💬 Chat with your document")
        
        # Chat input
        user_question = st.text_input(
            "Ask a question about your document:",
            placeholder="e.g., What is this document about?",
            key="user_input"
        )
        
        # Process question
        if user_question:
            with st.spinner("Thinking..."):
                # Get relevant context
                context = st.session_state.rag_pipeline.get_retrieval_context(
                    user_question, k=num_sources
                )
                
                # Get LLM response
                response = st.session_state.chat_handler.get_llm_response(
                    user_question, context
                )
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "question": user_question,
                    "answer": response,
                    "context": context
                })
        
        # Display chat history
        if st.session_state.chat_history:
            st.header("🗨️ Conversation")
            
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.container():
                    # Question
                    st.markdown(f"**🙋 You:** {chat['question']}")
                    
                    # Answer
                    st.markdown(f"**🤖 Assistant:** {chat['answer']}")
                    
                    # Source chunks (if enabled)
                    if show_sources and chat['context']:
                        with st.expander("📚 Source Chunks", expanded=False):
                            st.markdown(f"```\n{chat['context']}\n```")
                    
                    st.divider()
        
        # Clear chat button
        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.session_state.chat_handler.clear_history()
                st.rerun()
    
    else:
        # Welcome message
        st.markdown("""
        ### 👋 Welcome to RAG Chat Assistant!
        
        This application demonstrates **Retrieval-Augmented Generation (RAG)** technology:
        
        1. **Upload** a PDF or DOCX document
        2. **Ask questions** about the content
        3. **Get AI-powered answers** based on the document
        
        #### 🛠️ Tech Stack:
        - **LangChain** for RAG pipeline
        - **OpenAI GPT-3.5** for language understanding
        - **FAISS** for vector storage
        - **Streamlit** for the user interface
        
        #### 📋 How it works:
        1. Document is split into chunks
        2. Chunks are embedded using OpenAI embeddings
        3. Relevant chunks are retrieved for each question
        4. LLM generates answers using the retrieved context
        
        **👈 Upload a document to get started!**
        """)


if __name__ == "__main__":
    main()