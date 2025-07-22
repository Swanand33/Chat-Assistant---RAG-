"""
RAG Pipeline Module - WORKING VERSION
Fixed imports for current LangChain structure.
"""

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from typing import List
import streamlit as st
import os


class RAGPipeline:
    def __init__(self):
        """Initialize RAG pipeline with OpenAI embeddings."""
        self.embeddings = None
        self.vector_store = None
        self.retriever = None
        
        # Initialize OpenAI embeddings
        try:
            self.embeddings = OpenAIEmbeddings()
        except Exception as e:
            st.error(f"Failed to initialize embeddings: {str(e)}")
    
    def create_vector_store(self, text_chunks: List[str]) -> bool:
        """Create FAISS vector store from text chunks."""
        if not text_chunks or not self.embeddings:
            return False
        
        try:
            # Convert to documents
            documents = [Document(page_content=chunk) for chunk in text_chunks]
            
            # Create FAISS vector store
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            
            # Create retriever
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
            
            return True
            
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            return False
    
    def retrieve_relevant_chunks(self, query: str, k: int = 4) -> List[Document]:
        """Retrieve relevant document chunks for a query."""
        if not self.retriever:
            return []
        
        try:
            # Update k if needed
            if k != 4:
                self.retriever = self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": k}
                )
            
            relevant_docs = self.retriever.get_relevant_documents(query)
            return relevant_docs
            
        except Exception as e:
            st.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def get_retrieval_context(self, query: str, k: int = 4) -> str:
        """Get formatted context string from retrieved chunks."""
        relevant_docs = self.retrieve_relevant_chunks(query, k)
        
        if not relevant_docs:
            return ""
        
        # Format context with chunk numbers
        context_parts = []
        for i, doc in enumerate(relevant_docs, 1):
            context_parts.append(f"[Chunk {i}]:\n{doc.page_content}\n")
        
        return "\n".join(context_parts)
    
    def is_ready(self) -> bool:
        """Check if RAG pipeline is ready for queries."""
        return self.vector_store is not None and self.retriever is not None