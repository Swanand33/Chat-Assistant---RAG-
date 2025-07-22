"""
Document Processing Module - WORKING VERSION
Uses pdfplumber and fixed LangChain imports.
"""

import pdfplumber
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Tuple
import streamlit as st


class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize document processor with chunking parameters."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file using pdfplumber."""
        try:
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file."""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading DOCX: {str(e)}")
            return ""
    
    def process_uploaded_file(self, uploaded_file) -> Tuple[str, List[str]]:
        """Process uploaded file and return extracted text and chunks."""
        if uploaded_file is None:
            return "", []
        
        # Save uploaded file temporarily
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Extract text based on file type
        file_extension = uploaded_file.name.lower().split('.')[-1]
        
        if file_extension == 'pdf':
            text = self.extract_text_from_pdf(uploaded_file.name)
        elif file_extension == 'docx':
            text = self.extract_text_from_docx(uploaded_file.name)
        else:
            st.error("Unsupported file type. Please upload PDF or DOCX files.")
            return "", []
        
        if not text.strip():
            st.error("No text could be extracted from the file.")
            return "", []
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        return text, chunks
    
    def get_document_stats(self, text: str, chunks: List[str]) -> dict:
        """Get basic statistics about the processed document."""
        return {
            "total_characters": len(text),
            "total_words": len(text.split()),
            "total_chunks": len(chunks),
            "avg_chunk_size": len(text) // len(chunks) if chunks else 0
        }