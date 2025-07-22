# 🤖 RAG Chat Assistant

A production-ready **Retrieval-Augmented Generation (RAG)** application that lets you chat with your documents using AI. Built with LangChain, OpenAI GPT-3.5, and Streamlit.

screenshots\Screenshot 2025-07-22 115206.png
screenshots\Screenshot 2025-07-22 115305.png

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

- 📄 **Multi-format support**: PDF and DOCX files
- 🤖 **AI-powered chat**: Ask questions in natural language
- 🔍 **Intelligent retrieval**: Finds relevant document sections
- 💬 **Context-aware responses**: Maintains conversation history
- 📊 **Document analytics**: Shows processing statistics
- 🎨 **Modern UI**: Clean, professional interface

## 🏗️ Architecture

```
Document → Text Extraction → Chunking → Embeddings → Vector Store
                                                          ↓
User Query → Embedding → Similarity Search → Context → LLM → Response
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.8+ |
| **LLM** | OpenAI GPT-3.5-turbo |
| **RAG Framework** | LangChain |
| **Vector Database** | FAISS |
| **Embeddings** | OpenAI text-embedding-ada-002 |
| **Frontend** | Streamlit |
| **Document Processing** | PyMuPDF, python-docx |

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/rag-chat-assistant.git
cd rag-chat-assistant
```

### 2. Set up virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
```
Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

### 5. Run the application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 Usage

1. **Upload Document**: Choose a PDF or DOCX file from the sidebar
2. **Wait for Processing**: The app will extract text and create embeddings
3. **Start Chatting**: Ask questions about your document
4. **View Sources**: Toggle source chunks to see where answers come from

### Example Questions
- "What is this document about?"
- "Summarize the main points"
- "What are the key findings?"
- "Explain [specific topic] from the document"

## 🔧 Configuration

### Chunk Settings
- **Chunk Size**: 1000 characters (adjustable in `document_processor.py`)
- **Chunk Overlap**: 200 characters
- **Retrieval Count**: 4 chunks per query (adjustable in UI)

### Model Settings
- **LLM Model**: gpt-3.5-turbo
- **Temperature**: 0.1 (factual responses)
- **Max Tokens**: 500 per response