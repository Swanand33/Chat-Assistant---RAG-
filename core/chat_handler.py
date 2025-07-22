"""
Chat Handler Module
Manages conversations with OpenAI LLM using RAG context.
"""

from openai import OpenAI
from typing import List, Dict, Optional
import streamlit as st
import os


class ChatHandler:
    def __init__(self):
        """Initialize chat handler with OpenAI client."""
        self.client = None
        self.conversation_history = []
        
        # Initialize OpenAI client
        try:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except Exception as e:
            st.error(f"Failed to initialize OpenAI client: {str(e)}")
    
    def create_system_prompt(self) -> str:
        """Create system prompt for the RAG assistant."""
        return """You are a helpful document assistant. You answer questions based on the provided document context.

Guidelines:
- Use ONLY the information provided in the context to answer questions
- If the context doesn't contain relevant information, say "I cannot find information about that in the document"
- Be concise and accurate
- Quote specific parts of the document when relevant
- If you're unsure, admit it rather than making up information

Context will be provided with each question."""
    
    def format_prompt_with_context(self, query: str, context: str) -> str:
        """
        Format user query with RAG context.
        
        Args:
            query: User's question
            context: Retrieved document context
            
        Returns:
            Formatted prompt string
        """
        if not context.strip():
            return f"""No relevant context found in the document.

Question: {query}

Please let the user know that you cannot find relevant information in the document to answer their question."""
        
        return f"""Based on the following context from the document, please answer the question:

CONTEXT:
{context}

QUESTION: {query}

Please provide a clear, accurate answer based only on the information in the context above."""
    
    def get_llm_response(self, query: str, context: str) -> str:
        """
        Get response from OpenAI LLM with RAG context.
        
        Args:
            query: User's question
            context: Retrieved document context
            
        Returns:
            LLM response string
        """
        if not self.client:
            return "❌ OpenAI client not initialized. Please check your API key."
        
        try:
            # Format prompt with context
            user_prompt = self.format_prompt_with_context(query, context)
            
            # Create messages for chat completion
            messages = [
                {"role": "system", "content": self.create_system_prompt()},
                {"role": "user", "content": user_prompt}
            ]
            
            # Add conversation history (last 2 exchanges to avoid token limits)
            if self.conversation_history:
                recent_history = self.conversation_history[-4:]  # Last 2 Q&A pairs
                messages = [{"role": "system", "content": self.create_system_prompt()}] + recent_history + [{"role": "user", "content": user_prompt}]
            
            # Get completion from OpenAI
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=500,
                temperature=0.1,  # Low temperature for factual responses
                top_p=0.9
            )
            
            answer = response.choices[0].message.content
            
            # Update conversation history
            self.add_to_history(query, answer)
            
            return answer
            
        except Exception as e:
            error_msg = f"Error getting LLM response: {str(e)}"
            st.error(error_msg)
            return f"❌ {error_msg}"
    
    def add_to_history(self, question: str, answer: str):
        """Add Q&A pair to conversation history."""
        self.conversation_history.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ])
        
        # Keep only last 10 messages to avoid token limits
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get formatted conversation history for display."""
        formatted_history = []
        for i in range(0, len(self.conversation_history), 2):
            if i + 1 < len(self.conversation_history):
                formatted_history.append({
                    "question": self.conversation_history[i]["content"],
                    "answer": self.conversation_history[i + 1]["content"]
                })
        return formatted_history
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
    
    def is_ready(self) -> bool:
        """Check if chat handler is ready."""
        return self.client is not None