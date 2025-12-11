"""
Response generation using Google Gemini.
"""
import os
from typing import List, Optional
import google.generativeai as genai
from dotenv import load_dotenv
from .data_models import RetrievalResult, GenerationResult

class ResponseGenerator:
    """Generates responses using the Gemini API with retrieved context."""
    DEFAULT_SYSTEM_PROMPT = "Answer the question using only the provided context. If the answer is not in the context, say so."

    def __init__(self, model: str = "gemini-1.5-flash-latest", api_key: Optional[str] = None):
        load_dotenv()
        self.model_name = model
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found.")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model)

    def generate(self, query: str, context_chunks: List[RetrievalResult]) -> GenerationResult:
        """Generates a response using the provided query and context."""
        context_str = "\n\n".join(f"[Source: {c.chunk.source}]\n{c.chunk.text}" for c in context_chunks)
        prompt = f"{self.DEFAULT_SYSTEM_PROMPT}\n\nContext:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"
        
        response = self.model.generate_content(prompt)
        
        return GenerationResult(
            query=query,
            response=response.text,
            context_chunks=context_chunks,
            model=self.model_name,
        )

