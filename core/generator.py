"""
Response Generator Module - Phase B, Step 8

Generates responses using Google Gemini API with retrieved context.
"""
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import google.generativeai as genai
from dotenv import load_dotenv

from .retriever import RetrievalResult


@dataclass
class GenerationResult:
    """Represents a generation result."""
    query: str
    response: str
    context_chunks: List[RetrievalResult]
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def sources(self) -> List[str]:
        """Get list of source documents used."""
        return list(set(c.source for c in self.context_chunks))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'query': self.query,
            'response': self.response,
            'model': self.model,
            'sources': self.sources,
            'num_context_chunks': len(self.context_chunks),
            'total_tokens': self.total_tokens,
        }


class ResponseGenerator:
    """
    Generates responses using Google Gemini API with retrieved context.

    Step 8: Augmented Generation
    - Appends retrieved context to the query
    - Uses Gemini API to generate response
    """

    AVAILABLE_MODELS = {
        'gemini-2.0-flash-lite': {
            'description': 'Fast, efficient model (recommended)',
        },
        'gemini-2.0-flash': {
            'description': 'Balanced speed and quality',
        },
        'gemini-1.5-flash': {
            'description': 'Previous generation flash model',
        },
        'gemini-1.5-pro': {
            'description': 'Most capable model',
        },
    }

    DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

Instructions:
1. Answer the question using ONLY the information from the provided context.
2. If the answer cannot be found in the context, say "I cannot find this information in the provided documents."
3. Be concise and accurate.
4. When relevant, cite the source document.
5. Do not make up information that is not in the context."""

    def __init__(
        self,
        model: str = "gemini-2.0-flash-lite",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        api_key: Optional[str] = None
    ):
        """
        Initialize the response generator.

        Args:
            model: Gemini model name
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens in response
            api_key: Gemini API key (or set GEMINI_API_KEY env var)
        """
        # Load environment variables
        load_dotenv()

        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Get API key
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Configure Gemini
        genai.configure(api_key=self.api_key)

        # Initialize model
        self._model = genai.GenerativeModel(model)

    def _build_prompt(
        self,
        query: str,
        context_chunks: List[RetrievalResult],
        system_prompt: str
    ) -> str:
        """
        Build the augmented prompt with context.

        Args:
            query: User query
            context_chunks: Retrieved context chunks
            system_prompt: System prompt

        Returns:
            Complete prompt string
        """
        # Format context
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            context_parts.append(
                f"[Document {i}: {chunk.source} (relevance: {chunk.score:.2f})]\n{chunk.text}"
            )
        context_str = "\n\n---\n\n".join(context_parts)

        # Build full prompt
        prompt = f"""{system_prompt}

## Context Documents:

{context_str}

## Question:
{query}

## Answer:"""

        return prompt

    def generate(
        self,
        query: str,
        context_chunks: List[RetrievalResult],
        system_prompt: Optional[str] = None
    ) -> GenerationResult:
        """
        Generate a response using Gemini with retrieved context.

        Args:
            query: User query
            context_chunks: Retrieved context chunks
            system_prompt: Optional custom system prompt

        Returns:
            GenerationResult with response and metadata
        """
        if system_prompt is None:
            system_prompt = self.DEFAULT_SYSTEM_PROMPT

        # Build augmented prompt
        prompt = self._build_prompt(query, context_chunks, system_prompt)

        # Generate response
        generation_config = genai.GenerationConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )

        response = self._model.generate_content(
            prompt,
            generation_config=generation_config
        )

        # Extract token counts if available
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            prompt_tokens = getattr(usage, 'prompt_token_count', 0)
            completion_tokens = getattr(usage, 'candidates_token_count', 0)
            total_tokens = getattr(usage, 'total_token_count', 0)

        return GenerationResult(
            query=query,
            response=response.text,
            context_chunks=context_chunks,
            model=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            metadata={
                'temperature': self.temperature,
                'max_tokens': self.max_tokens,
            }
        )

    def generate_streaming(
        self,
        query: str,
        context_chunks: List[RetrievalResult],
        system_prompt: Optional[str] = None
    ):
        """
        Generate a streaming response.

        Args:
            query: User query
            context_chunks: Retrieved context chunks
            system_prompt: Optional custom system prompt

        Yields:
            Response text chunks
        """
        if system_prompt is None:
            system_prompt = self.DEFAULT_SYSTEM_PROMPT

        prompt = self._build_prompt(query, context_chunks, system_prompt)

        generation_config = genai.GenerationConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )

        response = self._model.generate_content(
            prompt,
            generation_config=generation_config,
            stream=True
        )

        for chunk in response:
            if chunk.text:
                yield chunk.text

    def update_parameters(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Update generation parameters.

        Args:
            model: New model name
            temperature: New temperature
            max_tokens: New max tokens
        """
        if model is not None and model != self.model_name:
            self.model_name = model
            self._model = genai.GenerativeModel(model)

        if temperature is not None:
            self.temperature = temperature

        if max_tokens is not None:
            self.max_tokens = max_tokens

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model configuration.

        Returns:
            Dictionary with model information
        """
        return {
            'model': self.model_name,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'available_models': list(self.AVAILABLE_MODELS.keys()),
        }
