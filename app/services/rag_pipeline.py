"""
RAG (Retrieval-Augmented Generation) pipeline.
Combines retrieved context from Endee with LLM generation.
"""

import logging
from openai import OpenAI
from app.config import settings

logger = logging.getLogger(__name__)

# Global OpenAI client
_openai_client: OpenAI | None = None


def get_openai_client() -> OpenAI | None:
    """Get the OpenAI client. Returns None if no API key is configured."""
    global _openai_client
    if _openai_client is None and settings.OPENAI_API_KEY and settings.OPENAI_API_KEY != "your-openai-api-key-here":
        _openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        logger.info("OpenAI client initialized")
    return _openai_client


def build_context(search_results: list[dict]) -> str:
    """
    Build a context string from search results.

    Args:
        search_results: List of results from Endee search

    Returns:
        Formatted context string
    """
    context_parts = []
    for i, result in enumerate(search_results, 1):
        meta = result.get("metadata", {})
        text = meta.get("text", "")
        source = meta.get("source", "Unknown")
        score = result.get("score", 0.0)

        context_parts.append(
            f"[Source {i}: {source} (relevance: {score:.3f})]\n{text}"
        )

    return "\n\n---\n\n".join(context_parts)


def generate_answer(
    query: str,
    search_results: list[dict],
    chat_history: list[dict] | None = None,
) -> dict:
    """
    Generate an answer using the RAG pipeline.

    If OpenAI is configured, sends the context + query to the LLM.
    Otherwise, returns the relevant context chunks directly.

    Args:
        query: User's question
        search_results: Retrieved context from Endee
        chat_history: Optional list of previous messages [{role, content}]

    Returns:
        Dict with 'answer', 'sources', and 'context_used'
    """
    context = build_context(search_results)

    # Extract source references
    sources = []
    for result in search_results:
        meta = result.get("metadata", {})
        sources.append({
            "document": meta.get("source", "Unknown"),
            "chunk_index": meta.get("chunk_index", 0),
            "relevance": round(result.get("score", 0.0), 4),
        })

    client = get_openai_client()

    if client is None:
        # No LLM available — return context-based response
        logger.info("No OpenAI key configured, returning context-only response")
        return {
            "answer": _format_context_response(query, context, search_results),
            "sources": sources,
            "context_used": True,
            "llm_used": False,
        }

    # Build messages for the LLM
    system_prompt = (
        "You are an intelligent AI Knowledge Assistant. Your task is to answer questions "
        "based on the provided context from uploaded documents. Follow these rules:\n"
        "1. Answer ONLY based on the provided context.\n"
        "2. If the context doesn't contain enough information, say so clearly.\n"
        "3. Cite the source documents when relevant.\n"
        "4. Be concise but thorough.\n"
        "5. Use markdown formatting for better readability.\n"
    )

    messages = [{"role": "system", "content": system_prompt}]

    # Add chat history if provided
    if chat_history:
        for msg in chat_history[-6:]:  # Keep last 6 messages for context
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })

    # Add the context and current query
    user_message = (
        f"Context from knowledge base:\n\n{context}\n\n"
        f"---\n\n"
        f"Question: {query}\n\n"
        f"Please provide a comprehensive answer based on the above context."
    )
    messages.append({"role": "user", "content": user_message})

    try:
        response = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
        )
        answer = response.choices[0].message.content
        logger.info("Generated LLM response successfully")

        return {
            "answer": answer,
            "sources": sources,
            "context_used": True,
            "llm_used": True,
        }
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return {
            "answer": _format_context_response(query, context, search_results),
            "sources": sources,
            "context_used": True,
            "llm_used": False,
            "error": f"LLM unavailable: {str(e)}",
        }


def _format_context_response(
    query: str,
    context: str,
    search_results: list[dict],
) -> str:
    """Format a response using only the retrieved context (no LLM)."""
    if not search_results:
        return (
            "I couldn't find any relevant information in the uploaded documents "
            "to answer your question. Please try uploading relevant documents first."
        )

    return (
        f"**Based on your uploaded documents, here are the most relevant passages:**\n\n"
        f"{context}\n\n"
        f"---\n\n"
        f"*Note: Connect an OpenAI API key for AI-generated answers. "
        f"Currently showing raw context from {len(search_results)} relevant chunks.*"
    )
