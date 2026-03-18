"""
Query and chat routes.
Handles semantic search and RAG-powered chat endpoints.
"""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.embedding import get_single_embedding
from app.services.vector_store import search_vectors
from app.services.rag_pipeline import generate_answer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["query"])


# ── Request / Response Models ────────────────────────────────────────────────
class QueryRequest(BaseModel):
    """Request body for semantic search."""
    query: str
    top_k: int = 5


class ChatRequest(BaseModel):
    """Request body for chat (RAG pipeline)."""
    query: str
    history: list[dict] = []
    top_k: int = 5


class QueryResponse(BaseModel):
    """Response for semantic search."""
    query: str
    results: list[dict]


class ChatResponse(BaseModel):
    """Response for chat."""
    answer: str
    sources: list[dict]
    context_used: bool
    llm_used: bool
    error: str | None = None


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Perform semantic search across uploaded documents.

    Converts the query into an embedding and searches Endee
    for the most similar document chunks.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        # Generate query embedding
        query_embedding = get_single_embedding(request.query)

        # Search Endee
        results = search_vectors(query_embedding, top_k=request.top_k)

        return QueryResponse(query=request.query, results=results)

    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Full RAG pipeline: semantic search + LLM generation.

    1. Embed the query
    2. Retrieve relevant chunks from Endee
    3. Pass context + query to the LLM
    4. Return generated answer with sources
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        # Step 1: Generate query embedding
        query_embedding = get_single_embedding(request.query)

        # Step 2: Search Endee for relevant context
        search_results = search_vectors(query_embedding, top_k=request.top_k)

        # Step 3: Generate answer via RAG pipeline
        result = generate_answer(
            query=request.query,
            search_results=search_results,
            chat_history=request.history,
        )

        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            context_used=result["context_used"],
            llm_used=result["llm_used"],
            error=result.get("error"),
        )

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")
