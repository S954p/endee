"""
Endee AI Assistant — FastAPI Backend
Main application with REST API endpoints for document upload,
semantic search, and RAG-powered chat.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.services.vector_store import init_index
from app.routes.upload import router as upload_router
from app.routes.query import router as query_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown) ────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    logger.info("🚀 Starting Endee AI Assistant backend...")
    try:
        init_index()
        logger.info("✅ Endee index initialized")
    except Exception as e:
        logger.warning(f"⚠️  Could not connect to Endee: {e}. Start Endee server first.")
    yield
    logger.info("👋 Shutting down backend...")


# ── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Endee AI Knowledge Assistant",
    description="RAG-powered knowledge assistant using Endee vector database",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL, "http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register Routers ─────────────────────────────────────────────────────────
app.include_router(upload_router)
app.include_router(query_router)


# ── Health Check ─────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "endee-ai-assistant"}


# ── Run directly ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )
