"""
Upload and document management routes.
Handles file upload, document listing, and deletion.
"""

import uuid
import logging
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.services.vector_store import store_vectors, delete_vectors_by_document
from app.services.embedding import get_embeddings
from app.utils.pdf_parser import extract_text, chunk_text

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["documents"])

# ── In-memory document registry ──────────────────────────────────────────────
# Tracks uploaded documents and their chunk IDs for management.
documents_store: dict[str, dict] = {}


# ── Request / Response Models ────────────────────────────────────────────────
class DocumentResponse(BaseModel):
    """Response for document info."""
    id: str
    filename: str
    num_chunks: int
    uploaded_at: str
    file_size: int


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF or TXT) for processing.

    Steps:
    1. Extract text from file
    2. Split into chunks
    3. Generate embeddings
    4. Store in Endee vector database
    """
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    allowed_extensions = (".pdf", ".txt")
    if not file.filename.lower().endswith(allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}",
        )

    try:
        # Read file content
        content = await file.read()
        file_size = len(content)

        if file_size == 0:
            raise HTTPException(status_code=400, detail="File is empty")

        logger.info(f"Processing upload: {file.filename} ({file_size} bytes)")

        # Step 1: Extract text
        text = extract_text(content, file.filename)
        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract any text from the file")

        # Step 2: Chunk text
        chunks = chunk_text(text)
        if not chunks:
            raise HTTPException(status_code=400, detail="No text chunks generated")

        # Step 3: Generate embeddings
        embeddings = get_embeddings(chunks)

        # Step 4: Store in Endee
        doc_id = str(uuid.uuid4())
        chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadata = [
            {
                "text": chunk,
                "source": file.filename,
                "document_id": doc_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            for i, chunk in enumerate(chunks)
        ]

        store_vectors(chunk_ids, embeddings, metadata)

        # Track document
        doc_info = {
            "id": doc_id,
            "filename": file.filename,
            "num_chunks": len(chunks),
            "chunk_ids": chunk_ids,
            "uploaded_at": datetime.now().isoformat(),
            "file_size": file_size,
        }
        documents_store[doc_id] = doc_info

        logger.info(f"✅ Document uploaded: {file.filename} → {len(chunks)} chunks")

        return DocumentResponse(
            id=doc_id,
            filename=file.filename,
            num_chunks=len(chunks),
            uploaded_at=doc_info["uploaded_at"],
            file_size=file_size,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@router.get("/documents", response_model=list[DocumentResponse])
async def list_documents():
    """List all uploaded documents."""
    return [
        DocumentResponse(
            id=doc["id"],
            filename=doc["filename"],
            num_chunks=doc["num_chunks"],
            uploaded_at=doc["uploaded_at"],
            file_size=doc["file_size"],
        )
        for doc in documents_store.values()
    ]


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its vectors from Endee."""
    if document_id not in documents_store:
        raise HTTPException(status_code=404, detail="Document not found")

    doc = documents_store[document_id]

    # Delete each chunk vector from Endee
    for chunk_id in doc.get("chunk_ids", []):
        delete_vectors_by_document(chunk_id)

    # Remove from local store
    del documents_store[document_id]
    logger.info(f"🗑️  Deleted document: {doc['filename']}")

    return {"message": f"Document '{doc['filename']}' deleted successfully"}
