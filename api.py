"""
api.py
FastAPI — RAG Pipeline Gateway (RunPod + Local Ollama Support)
"""

import sys
sys.modules['torchcodec'] = None

import json
import logging
import os
import uuid
import httpx
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from final_rag.qdrant_storage.store import QdrantManager
from final_rag.agent.orchestrator import get_orchestrator
from final_rag.ingestion.embedder import get_embedder
from final_rag.ingestion.parser import DocumentParser
from final_rag.ingestion.chunker import DocumentChunker
from final_rag.db.database import (
    create_tables,
    fetch_all_sessions,
    fetch_conversation_history,
    insert_conversation,
    upsert_session_title,
    delete_session,
    get_document_by_filename,
    insert_document,
    update_document_status,
    cleanup_stuck_documents,
    delete_document_record,
    list_documents,
    clear_all_documents,
    health_check_db,
)
import final_rag.config as config

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(
    title="c-net RAG API",
    version="1.0.0",
    description="RAG Backend with RunPod + Local Ollama Support"
)

# ── RunPod Configuration ───────────────────────────────────────────────
DEPLOYMENT_MODE = config.DEPLOYMENT_MODE
RUNPOD_ENDPOINT = config.RUNPOD_ENDPOINT_URL
RUNPOD_API_KEY = config.RUNPOD_API_KEY

logger.info(f"🚀 Deployment Mode: {DEPLOYMENT_MODE.upper()}")
if DEPLOYMENT_MODE == "runpod":
    logger.info(f"📡 RunPod Endpoint: {RUNPOD_ENDPOINT}")

# ── Helper: RunPod API Call ────────────────────────────────────────────
async def call_runpod(action: str, payload: dict) -> dict:
    """
    Call RunPod API for LLM inference.
    Fallback to local Ollama if RunPod unavailable.
    """
    if not RUNPOD_ENDPOINT or not RUNPOD_API_KEY:
        logger.warning("RunPod not configured, falling back to local Ollama")
        return {"status": "error", "message": "RunPod not configured"}
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}"
    }
    
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{RUNPOD_ENDPOINT}/run",
                json={"input": {**payload, "action": action}},
                headers=headers,
            )
            return response.json()
    except Exception as e:
        logger.error(f"RunPod call failed: {e}")
        return {"status": "error", "message": str(e)}

# ── CORS Configuration ─────────────────────────────────────────────────
_raw_origins = config.CORS_ORIGINS
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping")
def ping():
    return {"status": "ok", "mode": DEPLOYMENT_MODE}

# ── Globals ────────────────────────────────────────────────────────────
db           = None
embedder     = None
orchestrator = None
doc_parser   = None
doc_chunker  = None


# ── Startup ────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    global db, embedder, orchestrator, doc_parser, doc_chunker
    
    logger.info("=" * 60)
    logger.info("🚀 RAG API Startup")
    logger.info(f"   Mode: {DEPLOYMENT_MODE.upper()}")
    logger.info(f"   Models: {config.CLEANER_MODEL}, {config.GENERATOR_MODEL}")
    logger.info("=" * 60)
    
    create_tables()

    db           = QdrantManager()
    db.setup_database()
    embedder     = get_embedder(db=db)
    orchestrator = get_orchestrator(embedder=embedder)
    doc_parser   = DocumentParser(output_dir=config.MD_OUTPUT_DIR)
    doc_chunker  = DocumentChunker()

    cleanup_stuck_documents()
    logger.info("✅ Cleaned up stuck processing records on startup")
    logger.info(f"✅ API Ready | CORS origins: {ALLOWED_ORIGINS}")


# ── Shutdown ───────────────────────────────────────────────────────────
@app.on_event("shutdown")
def shutdown():
    logger.info("🛑 API shutting down...")
    global db
    try:
        if db and hasattr(db, "client") and db.client:
            db.client.close()
    except Exception as e:
        logger.error(f"Error closing Qdrant: {e}")

    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared.")
    except ImportError:
        pass


# ── Request Schemas ────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str
    query: str


class RenameTitleRequest(BaseModel):
    title: str


class DocumentContentUpdate(BaseModel):
    content: str


# ── POST /upload ───────────────────────────────────────────────────────
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and ingest a document"""
    
    existing_doc = get_document_by_filename(file.filename)
    if existing_doc:
        logger.info(f"File already exists, skipping: {file.filename}")
        return {
            "status": "already_exists",
            "message": "File is already stored in the database.",
        }

    try:
        file_bytes = await file.read()
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Could not read file.")

    doc_id = str(uuid.uuid4())
    doc_type = Path(file.filename).suffix.lower()

    try:
        insert_document(
            document_id=doc_id,
            file_name=file.filename,
            doc_type=doc_type,
            file_data=file_bytes,
            status="processing",
        )
    except Exception as e:
        logger.error(f"Failed to create document placeholder: {e}")
        raise HTTPException(status_code=500, detail="Database error.")

    try:
        logger.info(f"Starting ingestion for: {file.filename}")

        parsed_result = doc_parser.parse_bytes(file_bytes, file.filename)
        if not parsed_result.success:
            raise Exception(f"Parsing failed: {parsed_result.error}")

        chunks = doc_chunker.chunk(parsed_result)
        embedder.embed_and_store(chunks)

        update_document_status(file.filename, "ingested")

        logger.info(f"✅ Successfully ingested: {file.filename}")
        return {
            "status": "success",
            "message": "File successfully uploaded and processed.",
            "document_id": doc_id,
            "file_name": file.filename,
        }

    except Exception as e:
        logger.error(f"Upload pipeline failed for {file.filename}: {e}")

        try:
            db.delete_document(file.filename)
        except Exception as rollback_err:
            logger.critical(f"CRITICAL: Qdrant rollback failed: {rollback_err}")

        try:
            update_document_status(file.filename, "failed")
        except Exception:
            pass

        raise HTTPException(status_code=500, detail=str(e))


# ── POST /chat/stream ──────────────────────────────────────────────────
@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """Stream chat responses with RAG context"""
    history_turns = fetch_conversation_history(req.session_id, limit=6)
    history = [
        {"question": turn.question, "answer": turn.answer}
        for turn in history_turns
    ]
    if history and history[-1]["question"] == req.query and not history[-1]["answer"]:
        history.pop()

    def event_generator():
        full_answer = []
        metadata_chunk = ""
        source_name = ""
        page_label = ""
        page_no = 0
        stream_error = None

        try:
            for token in orchestrator.run(
                query=req.query,
                history=history,
                active_document=None,
            ):
                if token.startswith("__METADATA__:"):
                    metadata_chunk = token
                    continue
                full_answer.append(token)
                yield f"data: {json.dumps({'token': token})}\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            stream_error = str(e)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

        complete_answer = "".join(full_answer).strip()

        if metadata_chunk:
            try:
                _, metadata_part = metadata_chunk.split("__METADATA__:")
                sources = json.loads(metadata_part.strip())
                if sources:
                    top_source = sources[0]
                    source_name = top_source.get("source_name", "Unknown")
                    page_label = top_source.get("page_label", "")
                    page_no = top_source.get("page_no", 0)
            except Exception as e:
                logger.debug(f"Metadata parse error: {e}")

        try:
            insert_conversation(
                session_id=req.session_id,
                question=req.query,
                answer=complete_answer,
                source_name=source_name,
                page_label=page_label,
                page_no=page_no,
                error=stream_error,
            )
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── GET /sessions ──────────────────────────────────────────────────────
@app.get("/sessions")
def get_sessions():
    """Get all chat sessions"""
    sessions = fetch_all_sessions()
    return {"sessions": sessions}


# ── GET /sessions/{session_id}/history ─────────────────────────────────
@app.get("/sessions/{session_id}/history")
def get_session_history(session_id: str):
    """Get conversation history for a session"""
    history = fetch_conversation_history(session_id)
    return {"session_id": session_id, "history": history}


# ── DELETE /sessions/{session_id} ──────────────────────────────────────
@app.delete("/sessions/{session_id}")
def delete_session_endpoint(session_id: str):
    """Delete a session"""
    delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}


# ── PUT /sessions/{session_id}/title ────────────────────────────────────
@app.put("/sessions/{session_id}/title")
def rename_session(session_id: str, req: RenameTitleRequest):
    """Rename a session"""
    upsert_session_title(session_id, req.title)
    return {"status": "renamed", "session_id": session_id, "title": req.title}


# ── GET /api/documents ────────────────────────────────────────────────
@app.get("/api/documents")
def get_documents():
    """List all documents"""
    docs = list_documents()
    return {"documents": docs}


# ── DELETE /api/documents ─────────────────────────────────────────────
@app.delete("/api/documents")
def delete_all_documents():
    """Delete all documents"""
    clear_all_documents()
    return {"status": "cleared"}


# ── DELETE /api/documents/{file_name} ────────────────────────────────
@app.delete("/api/documents/{file_name}")
def delete_document(file_name: str):
    """Delete a specific document"""
    try:
        db.delete_document(file_name)
        delete_document_record(file_name)
        stem = Path(file_name).stem
        md_path = config.MD_OUTPUT_DIR / f"{stem}.md"
        if md_path.exists():
            md_path.unlink()
        return {"status": "deleted", "file_name": file_name}
    except Exception as e:
        logger.error(f"Failed to delete document {file_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /api/documents/{file_name}/search ───────────────────────────
@app.get("/api/documents/{file_name}/search")
def search_in_document(file_name: str, q: str = Query(...)):
    """Search within a specific document"""
    stem = Path(file_name).stem
    md_path = config.MD_OUTPUT_DIR / f"{stem}.md"
    if not md_path.exists():
        raise HTTPException(status_code=404, detail="Document not found")

    content = md_path.read_text(encoding="utf-8")
    
    results = []
    query_lower = q.lower()
    
    lines = content.split("\n")
    matches = 0
    snippets = []
    
    for i, line in enumerate(lines):
        if query_lower in line.lower():
            matches += 1
            snippet_text = line
            for j in range(max(0, i - 2), min(len(lines), i + 3)):
                if j == i:
                    continue
                paragraph = lines[j].strip()
                if paragraph:
                    snippet_text = paragraph
            
            import re
            snippet_clean = re.sub(r'\s+', ' ', snippet_text).strip()
            if snippet_clean and snippet_clean not in snippets:
                snippets.append(snippet_clean)
    
    if matches > 0:
        results.append({
            "file_name": file_name,
            "matches": matches,
            "snippets": snippets
        })
        
    return results


# ── GET /api/documents/{file_name}/content ────────────────────────────
@app.get("/api/documents/{file_name}/content")
def get_document_content(file_name: str):
    """Get document content"""
    stem = Path(file_name).stem
    md_path = config.MD_OUTPUT_DIR / f"{stem}.md"
    if not md_path.exists():
        raise HTTPException(status_code=404, detail="Content not found")
    content = md_path.read_text(encoding="utf-8")
    return {"file_name": file_name, "content": content}


# ── PUT /api/documents/{file_name}/content ────────────────────────────
@app.put("/api/documents/{file_name}/content")
def update_document_content(file_name: str, payload: DocumentContentUpdate):
    """Update document content"""
    stem = Path(file_name).stem
    md_path = config.MD_OUTPUT_DIR / f"{stem}.md"
    
    md_path.parent.mkdir(exist_ok=True, parents=True)
    md_path.write_text(payload.content, encoding="utf-8")
    
    try:
        db.delete_document(file_name)
    except Exception as e:
        logger.error(f"Error deleting old chunks: {e}")
        
    try:
        from final_rag.ingestion.parser import ParseResult, BlockRecord, DocumentMeta, ExtractionMethod
        import hashlib
        import re
        
        file_hash = hashlib.md5(payload.content.encode('utf-8')).hexdigest()
        
        meta = DocumentMeta(
            doc_id=file_hash[:12],
            file_name=file_name,
            file_path=str(md_path),
            file_type=".md",
            file_size_kb=len(payload.content) / 1024,
            page_count=1,
            has_tables=False,
            parse_success=True,
            filename_tokens=re.split(r'[_\-\s]+', stem.lower()),
        )
        
        blocks = [BlockRecord(block_type="text", content=payload.content, page_no=1, page_label="1")]
        
        parse_res = ParseResult(
            file_name=file_name,
            file_type=".md",
            method_used=ExtractionMethod.YOLO,
            markdown=payload.content,
            meta=meta,
            total_pages=1,
            success=True,
            blocks=blocks,
            doc_id=meta.doc_id,
            filename_tokens=meta.filename_tokens
        )
        
        chunks = doc_chunker.chunk(parse_res)
        embedder.embed_and_store(chunks)
        
        doc_record = get_document_by_filename(file_name)
        if doc_record:
            update_document_status(doc_record.document_id, "success")
        
        return {"status": "success", "chunks_indexed": len(chunks)}
    except Exception as e:
        logger.error(f"Error re-ingesting {file_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /api/documents/preview-parse ──────────────────────────────────
@app.post("/api/documents/preview-parse")
async def preview_parse_document(file: UploadFile = File(...)):
    """Preview document parsing"""
    try:
        file_bytes = await file.read()
    except Exception as e:
        logger.error(f"Failed to read uploaded file for preview: {e}")
        raise HTTPException(status_code=500, detail="Could not read file.")

    try:
        preview_parser = DocumentParser(output_dir=None)
        parsed_result = preview_parser.parse_bytes(file_bytes, file.filename)
        if not parsed_result.success:
            raise Exception(f"Parsing failed: {parsed_result.error}")

        return {
            "file_name": file.filename,
            "content": parsed_result.markdown,
        }
    except Exception as e:
        logger.error(f"Preview parse failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /api/documents/replace ────────────────────────────────────────
@app.post("/api/documents/replace")
async def replace_document(
    file: UploadFile = File(...),
    old_file_name: str = Form(...),
):
    """Replace a document"""
    try:
        db.delete_document(old_file_name)
        logger.info(f"Deleted old vectors for '{old_file_name}' from Qdrant.")
    except Exception as e:
        logger.error(f"Failed to delete old vectors: {e}")

    try:
        delete_document_record(old_file_name)
        logger.info(f"Deleted old DB record for '{old_file_name}'.")
    except Exception as e:
        logger.error(f"Failed to delete old DB record: {e}")

    try:
        old_stem = Path(old_file_name).stem
        old_md_path = config.MD_OUTPUT_DIR / f"{old_stem}.md"
        if old_md_path.exists():
            old_md_path.unlink()
            logger.info(f"Deleted old md file: {old_md_path}")
    except Exception as e:
        logger.error(f"Failed to delete old md file: {e}")

    try:
        file_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail="Could not read new file.")

    doc_id = str(uuid.uuid4())
    doc_type = Path(file.filename).suffix.lower()

    try:
        insert_document(
            document_id=doc_id,
            file_name=file.filename,
            doc_type=doc_type,
            file_data=file_bytes,
            status="processing",
        )
    except Exception as e:
        logger.error(f"Failed to create document placeholder: {e}")
        raise HTTPException(status_code=500, detail="Database error.")

    try:
        parsed_result = doc_parser.parse_bytes(file_bytes, file.filename)
        if not parsed_result.success:
            raise Exception(f"Parsing failed: {parsed_result.error}")

        chunks = doc_chunker.chunk(parsed_result)
        embedder.embed_and_store(chunks)
        update_document_status(file.filename, "ingested")

        logger.info(f"Successfully replaced '{old_file_name}' with '{file.filename}'")
        return {
            "status": "success",
            "old_deleted": old_file_name,
            "new_file": file.filename,
            "chunks_indexed": len(chunks),
        }
    except Exception as e:
        logger.error(f"Replace pipeline failed: {e}")
        try:
            db.delete_document(file.filename)
        except Exception:
            pass
        try:
            update_document_status(file.filename, "failed")
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /api/stats ────────────────────────────────────────────────────
@app.get("/api/stats")
def get_stats():
    """Get RAG statistics"""
    try:
        num_chunks = db.client.count(db.collection_name).count
    except Exception as e:
        logger.error(f"Failed to get Qdrant count: {e}")
        num_chunks = 0
        
    files = set()
    try:
        points, _ = db.client.scroll(
            collection_name=db.collection_name, 
            limit=10000, 
            with_payload=True, 
            with_vectors=False
        )
        for point in points:
            src = (point.payload.get("source_file") or 
                   point.payload.get("source") or 
                   point.payload.get("file_name") or 
                   point.payload.get("source_name"))
            if src:
                files.add(str(src))
    except Exception as e:
        logger.error(f"Failed to get Qdrant files: {e}")
        pass

    return {
        "status": "online",
        "mode": DEPLOYMENT_MODE,
        "num_chunks": num_chunks,
        "files": list(files),
        "graph": {"connected": False, "entities": 0, "relationships": 0}
    }


# ── GET /health ────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        db.get_client().get_collections()
        health_check_db()

        return {
            "status": "healthy",
            "mode": DEPLOYMENT_MODE,
            "qdrant": "connected",
            "database": "connected",
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
        }


# ── GET / ──────────────────────────────────────────────────────────────
@app.get("/")
def root():
    """API information"""
    return {
        "name": "c-net RAG API",
        "version": "1.0.0",
        "mode": DEPLOYMENT_MODE,
        "status": "running",
        "endpoints": [
            "/ping",
            "/health",
            "/docs",
            "/upload",
            "/chat/stream",
            "/sessions",
            "/api/documents",
            "/api/stats",
        ]
    }


# ── Main ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "final_rag.api:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )