import os
import sys
import shutil
from typing import List, Dict, Any, Optional

# Add the parent folder to the python path to resolve local imports cleanly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.config import settings
from backend.services.document_parser import DocumentParser
from backend.services.chunker import Chunker
from backend.services.vector_store import FAISSVectorStore
from backend.services.retriever import HybridRetriever
from backend.services.generator import LLMGenerator


app = FastAPI(title="Local RAG & Vector DB Playground API")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Services lazily so the app starts instantly
vector_store = None
retriever = None
generator = None

def get_services():
    global vector_store, retriever, generator
    if vector_store is None:
        vector_store = FAISSVectorStore()
        retriever = HybridRetriever(vector_store)
        generator = LLMGenerator()
    return vector_store, retriever, generator

# Pydantic schemas for request validation
class QueryRequest(BaseModel):
    query: str
    mode: str = "hybrid"  # dense, sparse, hybrid
    k: int = 4
    provider: str = "simulation"  # simulation, gemini, ollama
    model: str = "gemini-1.5-flash"
    api_key: Optional[str] = None

class ConfigRequest(BaseModel):
    api_key: str

# API Endpoints
@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(50),
    chunk_strategy: str = Form("recursive")  # recursive, fixed
):
    """Saves document, extracts text, performs chunking, and returns visual layout for comparison."""
    # Ensure directory exists
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    
    file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Parse file contents
        content = DocumentParser.parse_file(file_path)
        
        # Chunk text
        if chunk_strategy == "fixed":
            chunks = Chunker.fixed_size_chunk(content, chunk_size, chunk_overlap)
        else:
            chunks = Chunker.recursive_character_chunk(content, chunk_size, chunk_overlap)
            
        return {
            "filename": file.filename,
            "char_count": len(content),
            "chunk_count": len(chunks),
            "chunks": chunks,
            "original_text": content
        }
    except Exception as e:
        # Cleanup file if failed
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/index")
async def index_chunks(payload: Dict[str, Any]):
    """Receives chunks and indexes them into FAISS and BM25."""
    chunks = payload.get("chunks", [])
    filename = payload.get("filename", "unknown")
    
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks provided for indexing.")
        
    try:
        vs, ret, _ = get_services()
        
        # Inject filename metadata into chunks
        formatted_chunks = []
        for chunk in chunks:
            formatted_chunks.append({
                "text": chunk["text"],
                "start_idx": chunk.get("start_idx"),
                "end_idx": chunk.get("end_idx"),
                "file_name": filename
            })
            
        vs.add_chunks(formatted_chunks)
        ret.rebuild_sparse_index()
        
        return {
            "status": "success",
            "message": f"Successfully indexed {len(chunks)} chunks from '{filename}'.",
            "total_indexed_chunks": len(vs.chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
async def query_rag(payload: QueryRequest):
    """Performs retrieval and generation for a user query."""
    try:
        vs, ret, gen = get_services()
        
        # If API key is provided in the request, update generator key
        if payload.api_key:
            gen.update_gemini_key(payload.api_key)
            
        # 1. Retrieve
        retrieved_chunks = ret.retrieve(
            query=payload.query,
            k=payload.k,
            mode=payload.mode
        )
        
        # 2. Generate
        generation_result = gen.generate_response(
            query=payload.query,
            chunks=retrieved_chunks,
            provider=payload.provider,
            model=payload.model
        )
        
        return {
            "query": payload.query,
            "retrieved_chunks": retrieved_chunks,
            "answer": generation_result["answer"],
            "provider": generation_result["provider"],
            "model": generation_result["model"],
            "error": generation_result.get("error", False)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/clear")
async def clear_database():
    """Clears uploaded files, local FAISS index, and metadata."""
    try:
        vs, ret, _ = get_services()
        vs.clear()
        ret.rebuild_sparse_index()
        
        # Clear uploads folder
        if os.path.exists(settings.UPLOAD_DIR):
            for file_name in os.listdir(settings.UPLOAD_DIR):
                file_path = os.path.join(settings.UPLOAD_DIR, file_name)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")
                    
        return {"status": "success", "message": "Database and uploaded documents cleared successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/config")
async def update_config(payload: ConfigRequest):
    """Updates the Gemini API Key configuration dynamically."""
    try:
        _, _, gen = get_services()
        gen.update_gemini_key(payload.api_key)
        return {"status": "success", "message": "API key updated successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def get_status():
    """Returns database and index counts."""
    try:
        vs, _, gen = get_services()
        
        # Check files in uploads
        files = []
        if os.path.exists(settings.UPLOAD_DIR):
            files = os.listdir(settings.UPLOAD_DIR)
            
        return {
            "total_chunks": len(vs.chunks),
            "files": files,
            "gemini_ready": gen.gemini_configured,
            "embedding_model": settings.EMBEDDING_MODEL_NAME
        }
    except Exception as e:
        return {
            "total_chunks": 0,
            "files": [],
            "gemini_ready": False,
            "embedding_model": settings.EMBEDDING_MODEL_NAME,
            "error": str(e)
        }

# Serve Frontend static files
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")

# Route for root / to serve index.html
@app.get("/")
async def read_root():
    return FileResponse(os.path.join(frontend_dir, "index.html"))

# Mount remaining static files (style.css, app.js, etc.)
app.mount("/", StaticFiles(directory=frontend_dir), name="frontend")

if __name__ == "__main__":
    import uvicorn
    # Start server
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
