import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Ensure the root of the project is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from backend.services.chunker import Chunker

# Mock the embedding model before importing main to prevent it from downloading/loading the actual model in tests
mock_transformer = MagicMock()
mock_transformer.get_sentence_embedding_dimension.return_value = 384
mock_transformer.encode.return_value = [[0.1] * 384]

with patch("backend.services.vector_store.SentenceTransformer", return_value=mock_transformer):
    from backend.main import app, get_services

client = TestClient(app)

def test_fixed_size_chunking():
    text = "This is a simple text that we want to split into smaller pieces for testing."
    # Test normal fixed size chunking
    chunks = Chunker.fixed_size_chunk(text, chunk_size=20, chunk_overlap=5)
    assert len(chunks) > 0
    assert all("id" in c and "text" in c and "start_idx" in c and "end_idx" in c for c in chunks)
    assert chunks[0]["text"] == text[0:20]

    # Test edge cases: empty text
    assert Chunker.fixed_size_chunk("", 20, 5) == []
    
    # Test edge case: invalid arguments
    chunks_invalid = Chunker.fixed_size_chunk(text, chunk_size=-10, chunk_overlap=50)
    assert len(chunks_invalid) > 0

def test_recursive_character_chunking():
    text = "This is paragraph one.\n\nThis is paragraph two. It has two sentences."
    chunks = Chunker.recursive_character_chunk(text, chunk_size=30, chunk_overlap=5)
    assert len(chunks) > 0
    assert all("id" in c and "text" in c and "start_idx" in c and "end_idx" in c for c in chunks)
    
    # Test edge cases: empty text
    assert Chunker.recursive_character_chunk("", 30, 5) == []

def test_api_status():
    # Mock vector store and generator
    mock_vs = MagicMock()
    mock_vs.chunks = [{"id": 0, "text": "chunk1"}]
    mock_gen = MagicMock()
    mock_gen.gemini_configured = False
    
    with patch("backend.main.get_services") as mock_get_services:
        mock_get_services.return_value = (mock_vs, MagicMock(), mock_gen)
        
        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert data["total_chunks"] == 1
        assert data["gemini_ready"] is False

def test_api_clear():
    mock_vs = MagicMock()
    mock_ret = MagicMock()
    
    with patch("backend.main.get_services") as mock_get_services:
        mock_get_services.return_value = (mock_vs, mock_ret, MagicMock())
        
        response = client.post("/api/clear")
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        mock_vs.clear.assert_called_once()
        mock_ret.rebuild_sparse_index.assert_called_once()

def test_api_config():
    mock_gen = MagicMock()
    
    with patch("backend.main.get_services") as mock_get_services:
        mock_get_services.return_value = (MagicMock(), MagicMock(), mock_gen)
        
        response = client.post("/api/config", json={"api_key": "test-api-key"})
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        mock_gen.update_gemini_key.assert_called_once_with("test-api-key")

def test_api_query():
    mock_vs = MagicMock()
    mock_ret = MagicMock()
    mock_gen = MagicMock()
    
    mock_ret.retrieve.return_value = [{"text": "retrieved text", "score": 0.9}]
    mock_gen.generate_response.return_value = {
        "answer": "This is the generated answer.",
        "provider": "simulation",
        "model": "gemini-1.5-flash"
    }
    
    with patch("backend.main.get_services") as mock_get_services:
        mock_get_services.return_value = (mock_vs, mock_ret, mock_gen)
        
        payload = {
            "query": "What is RAG?",
            "mode": "hybrid",
            "k": 4,
            "provider": "simulation",
            "model": "gemini-1.5-flash",
            "api_key": "test-key"
        }
        response = client.post("/api/query", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "This is the generated answer."
        assert len(data["retrieved_chunks"]) == 1
        mock_gen.update_gemini_key.assert_called_once_with("test-key")
