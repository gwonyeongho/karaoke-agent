from pathlib import Path
import os

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent
KNOWLEDGE_DIR = BACKEND_DIR / "knowledge"
PERSIST_DIR = PROJECT_DIR / ".rag_index"

RAG_CHAT_MODEL = os.getenv("RAG_CHAT_MODEL", "qwen3:1.7b")
# bge-m3 supports Korean and multilingual semantic retrieval.
RAG_EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "bge-m3")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
RAG_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.65"))
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "500"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "80"))
