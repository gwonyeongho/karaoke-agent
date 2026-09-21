from functools import lru_cache
from pathlib import Path
import hashlib

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings

from .config import (
    KNOWLEDGE_DIR,
    PERSIST_DIR,
    RAG_CHAT_MODEL,
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    RAG_EMBED_MODEL,
    RAG_SCORE_THRESHOLD,
)
from .documents import knowledge_version
from .loader import load_and_split_documents
from .service import RagService


class RagUnavailableError(RuntimeError):
    pass


def build_service(
    *,
    knowledge_dir: Path = KNOWLEDGE_DIR,
    persist_dir: Path = PERSIST_DIR,
    store_factory=Chroma,
    embedding_factory=OllamaEmbeddings,
    chat_factory=ChatOllama,
) -> RagService:
    chunks = load_and_split_documents(
        knowledge_dir,
        chunk_size=RAG_CHUNK_SIZE,
        overlap=RAG_CHUNK_OVERLAP,
    )
    if not chunks:
        raise RagUnavailableError("검색할 지식 문서가 없습니다.")

    version = knowledge_version(knowledge_dir)
    collection_name = f"karaoke_{version}"
    try:
        embeddings = embedding_factory(model=RAG_EMBED_MODEL)
        vector_store = store_factory(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
        )
        current = vector_store.get().get("ids", [])
        if not current:
            ids = [
                hashlib.sha256(
                    f"{chunk.metadata.get('source')}:{index}:{chunk.page_content}".encode("utf-8")
                ).hexdigest()
                for index, chunk in enumerate(chunks)
            ]
            vector_store.add_documents(chunks, ids=ids)
        chat_model = chat_factory(model=RAG_CHAT_MODEL, temperature=0)
    except Exception as exc:
        raise RagUnavailableError(
            f"RAG 초기화 실패: Ollama와 {RAG_EMBED_MODEL} 모델을 확인하세요."
        ) from exc

    return RagService(
        vector_store=vector_store,
        chat_model=chat_model,
        score_threshold=RAG_SCORE_THRESHOLD,
    )


@lru_cache(maxsize=1)
def get_rag_service() -> RagService:
    return build_service()
