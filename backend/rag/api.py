from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .config import KNOWLEDGE_DIR, RAG_CHAT_MODEL, RAG_EMBED_MODEL
from .documents import get_knowledge_document, knowledge_version, list_knowledge_documents
from .runtime import RagUnavailableError, get_rag_service
from .schemas import RagQueryRequest, RagQueryResponse

router = APIRouter(prefix="/api/v1/rag", tags=["RAG"])


class DocumentSummary(BaseModel):
    id: str
    title: str
    description: str
    filename: str


class DocumentDetail(DocumentSummary):
    content: str


@router.get("/documents", response_model=list[DocumentSummary])
def documents() -> list[DocumentSummary]:
    return [
        DocumentSummary(
            id=doc.id,
            title=doc.title,
            description=doc.description,
            filename=doc.filename,
        )
        for doc in list_knowledge_documents(KNOWLEDGE_DIR)
    ]


@router.get("/documents/{document_id}", response_model=DocumentDetail)
def document_detail(document_id: str) -> DocumentDetail:
    doc = get_knowledge_document(KNOWLEDGE_DIR, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="검색 자료를 찾을 수 없습니다.")
    return DocumentDetail(**doc.__dict__)


@router.get("/health")
def health():
    docs = list_knowledge_documents(KNOWLEDGE_DIR)
    return {
        "document_count": len(docs),
        "knowledge_version": knowledge_version(KNOWLEDGE_DIR),
        "chat_model": RAG_CHAT_MODEL,
        "embedding_model": RAG_EMBED_MODEL,
        "note": "문서 조회는 모델 실행 상태와 독립적입니다.",
    }


@router.post("/query", response_model=RagQueryResponse)
def query(request: RagQueryRequest) -> RagQueryResponse:
    try:
        return get_rag_service().answer(request.question, request.top_k)
    except RagUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="RAG 응답 생성에 실패했습니다. Ollama 모델 상태를 확인하세요.",
        ) from exc
