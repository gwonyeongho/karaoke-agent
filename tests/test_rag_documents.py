from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.rag import api
from backend.rag.documents import get_knowledge_document, list_knowledge_documents


def test_document_allowlist_lists_and_reads_markdown(tmp_path: Path):
    (tmp_path / "guide.md").write_text("# 안내\n\n예약 버튼을 누릅니다.", encoding="utf-8")

    documents = list_knowledge_documents(tmp_path)

    assert len(documents) == 1
    assert documents[0].id == "guide"
    assert documents[0].title == "안내"
    assert get_knowledge_document(tmp_path, "guide").content.endswith("누릅니다.")
    assert get_knowledge_document(tmp_path, "../guide") is None


def test_document_api_works_without_ollama(monkeypatch, tmp_path: Path):
    (tmp_path / "help.md").write_text("# 도움말\n\n마이크 권한을 확인합니다.", encoding="utf-8")
    monkeypatch.setattr(api, "KNOWLEDGE_DIR", tmp_path)
    app = FastAPI()
    app.include_router(api.router)
    client = TestClient(app)

    listing = client.get("/api/v1/rag/documents")
    detail = client.get("/api/v1/rag/documents/help")
    traversal = client.get("/api/v1/rag/documents/..%2Fhelp")

    assert listing.status_code == 200
    assert listing.json()[0]["title"] == "도움말"
    assert detail.status_code == 200
    assert "마이크 권한" in detail.json()["content"]
    assert traversal.status_code == 404
