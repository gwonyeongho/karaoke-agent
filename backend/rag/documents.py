import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KnowledgeDocument:
    id: str
    title: str
    description: str
    filename: str
    content: str


def _slug_for(path: Path) -> str:
    return path.stem.replace("_", "-")


def _title_and_description(content: str, fallback: str) -> tuple[str, str]:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    title = fallback
    body_start = 0
    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip()
        body_start = 1
    description = ""
    for line in lines[body_start:]:
        if not line.startswith("#") and not line.startswith("```"):
            description = re.sub(r"[`*_]", "", line)[:160]
            break
    return title, description


def list_knowledge_documents(knowledge_dir: Path) -> list[KnowledgeDocument]:
    documents: list[KnowledgeDocument] = []
    if not knowledge_dir.exists():
        return documents
    for path in sorted(knowledge_dir.glob("*.md")):
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        title, description = _title_and_description(content, path.stem)
        documents.append(
            KnowledgeDocument(
                id=_slug_for(path),
                title=title,
                description=description,
                filename=path.name,
                content=content,
            )
        )
    return documents


def get_knowledge_document(knowledge_dir: Path, document_id: str) -> KnowledgeDocument | None:
    # IDs are looked up in a server-created allowlist; user input never becomes a path.
    return next(
        (doc for doc in list_knowledge_documents(knowledge_dir) if doc.id == document_id),
        None,
    )


def knowledge_version(knowledge_dir: Path) -> str:
    digest = hashlib.sha256()
    for document in list_knowledge_documents(knowledge_dir):
        digest.update(document.filename.encode("utf-8"))
        digest.update(document.content.encode("utf-8"))
    return digest.hexdigest()[:16]
