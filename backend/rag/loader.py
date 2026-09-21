from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


def load_and_split_documents(
    knowledge_dir: Path,
    *,
    chunk_size: int = 500,
    overlap: int = 80,
) -> list[Document]:
    """Load Markdown knowledge files and preserve source/section metadata."""
    section_documents: list[Document] = []
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "title"), ("##", "section"), ("###", "subsection")],
        strip_headers=False,
    )

    for path in sorted(knowledge_dir.glob("*.md")):
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        for document in header_splitter.split_text(content):
            document.metadata["source"] = path.name
            section_documents.append(document)

    if not section_documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    return splitter.split_documents(section_documents)
