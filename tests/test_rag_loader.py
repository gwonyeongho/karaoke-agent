from pathlib import Path

from backend.rag.loader import load_and_split_documents


def test_load_and_split_documents_keeps_source_metadata(tmp_path: Path):
    (tmp_path / "guide.md").write_text(
        "# 예약\n곡 번호를 입력하고 예약 버튼을 누릅니다.",
        encoding="utf-8",
    )

    chunks = load_and_split_documents(tmp_path, chunk_size=80, overlap=10)

    assert chunks
    assert chunks[0].metadata["source"] == "guide.md"
    assert "예약" in chunks[0].page_content


def test_load_and_split_documents_records_markdown_section(tmp_path: Path):
    (tmp_path / "guide.md").write_text(
        "# 노래방 안내\n\n## 일반 예약\n곡 번호를 입력하고 예약 버튼을 누릅니다.\n\n"
        "## 예약 취소\n곡 번호를 입력하고 예약취소 버튼을 누릅니다.",
        encoding="utf-8",
    )

    chunks = load_and_split_documents(tmp_path, chunk_size=100, overlap=10)

    sections = {chunk.metadata.get("section") for chunk in chunks}
    assert "일반 예약" in sections
    assert "예약 취소" in sections
