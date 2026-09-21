import pytest
from pydantic import ValidationError

from backend.rag.schemas import RagQueryRequest, RagQueryResponse, RagSource


def test_rag_query_request_trims_question_and_bounds_top_k():
    request = RagQueryRequest(question="  예약은 어떻게 해?  ", top_k=5)

    assert request.question == "예약은 어떻게 해?"
    assert request.top_k == 5

    with pytest.raises(ValidationError):
        RagQueryRequest(question="   ")
    with pytest.raises(ValidationError):
        RagQueryRequest(question="질문", top_k=6)


def test_rag_query_response_contains_grounding_sources():
    response = RagQueryResponse(
        answer="예약 버튼을 누르세요.",
        grounded=True,
        sources=[RagSource(source="guide.md", excerpt="예약 버튼", score=0.2)],
    )

    assert response.grounded is True
    assert response.sources[0].source == "guide.md"
