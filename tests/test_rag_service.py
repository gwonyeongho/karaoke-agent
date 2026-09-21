from types import SimpleNamespace

from langchain_core.documents import Document

from backend.rag.service import NO_EVIDENCE_ANSWER, RagService


class FakeVectorStore:
    def __init__(self, results):
        self.results = results

    def similarity_search_with_score(self, question, k):
        return self.results[:k]


class FakeChatModel:
    def __init__(self, answer="예약 버튼을 누르세요."):
        self.answer = answer
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(content=self.answer)


def test_answer_uses_retrieved_context_and_returns_source():
    document = Document(
        page_content="곡 번호를 입력한 뒤 예약 버튼을 누릅니다.",
        metadata={"source": "reservation_guide.md", "section": "일반 예약"},
    )
    model = FakeChatModel()
    service = RagService(
        vector_store=FakeVectorStore([(document, 0.2)]),
        chat_model=model,
        score_threshold=0.65,
    )

    response = service.answer("예약은 어떻게 해?", top_k=4)

    assert response.grounded is True
    assert response.answer == "예약 버튼을 누르세요."
    assert response.sources[0].source == "reservation_guide.md"
    assert "예약 버튼" in response.sources[0].excerpt
    assert "곡 번호를 입력" in str(model.calls[0])


def test_answer_refuses_when_retrieval_has_no_relevant_evidence():
    document = Document(
        page_content="음정은 -6부터 6까지 조절합니다.",
        metadata={"source": "audio_guide.md"},
    )
    model = FakeChatModel()
    service = RagService(
        vector_store=FakeVectorStore([(document, 0.91)]),
        chat_model=model,
        score_threshold=0.65,
    )

    response = service.answer("오늘 날씨가 어때?", top_k=4)

    assert response.answer == NO_EVIDENCE_ANSWER
    assert response.grounded is False
    assert response.sources == []
    assert model.calls == []
