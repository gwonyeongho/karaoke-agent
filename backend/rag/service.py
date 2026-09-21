from typing import Any

from .schemas import RagQueryResponse, RagSource

NO_EVIDENCE_ANSWER = "제공된 문서에서 확인할 수 없습니다."

SYSTEM_PROMPT = """너는 노래방 제어 패널 사용 도우미다.
반드시 제공된 검색 문맥만 사용한다.
문맥에 없는 내용을 추측하거나 일반 지식으로 보완하지 않는다.
답을 찾을 수 없으면 '제공된 문서에서 확인할 수 없습니다.'라고 답한다.
기기 상태를 변경했다고 말하지 않는다.
짧고 명확한 한국어로 답한다."""


class RagService:
    def __init__(
        self,
        *,
        vector_store: Any,
        chat_model: Any,
        score_threshold: float = 0.65,
    ) -> None:
        self.vector_store = vector_store
        self.chat_model = chat_model
        self.score_threshold = score_threshold

    def answer(self, question: str, top_k: int = 4) -> RagQueryResponse:
        candidates = self.vector_store.similarity_search_with_score(question, k=top_k)
        relevant = [
            (document, float(score))
            for document, score in candidates
            if float(score) <= self.score_threshold
        ]
        if not relevant:
            return RagQueryResponse(
                answer=NO_EVIDENCE_ANSWER,
                grounded=False,
                sources=[],
            )

        context_parts: list[str] = []
        sources: list[RagSource] = []
        for index, (document, score) in enumerate(relevant, start=1):
            source = str(document.metadata.get("source", "unknown"))
            section = document.metadata.get("section")
            label = f"{source} / {section}" if section else source
            context_parts.append(f"[{index}] {label}\n{document.page_content}")
            sources.append(
                RagSource(
                    source=source,
                    excerpt=document.page_content.strip(),
                    score=score,
                )
            )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "[검색 문맥]\n"
                    + "\n\n".join(context_parts)
                    + "\n\n[질문]\n"
                    + question
                ),
            },
        ]
        result = self.chat_model.invoke(messages)
        answer = str(getattr(result, "content", result)).strip()
        if not answer:
            answer = NO_EVIDENCE_ANSWER
            return RagQueryResponse(answer=answer, grounded=False, sources=[])

        return RagQueryResponse(answer=answer, grounded=True, sources=sources)
