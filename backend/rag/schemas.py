from pydantic import BaseModel, Field, field_validator


class RagQueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=4, ge=1, le=5)

    @field_validator("question", mode="before")
    @classmethod
    def strip_question(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class RagSource(BaseModel):
    source: str
    excerpt: str
    score: float | None = None


class RagQueryResponse(BaseModel):
    answer: str
    grounded: bool
    sources: list[RagSource] = Field(default_factory=list)
