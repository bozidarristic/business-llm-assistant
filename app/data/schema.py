from pydantic import BaseModel, Field


class BusinessDocument(BaseModel):
    id: str
    source: str
    content: str
    metadata: dict = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    id: str
    content: str
    metadata: dict
    score: float | None = None
