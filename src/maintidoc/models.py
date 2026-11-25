from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ElementType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


class DocumentElement(BaseModel):
    document_id: str
    file_name: str
    page: int = Field(ge=1)
    element_id: str
    element_type: ElementType
    content: str = ""
    image_path: Path | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchHit(BaseModel):
    document_id: str
    file_name: str
    page: int
    element_id: str
    element_type: ElementType
    content: str
    score: float
    image_path: str | None = None

    @property
    def citation(self) -> str:
        return f"[{self.file_name}, p. {self.page}, {self.element_id}]"


class IndexingStats(BaseModel):
    document_id: str
    pages: int
    text_chunks: int
    tables: int
    images: int


class AgentAnswer(BaseModel):
    answer: str
    sources: list[SearchHit] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
