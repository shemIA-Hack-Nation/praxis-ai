"""
API request and response schemas.
"""
from pydantic import BaseModel
from typing import Optional


class UploadNotebookResponse(BaseModel):
    """Response schema for notebook upload."""
    novelty_analysis: str
    knowledge_graph: dict
    status: str = "success"


class ErrorResponse(BaseModel):
    """Error response schema."""
    message: str
    detail: Optional[str] = None


class PaperMetadata(BaseModel):
    """Metadata for a research paper."""
    title: str
    abstract: str
    url: Optional[str] = None
    year: Optional[int] = None
    citations: Optional[int] = None
    authors: Optional[list[str]] = None


class NoveltyScore(BaseModel):
    """Novelty score and justification."""
    score: float  # 1-10
    justification: str
    semantic_distance: Optional[float] = None
    similar_papers: Optional[list[PaperMetadata]] = None
    kg_gaps: Optional[int] = None
