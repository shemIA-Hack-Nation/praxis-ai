from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class NotebookCell(BaseModel):
    """Schema for notebook cell data"""
    type: str
    content: Any
    execution_count: Optional[int] = None
    source: Optional[str] = None
    outputs: Optional[List[Dict[str, Any]]] = None

class NotebookMetadata(BaseModel):
    """Schema for notebook metadata"""
    notebook_id: str
    filename: str
    language_info: Dict[str, Any]
    kernelspec: Dict[str, Any]
    total_cells: int
    
class NotebookSummary(BaseModel):
    """Schema for notebook summary statistics"""
    total_elements: int
    code_cells: int
    markdown_cells: int
    output_cells: int
    
class NotebookUploadResponse(BaseModel):
    """Schema for notebook upload response"""
    success: bool
    message: str
    notebook_metadata: Optional[NotebookMetadata] = None
    cells: List[NotebookCell]
    summary: NotebookSummary

class NotebookProcessRequest(BaseModel):
    """Schema for notebook processing request"""
    notebook_data: Dict[str, Any]
    processing_options: Optional[Dict[str, Any]] = None

class OrchestratorResponse(BaseModel):
    """Schema for orchestrator processing response"""
    success: bool
    message: str
    processed_data: Dict[str, Any]
    next_steps: List[str]

class ErrorResponse(BaseModel):
    """Schema for error responses"""
    success: bool = False
    error: str
    detail: Optional[str] = None

class NotebookListResponse(BaseModel):
    """Schema for listing notebooks response"""
    success: bool
    notebooks: List[NotebookMetadata]
    count: int
