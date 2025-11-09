"""
API v1 endpoints.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
import aiofiles
import os
import logging
from pathlib import Path
from typing import Optional

from app.core.config import settings
from app.api.v1.schemas import UploadNotebookResponse, ErrorResponse
from app.services.pipeline_manager import PipelineManager
from app.api.v1.agents import router as agents_router
from app.api.v1.websocket import notify_scraping_progress

logger = logging.getLogger(__name__)

router = APIRouter()

# Include agent routes for A2A communication
router.include_router(agents_router)

# Initialize pipeline manager (lazy initialization - won't load heavy models until first use)
pipeline_manager = PipelineManager()


@router.post("/upload_paper/", response_model=UploadNotebookResponse)
async def upload_paper(
    file: UploadFile = File(...),
    client_id: Optional[str] = Query(None, description="WebSocket client ID for real-time progress updates")
):
    """
    Upload a research paper PDF and generate analysis.
    Supports real-time progress updates via WebSocket if client_id is provided.
    
    Args:
        file: Research paper PDF file (.pdf)
        client_id: Optional WebSocket client ID for real-time progress updates
    
    Returns:
        UploadNotebookResponse with paper and analysis in markdown format
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a research paper PDF (.pdf) file."
        )
    
    # Check file size
    file_content = await file.read()
    if len(file_content) > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds maximum allowed size of {settings.MAX_FILE_SIZE / 1024 / 1024}MB"
        )
    
    # Set up progress callback if client_id provided
    if client_id:
        async def progress_callback(event: str, data: dict):
            await notify_scraping_progress(client_id, event, data)
        
        pipeline_manager.set_progress_callback(progress_callback)
        
        await notify_scraping_progress(client_id, "upload_started", {
            "filename": file.filename,
            "message": "File uploaded, starting processing..."
        })
    
    # Save uploaded file
    file_path = Path(settings.UPLOAD_DIR) / file.filename
    async with aiofiles.open(file_path, 'wb') as f:
        await f.write(file_content)
    
    try:
        # Process paper through pipeline
        result = await pipeline_manager.process_paper(str(file_path))
        
        if client_id:
            await notify_scraping_progress(client_id, "processing_complete", {
                "message": "Processing complete"
            })
        
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return UploadNotebookResponse(
            novelty_analysis=result["novelty_analysis"],
            knowledge_graph=result["knowledge_graph"],
            status="success"
        )
    
    except Exception as e:
        # Clean up uploaded file on error
        if os.path.exists(file_path):
            os.remove(file_path)
        
        if client_id:
            await notify_scraping_progress(client_id, "error", {
                "error": str(e),
                "message": "Error processing paper"
            })
        
        logger.error(f"Error processing paper: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing paper: {str(e)}" if settings.LOG_LEVEL == "DEBUG" else "An error occurred while processing the research paper"
        )


@router.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}
