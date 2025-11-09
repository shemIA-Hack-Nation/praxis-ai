from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any, List

# Simple test router
router = APIRouter()

@router.post("/upload_notebook/")
async def upload_notebook_simple(file: UploadFile = File(...)):
    """
    Simple test upload endpoint that returns mock data
    """
    # Validate file type
    if not file.filename or not file.filename.endswith('.ipynb'):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Please upload a .ipynb file."
        )
    
    # Return simple mock response for testing
    mock_response = {
        "success": True,
        "message": f"Successfully parsed notebook: {file.filename}",
        "cells": [
            {
                "type": "markdown",
                "content": "# Test Notebook",
                "source": "# Test Notebook\n\nThis is a test markdown cell."
            },
            {
                "type": "code", 
                "content": "print('Hello World')",
                "source": "print('Hello World')",
                "execution_count": 1
            },
            {
                "type": "output_text",
                "content": "Hello World\n"
            }
        ],
        "summary": {
            "total_elements": 3,
            "code_cells": 1,
            "markdown_cells": 1,
            "output_cells": 1
        }
    }
    
    return JSONResponse(content=mock_response)