from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import tempfile
import json
from pathlib import Path
import sys

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.app.agents.agent_tools import parse_notebook
from backend.app.api.v1.endpoints import router as api_router
from backend.app.api.v1.test_endpoints import router as test_router

app = FastAPI(
    title="Praxis AI Backend",
    description="Backend API for research paper generation from Jupyter notebooks",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:8080"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")
app.include_router(test_router, prefix="/api/v1/test")

# Serve static files for notebook images
data_images_path = Path("data/notebook_images")
data_images_path.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=str(data_images_path)), name="images")

@app.get("/")
async def root():
    return {"message": "Praxis AI Backend API", "status": "active"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "praxis-ai-backend"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
