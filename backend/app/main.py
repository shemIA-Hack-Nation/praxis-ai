# in backend/app/main.py

import os
from dotenv import load_dotenv # <-- Import load_dotenv
from fastapi import FastAPI
from app.api.v1.endpoints import api_router # <-- Your API routes will be imported here


load_dotenv() 

# Now, any module imported after this point (like your agents) 
# can safely access os.getenv("GEMINI_API_KEY")

# You can optionally check if it loaded:
if not os.getenv("GEMINI_API_KEY"):
    print("🚨 WARNING: GEMINI_API_KEY not loaded from .env!")

# Initialize the FastAPI app
app = FastAPI(
    title="Praxis-AI Research Agents",
    description="CrewAI agents for PaperGen and IndustryNav.",
    version="0.1.0",
)

# Include the API router
app.include_router(api_router, prefix="/api/v1")