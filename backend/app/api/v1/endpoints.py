# in backend/app/api/v1/endpoints.py
from fastapi import APIRouter, File, UploadFile
from app.agents.industrynav.orchestrator_sprint2 import run_industry_nav_crew

router = APIRouter()

@router.post("/run-industry-nav/")
async def run_industry_nav(file: UploadFile = File(...)):
    
    # 1. Get content from uploaded file
    paper_content_bytes = await file.read()
    paper_content_string = paper_content_bytes.decode('utf-8') 
    # (You'll need a real PDF/text parser here, maybe in 'pipeline_manager.py')
    
    

    # 2. Call YOUR orchestrator function!
    report = run_industry_nav_crew(paper_content=paper_content_string)
    
    # 3. Return the final report
    return {"report": report}

