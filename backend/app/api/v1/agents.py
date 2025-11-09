"""
Agent REST APIs for A2A (Agent-to-Agent) communication.
"""
import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, List
import uuid
from datetime import datetime

from app.core.a2a_protocol import (
    A2ATaskRequest, A2ATaskResponse, A2AAgentInfo, AgentStatus,
    ResearchExpandRequest, PaperParseRequest, NoveltyAssessRequest,
    ReportGenerateRequest
)
from app.agents.industrynav.agent_research_expander import ResearchExpanderAgent
from app.agents.industrynav.agent_paper_parser import PaperParserAgent
from app.agents.industrynav.agent_novelty_assessor import NoveltyAssessorAgent
from app.agents.industrynav.agent_report_generator import ReportGeneratorAgent
from app.services.paper_scraper import PaperScraper

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["agents"])

# Lazy initialization - agents will be created on first use
research_expander = None
paper_parser = None
novelty_assessor = None
report_generator = None

def _ensure_agents_initialized():
    """Lazy initialization of agents."""
    global research_expander, paper_parser, novelty_assessor, report_generator
    if research_expander is None:
        research_expander = ResearchExpanderAgent()
    if paper_parser is None:
        paper_parser = PaperParserAgent()
    if novelty_assessor is None:
        novelty_assessor = NoveltyAssessorAgent()
    if report_generator is None:
        report_generator = ReportGeneratorAgent()

# Agent status tracking
agent_statuses: Dict[str, AgentStatus] = {
    "research-expander": AgentStatus.IDLE,
    "paper-parser": AgentStatus.IDLE,
    "novelty-assessor": AgentStatus.IDLE,
    "report-generator": AgentStatus.IDLE,
}


@router.get("/", response_model=List[A2AAgentInfo])
async def list_agents():
    """List all available agents."""
    return [
        A2AAgentInfo(
            agent_name="research-expander",
            agent_type="research",
            status=agent_statuses["research-expander"],
            capabilities=["paper_search", "corpus_building", "full_text_scraping"],
            endpoint="/api/v1/agents/research-expander"
        ),
        A2AAgentInfo(
            agent_name="paper-parser",
            agent_type="processing",
            status=agent_statuses["paper-parser"],
            capabilities=["concept_extraction", "embedding_generation", "kg_building"],
            endpoint="/api/v1/agents/paper-parser"
        ),
        A2AAgentInfo(
            agent_name="novelty-assessor",
            agent_type="analysis",
            status=agent_statuses["novelty-assessor"],
            capabilities=["novelty_scoring", "semantic_analysis", "gap_analysis"],
            endpoint="/api/v1/agents/novelty-assessor"
        ),
        A2AAgentInfo(
            agent_name="report-generator",
            agent_type="generation",
            status=agent_statuses["report-generator"],
            capabilities=["paper_generation", "analysis_generation", "report_formatting"],
            endpoint="/api/v1/agents/report-generator"
        ),
    ]


@router.get("/{agent_name}/status", response_model=A2AAgentInfo)
async def get_agent_status(agent_name: str):
    """Get status of a specific agent."""
    if agent_name not in agent_statuses:
        raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")
    
    agent_info_map = {
        "research-expander": {
            "agent_type": "research",
            "capabilities": ["paper_search", "corpus_building", "full_text_scraping"],
        },
        "paper-parser": {
            "agent_type": "processing",
            "capabilities": ["concept_extraction", "embedding_generation", "kg_building"],
        },
        "novelty-assessor": {
            "agent_type": "analysis",
            "capabilities": ["novelty_scoring", "semantic_analysis", "gap_analysis"],
        },
        "report-generator": {
            "agent_type": "generation",
            "capabilities": ["paper_generation", "analysis_generation", "report_formatting"],
        },
    }
    
    info = agent_info_map.get(agent_name, {})
    return A2AAgentInfo(
        agent_name=agent_name,
        agent_type=info.get("agent_type", "unknown"),
        status=agent_statuses[agent_name],
        capabilities=info.get("capabilities", []),
        endpoint=f"/api/v1/agents/{agent_name}"
    )


@router.post("/research-expander/expand", response_model=A2ATaskResponse)
async def research_expander_task(request: ResearchExpandRequest):
    """Research Expander agent task endpoint."""
    _ensure_agents_initialized()
    task_id = str(uuid.uuid4())
    agent_statuses["research-expander"] = AgentStatus.PROCESSING
    
    try:
        # Scrape full text if enabled
        paper_corpus = await research_expander.expand_research(
            request.research_context,
            max_papers=request.max_papers,
            min_papers=request.min_papers
        )
        
        if request.enable_full_text:
            async with PaperScraper() as scraper:
                paper_corpus = await scraper.scrape_papers_batch(paper_corpus, max_workers=5)
        
        agent_statuses["research-expander"] = AgentStatus.COMPLETED
        
        return A2ATaskResponse(
            task_id=task_id,
            status=AgentStatus.COMPLETED,
            result={"paper_corpus": paper_corpus},
            metadata={"papers_found": len(paper_corpus)}
        )
    
    except Exception as e:
        agent_statuses["research-expander"] = AgentStatus.ERROR
        logger.error(f"Error in research expander: {str(e)}", exc_info=True)
        return A2ATaskResponse(
            task_id=task_id,
            status=AgentStatus.ERROR,
            error=str(e)
        )
    finally:
        agent_statuses["research-expander"] = AgentStatus.IDLE


@router.post("/paper-parser/parse", response_model=A2ATaskResponse)
async def paper_parser_task(request: PaperParseRequest):
    """Paper Parser agent task endpoint."""
    _ensure_agents_initialized()
    task_id = str(uuid.uuid4())
    agent_statuses["paper-parser"] = AgentStatus.PROCESSING
    
    try:
        parsed_data = await paper_parser.parse_and_index(
            request.paper_corpus,
            request.target_paper
        )
        
        agent_statuses["paper-parser"] = AgentStatus.COMPLETED
        
        return A2ATaskResponse(
            task_id=task_id,
            status=AgentStatus.COMPLETED,
            result={"parsed_data": parsed_data},
            metadata={"papers_parsed": len(request.paper_corpus)}
        )
    
    except Exception as e:
        agent_statuses["paper-parser"] = AgentStatus.ERROR
        logger.error(f"Error in paper parser: {str(e)}", exc_info=True)
        return A2ATaskResponse(
            task_id=task_id,
            status=AgentStatus.ERROR,
            error=str(e)
        )
    finally:
        agent_statuses["paper-parser"] = AgentStatus.IDLE


@router.post("/novelty-assessor/assess", response_model=A2ATaskResponse)
async def novelty_assessor_task(request: NoveltyAssessRequest):
    """Novelty Assessor agent task endpoint."""
    _ensure_agents_initialized()
    task_id = str(uuid.uuid4())
    agent_statuses["novelty-assessor"] = AgentStatus.PROCESSING
    
    try:
        novelty_result = await novelty_assessor.assess_novelty(
            request.target_paper,
            request.parsed_data
        )
        
        agent_statuses["novelty-assessor"] = AgentStatus.COMPLETED
        
        return A2ATaskResponse(
            task_id=task_id,
            status=AgentStatus.COMPLETED,
            result={"novelty_result": novelty_result},
            metadata={"score": novelty_result.get("score", 0)}
        )
    
    except Exception as e:
        agent_statuses["novelty-assessor"] = AgentStatus.ERROR
        logger.error(f"Error in novelty assessor: {str(e)}", exc_info=True)
        return A2ATaskResponse(
            task_id=task_id,
            status=AgentStatus.ERROR,
            error=str(e)
        )
    finally:
        agent_statuses["novelty-assessor"] = AgentStatus.IDLE


@router.post("/report-generator/generate", response_model=A2ATaskResponse)
async def report_generator_task(request: ReportGenerateRequest):
    """Report Generator agent task endpoint."""
    _ensure_agents_initialized()
    task_id = str(uuid.uuid4())
    agent_statuses["report-generator"] = AgentStatus.PROCESSING
    
    try:
        reports = await report_generator.generate_reports(
            request.paper_data,
            request.research_context,
            request.paper_corpus,
            request.parsed_data,
            request.novelty_result
        )
        
        agent_statuses["report-generator"] = AgentStatus.COMPLETED
        
        return A2ATaskResponse(
            task_id=task_id,
            status=AgentStatus.COMPLETED,
            result={"reports": reports},
            metadata={"paper_length": len(reports.get("paper", "")), "analysis_length": len(reports.get("analysis", ""))}
        )
    
    except Exception as e:
        agent_statuses["report-generator"] = AgentStatus.ERROR
        logger.error(f"Error in report generator: {str(e)}", exc_info=True)
        return A2ATaskResponse(
            task_id=task_id,
            status=AgentStatus.ERROR,
            error=str(e)
        )
    finally:
        agent_statuses["report-generator"] = AgentStatus.IDLE
