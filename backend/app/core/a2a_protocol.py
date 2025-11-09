"""
A2A (Agent-to-Agent) Communication Protocol.

Defines the standard protocol for agents to communicate with each other via REST APIs.
"""
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from enum import Enum
from datetime import datetime


class MessageType(str, Enum):
    """Types of A2A messages."""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    STATUS_UPDATE = "status_update"
    ERROR = "error"


class AgentStatus(str, Enum):
    """Agent status values."""
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


class A2AMessage(BaseModel):
    """Base A2A message structure."""
    message_id: str = Field(..., description="Unique message ID")
    message_type: MessageType = Field(..., description="Type of message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")
    sender_agent: str = Field(..., description="Name of sending agent")
    receiver_agent: str = Field(..., description="Name of receiving agent")
    payload: Dict[str, Any] = Field(..., description="Message payload")
    correlation_id: Optional[str] = Field(None, description="Correlation ID for request/response matching")


class A2ATaskRequest(BaseModel):
    """A2A task request structure."""
    task_id: str = Field(..., description="Unique task ID")
    task_type: str = Field(..., description="Type of task")
    input_data: Dict[str, Any] = Field(..., description="Input data for the task")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Task parameters")
    priority: int = Field(default=5, description="Task priority (1-10)")
    timeout: Optional[int] = Field(None, description="Task timeout in seconds")


class A2ATaskResponse(BaseModel):
    """A2A task response structure."""
    task_id: str = Field(..., description="Task ID this response corresponds to")
    status: AgentStatus = Field(..., description="Task status")
    result: Optional[Dict[str, Any]] = Field(None, description="Task result data")
    error: Optional[str] = Field(None, description="Error message if task failed")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class A2AAgentInfo(BaseModel):
    """Agent information structure."""
    agent_name: str = Field(..., description="Agent name")
    agent_type: str = Field(..., description="Agent type")
    status: AgentStatus = Field(..., description="Current agent status")
    capabilities: List[str] = Field(..., description="List of agent capabilities")
    endpoint: str = Field(..., description="Agent API endpoint")
    version: str = Field(default="1.0.0", description="Agent version")


# Agent-specific payload schemas

class ResearchExpandRequest(BaseModel):
    """Request schema for Research Expander agent."""
    research_context: Dict[str, Any] = Field(..., description="Research context from notebook")
    max_papers: int = Field(default=50, description="Maximum number of papers")
    min_papers: int = Field(default=20, description="Minimum number of papers")
    enable_full_text: bool = Field(default=False, description="Enable full text scraping")


class PaperParseRequest(BaseModel):
    """Request schema for Paper Parser agent."""
    paper_corpus: List[Dict[str, Any]] = Field(..., description="List of papers to parse")
    target_paper: Dict[str, Any] = Field(..., description="Target paper (research context)")
    enable_embeddings: bool = Field(default=True, description="Generate embeddings")
    enable_kg: bool = Field(default=True, description="Build knowledge graph")


class NoveltyAssessRequest(BaseModel):
    """Request schema for Novelty Assessor agent."""
    target_paper: Dict[str, Any] = Field(..., description="Target paper")
    parsed_data: Dict[str, Any] = Field(..., description="Parsed data from Paper Parser")
    k_neighbors: int = Field(default=10, description="Number of nearest neighbors")


class ReportGenerateRequest(BaseModel):
    """Request schema for Report Generator agent."""
    paper_data: Dict[str, Any] = Field(..., description="Paper data")
    research_context: Dict[str, Any] = Field(..., description="Research context")
    paper_corpus: List[Dict[str, Any]] = Field(..., description="Paper corpus")
    parsed_data: Dict[str, Any] = Field(..., description="Parsed data")
    novelty_result: Dict[str, Any] = Field(..., description="Novelty result")
