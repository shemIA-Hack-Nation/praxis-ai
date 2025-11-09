"""
Papergen Agent Package

Multi-agent system for research paper generation from Jupyter notebooks.
Contains the orchestrator agent that coordinates specialized agents.
"""

from .orchestrator_papergen import PapergenOrchestrator, create_papergen_orchestrator

__all__ = ["PapergenOrchestrator", "create_papergen_orchestrator"]
