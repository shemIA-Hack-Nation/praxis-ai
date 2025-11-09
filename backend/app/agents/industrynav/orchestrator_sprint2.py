"""
Orchestrator for Sprint 2: Coordinates all agents in the research analysis pipeline.
"""
import logging
from typing import Dict, Any
from pathlib import Path

from app.services.pipeline_manager import PipelineManager

logger = logging.getLogger(__name__)


class OrchestratorSprint2:
    """Orchestrator for the research analysis pipeline."""
    
    def __init__(self):
        """Initialize orchestrator."""
        self.pipeline_manager = PipelineManager()
    
    async def process_notebook(self, notebook_path: str) -> Dict[str, str]:
        """
        Process a notebook through the entire pipeline.
        
        This is the main entry point for the orchestrator.
        It delegates to the PipelineManager which coordinates all agents.
        
        Args:
            notebook_path: Path to the Jupyter notebook file
        
        Returns:
            Dictionary with 'paper' and 'analysis' keys containing markdown content
        """
        logger.info(f"Orchestrator: Starting processing for {notebook_path}")
        
        try:
            # Validate notebook path
            if not Path(notebook_path).exists():
                raise FileNotFoundError(f"Notebook not found: {notebook_path}")
            
            # Process through pipeline
            result = await self.pipeline_manager.process_notebook(notebook_path)
            
            logger.info("Orchestrator: Processing completed successfully")
            return result
        
        except Exception as e:
            logger.error(f"Orchestrator: Error processing notebook: {str(e)}", exc_info=True)
            raise


# Convenience function for direct usage
async def process_research_notebook(notebook_path: str) -> Dict[str, str]:
    """
    Convenience function to process a research notebook.
    
    Args:
        notebook_path: Path to the Jupyter notebook file
    
    Returns:
        Dictionary with 'paper' and 'analysis' keys
    """
    orchestrator = OrchestratorSprint2()
    return await orchestrator.process_notebook(notebook_path)
