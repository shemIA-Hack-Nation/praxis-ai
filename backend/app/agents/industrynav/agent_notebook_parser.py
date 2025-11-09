"""
Notebook parser for extracting research content from Jupyter notebooks.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

import nbformat
from nbconvert import PythonExporter

logger = logging.getLogger(__name__)


class NotebookParser:
    """Parser for Jupyter notebook files."""
    
    def __init__(self):
        """Initialize notebook parser."""
        self.exporter = PythonExporter()
    
    async def parse_notebook(self, notebook_path: str) -> Dict[str, Any]:
        """
        Parse a Jupyter notebook file.
        
        Args:
            notebook_path: Path to the notebook file
        
        Returns:
            Dictionary containing notebook structure and content
        """
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook = nbformat.read(f, as_version=4)
            
            # Extract cells content
            cells_data = []
            for cell in notebook.cells:
                cell_data = {
                    "cell_type": cell.cell_type,
                    "source": cell.source,
                }
                if cell.cell_type == "code":
                    cell_data["outputs"] = [str(output) for output in cell.outputs]
                cells_data.append(cell_data)
            
            return {
                "notebook": notebook,
                "cells": cells_data,
                "metadata": notebook.metadata
            }
        
        except Exception as e:
            logger.error(f"Error parsing notebook: {str(e)}")
            raise
    
    async def extract_research_context(self, notebook_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract research context from notebook (title, abstract, methods, results).
        
        Args:
            notebook_data: Parsed notebook data
        
        Returns:
            Dictionary with research context
        """
        cells = notebook_data["cells"]
        
        # Extract markdown cells for title and description
        markdown_content = []
        code_content = []
        
        for cell in cells:
            if cell["cell_type"] == "markdown":
                markdown_content.append(cell["source"])
            elif cell["cell_type"] == "code":
                code_content.append(cell["source"])
        
        # Try to extract title from first markdown cell
        title = "Research Notebook"
        if markdown_content:
            first_line = markdown_content[0].split('\n')[0]
            if first_line.startswith('# '):
                title = first_line[2:].strip()
            elif first_line.startswith('## '):
                title = first_line[3:].strip()
        
        # Combine all markdown as abstract/description
        abstract = "\n".join(markdown_content[:3])  # First 3 markdown cells
        
        # Extract methods from code cells
        methods = "\n".join(code_content)
        
        return {
            "title": title,
            "abstract": abstract,
            "methods": methods,
            "full_content": "\n\n".join(markdown_content + code_content),
            "markdown_cells": markdown_content,
            "code_cells": code_content
        }
