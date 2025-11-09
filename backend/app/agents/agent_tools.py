import nbformat
from langchain.tools import tool
import base64
import os
import hashlib
from pathlib import Path
import uuid
from typing import Optional

def create_notebook_images_folder(notebook_path: str) -> str:
    """
    Create a dedicated folder for notebook images based on notebook name and hash.
    
    Args:
        notebook_path (str): Path to the notebook file
        
    Returns:
        str: Path to the created images folder
    """
    # Get notebook name without extension
    notebook_name = Path(notebook_path).stem
    
    # Create a unique identifier for this notebook based on its path
    notebook_hash = hashlib.md5(os.path.abspath(notebook_path).encode()).hexdigest()[:8]
    
    # Create images directory structure: data/notebook_images/{notebook_name}_{hash}/
    base_dir = Path("data/notebook_images")
    images_folder = base_dir / f"{notebook_name}_{notebook_hash}"
    
    # Create the directory if it doesn't exist
    images_folder.mkdir(parents=True, exist_ok=True)
    
    return str(images_folder)

def save_image_from_base64(base64_data: str, images_folder: str, cell_index: int, output_index: int) -> str:
    """
    Save a base64 encoded image to the images folder.
    
    Args:
        base64_data (str): Base64 encoded image data
        images_folder (str): Path to the images folder
        cell_index (int): Index of the cell containing the image
        output_index (int): Index of the output within the cell
        
    Returns:
        str: Relative path to the saved image file or empty string on error
    """
    # Generate filename: cell_{cell_index}_output_{output_index}_{unique_id}.png
    unique_id = str(uuid.uuid4())[:8]
    filename = f"cell_{cell_index}_output_{output_index}_{unique_id}.png"
    
    # Full path to save the image
    image_path = Path(images_folder) / filename
    
    # Decode base64 and save image
    try:
        image_bytes = base64.b64decode(base64_data)
        with open(image_path, 'wb') as f:
            f.write(image_bytes)
        
        # Return just the filename (not the full path)
        return filename
    except Exception as e:
        print(f"Error saving image: {e}")
        return ""

@tool("notebook_parser", return_direct=True)
def parse_notebook(notebook_path: str) -> list:
    """
    Parse a Jupyter notebook file and extract structured cell content.
    Extracts and saves images from output cells to a dedicated folder.
    
    Args:
        notebook_path (str): Path to the Jupyter notebook file (.ipynb)
        
    Returns:
        list: A list of dictionaries containing structured cell data with types:
              - markdown: Markdown content
              - code: Python code content
              - output_text: Text output from code execution
              - output_plot: Dictionary with image_path and metadata
    """
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
    except Exception as e:
        raise ValueError(f"Error reading or parsing the notebook file: {e}")

    # Create dedicated folder for this notebook's images
    images_folder = create_notebook_images_folder(notebook_path)
    
    structured_cells = []
    cell_index = 0
    
    for cell in nb.cells:
        if cell.cell_type == 'markdown':
            structured_cells.append({
                "type": "markdown",
                "content": cell.source,
                "source": cell.source  # Add source field for frontend compatibility
            })
            cell_index += 1
        elif cell.cell_type == 'code':
            structured_cells.append({
                "type": "code",
                "content": cell.source,
                "source": cell.source,  # Add source field for frontend compatibility
                "execution_count": cell.get('execution_count', None)
            })
            
            output_index = 0
            for output in cell.get('outputs', []):
                if output.output_type == 'stream':
                    structured_cells.append({
                        "type": "output_text",
                        "content": output.text
                    })
                elif output.output_type == 'execute_result':
                    if 'text/plain' in output.data:
                        structured_cells.append({
                            "type": "output_text",
                            "content": output.data['text/plain']
                        })
                elif output.output_type == 'display_data':
                    if 'image/png' in output.data:
                        # Extract image data
                        image_data = output.data['image/png']
                        if isinstance(image_data, str):
                            # Already base64 encoded
                            base64_image = image_data
                        else:
                            # If it's raw bytes
                            base64_image = base64.b64encode(image_data).decode('utf-8')
                        
                        # Save image to dedicated folder
                        saved_image_path = save_image_from_base64(
                            base64_image, images_folder, cell_index, output_index
                        )
                        
                        # Get notebook identifier for linking
                        notebook_name = Path(notebook_path).stem
                        notebook_hash = hashlib.md5(os.path.abspath(notebook_path).encode()).hexdigest()[:8]
                        notebook_id = f"{notebook_name}_{notebook_hash}"
                        
                        structured_cells.append({
                            "type": "output_plot",
                            "content": {
                                "image_path": saved_image_path,
                                "notebook_id": notebook_id,
                                "cell_index": cell_index,
                                "output_index": output_index,
                                "original_notebook": os.path.abspath(notebook_path),
                                "base64_data": base64_image  # Keep original for compatibility
                            }
                        })
                    elif 'text/html' in output.data:
                         structured_cells.append({
                            "type": "output_text",
                            "content": output.data['text/html']
                        })
                output_index += 1
            cell_index += 1
    
    # Create metadata for this notebook
    notebook_id = f"{Path(notebook_path).stem}_{hashlib.md5(os.path.abspath(notebook_path).encode()).hexdigest()[:8]}"
    metadata = {
        "notebook_path": os.path.abspath(notebook_path),
        "notebook_name": Path(notebook_path).stem,
        "notebook_id": notebook_id,
        "images_folder": images_folder,
        "total_cells": cell_index,
        "parsed_at": str(Path(notebook_path).stat().st_mtime)
    }
    
    # Count different cell types
    code_cells = sum(1 for cell in structured_cells if cell.get('type') == 'code')
    markdown_cells = sum(1 for cell in structured_cells if cell.get('type') == 'markdown')
    output_cells = sum(1 for cell in structured_cells if cell.get('type') in ['output_text', 'output_plot'])
    image_cells = sum(1 for cell in structured_cells if cell.get('type') == 'output_plot')
    
    # Create summary
    summary = {
        "total_elements": len(structured_cells),
        "code_cells": code_cells,
        "markdown_cells": markdown_cells,
        "output_cells": output_cells,
        "images_extracted": image_cells,
        "notebook_id": notebook_id
    }
    
    # Save basic metadata file
    metadata_path = Path(images_folder) / "notebook_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        import json
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Save complete parsed data including all cells
    complete_data = {
        "notebook_metadata": metadata,
        "cells": structured_cells,
        "summary": summary
    }
    
    parsed_data_path = Path(images_folder) / "notebook_parsed_data.json"
    with open(parsed_data_path, 'w', encoding='utf-8') as f:
        import json
        json.dump(complete_data, f, indent=2, ensure_ascii=False)
    
    return structured_cells


@tool("write_abstract", return_direct=True)
def write_abstract(synthesis_data_json: str) -> str:
    """
    Write an abstract section for a research paper.
    
    Args:
        synthesis_data_json (str): JSON string containing synthesis data with keys:
                                 - user_notes: User's notes and project description
                                 - method_draft: Methodology section content
                                 - results_draft: Results section content
    
    Returns:
        str: Generated abstract section formatted with section header
    """
    try:
        from backend.app.agents.papergen.agent_literary import LiteraryAgent
        import json
        
        synthesis_data = json.loads(synthesis_data_json)
        agent = LiteraryAgent()
        return agent.write_abstract(synthesis_data)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON format - {str(e)}"
    except Exception as e:
        return f"Error: Abstract generation failed - {str(e)}"

@tool("write_introduction", return_direct=True)
def write_introduction(synthesis_data_json: str) -> str:
    """
    Write an introduction section for a research paper.
    
    Args:
        synthesis_data_json (str): JSON string containing synthesis data with keys:
                                 - user_notes: User's notes and project description
                                 - method_draft: Methodology section content
                                 - results_draft: Results section content
    
    Returns:
        str: Generated introduction section formatted with section header
    """
    try:
        from backend.app.agents.papergen.agent_literary import LiteraryAgent
        import json
        
        synthesis_data = json.loads(synthesis_data_json)
        agent = LiteraryAgent()
        return agent.write_introduction(synthesis_data)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON format - {str(e)}"
    except Exception as e:
        return f"Error: Introduction generation failed - {str(e)}"

@tool("write_conclusion", return_direct=True)
def write_conclusion(synthesis_data_json: str) -> str:
    """
    Write a conclusion section for a research paper.
    
    Args:
        synthesis_data_json (str): JSON string containing synthesis data with keys:
                                 - user_notes: User's notes and project description
                                 - method_draft: Methodology section content
                                 - results_draft: Results section content
    
    Returns:
        str: Generated conclusion section formatted with section header
    """
    try:
        from backend.app.agents.papergen.agent_literary import LiteraryAgent
        import json
        
        synthesis_data = json.loads(synthesis_data_json)
        agent = LiteraryAgent()
        return agent.write_conclusion(synthesis_data)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON format - {str(e)}"
    except Exception as e:
        return f"Error: Conclusion generation failed - {str(e)}"

@tool("literary_agent_processor", return_direct=True)
def literary_agent_processor(task_description: str, synthesis_data_json: str) -> str:
    """
    Process literary writing tasks for research papers using the Literary Agent.
    
    This tool generates complete research paper reports or specific sections
    by synthesizing information from methodology and results sections.
    
    Args:
        task_description (str): Task description specifying what to generate:
                              - "Generate full report" or "Complete report" for full paper
                              - "Write ABSTRACT" for abstract only
                              - "Write INTRODUCTION" for introduction only  
                              - "Write CONCLUSION" for conclusion only
                              - "Write ABSTRACT and CONCLUSION" for both
        synthesis_data_json (str): JSON string containing synthesis data with keys:
                                 - method_draft: Methodology section content
                                 - results_draft: Results section content
    
    Returns:
        str: Generated literary content or complete research paper report
        
    Example:
        task = "Generate full report"
        data = '{"method_draft": "We used CNN...", "results_draft": "Achieved 91.5% accuracy..."}'
        result = literary_agent_processor(task, data)
    """
    try:
        from backend.app.agents.papergen.agent_literary import LiteraryAgent
        import json
        
        synthesis_data = json.loads(synthesis_data_json)
        agent = LiteraryAgent()
        return agent.process_task(task_description, synthesis_data)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON format - {str(e)}"
    except Exception as e:
        return f"Error: Literary Agent processing failed - {str(e)}"

@tool("synthesize_literary_sections", return_direct=True)
def synthesize_literary_sections(synthesis_data_json: str) -> str:
    """
    Generate all narrative sections (Abstract, Introduction, Conclusion) for a research paper.
    
    Args:
        synthesis_data_json (str): JSON string containing synthesis data with keys:
                                 - user_notes: User's notes and project description
                                 - method_draft: Methodology section content
                                 - results_draft: Results section content
    
    Returns:
        str: All generated sections formatted with section headers
    """
    try:
        from backend.app.agents.papergen.agent_literary import LiteraryAgent
        import json
        
        synthesis_data = json.loads(synthesis_data_json)
        agent = LiteraryAgent()
        sections = agent.synthesize_all_sections(synthesis_data)
        
        # Combine all sections
        result = f"{sections['abstract']}\n\n{sections['introduction']}\n\n{sections['conclusion']}"
        return result
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON format - {str(e)}"
    except Exception as e:
        return f"Error: Literary synthesis failed - {str(e)}"

# =============================================================================
# METHODOLOGY WRITER AGENT TOOLS
# =============================================================================

@tool("write_methodology", return_direct=True)
def write_methodology(notebook_json: str) -> str:
    """
    Generate methodology section by analyzing code cells and methodology descriptions.
    
    Args:
        notebook_json (str): JSON string of parsed notebook data containing cells
        
    Returns:
        str: Generated methodology section
    """
    try:
        from backend.app.agents.papergen.agent_methodology_writer import methodology_writer
        import json
        
        json_data = json.loads(notebook_json)
        return methodology_writer(json_data)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON format - {str(e)}"
    except Exception as e:
        return f"Error: Methodology writing failed - {str(e)}"

@tool("analyze_methodology_cells", return_direct=True)
def analyze_methodology_cells(notebook_json: str) -> str:
    """
    Identify and analyze cells relevant to methodology section.
    
    Args:
        notebook_json (str): JSON string of parsed notebook data
        
    Returns:
        str: Analysis of methodology-relevant content
    """
    try:
        import json
        
        json_data = json.loads(notebook_json)
        methodology_cells = []
        
        for i, cell in enumerate(json_data.get("cells", [])):
            cell_type = cell.get("type")
            cell_content_raw = cell.get("content", "")
            
            # Handle different content formats
            if isinstance(cell_content_raw, str):
                cell_content = cell_content_raw.strip()
            elif isinstance(cell_content_raw, list):
                cell_content = "\n".join(cell_content_raw).strip()
            elif isinstance(cell_content_raw, dict):
                cell_content = cell_content_raw.get("text/plain", "").strip()
            else:
                cell_content = ""
            
            # Check for methodology-relevant content
            if cell_type == "code":
                methodology_keywords = [
                    "import", "from", "load_data", "preprocess", "model", "sequential", 
                    "dense", "conv", "lstm", "dropout", "compile", "optimizer", 
                    "loss", "metric", "split", "train_test_split", "normalize",
                    "standardize", "reshape", "transform", "pipeline"
                ]
                if any(keyword in cell_content.lower() for keyword in methodology_keywords):
                    methodology_cells.append({
                        "index": i,
                        "type": cell_type,
                        "content": cell_content[:200] + "..." if len(cell_content) > 200 else cell_content
                    })
            
            elif cell_type == "markdown":
                methodology_keywords = [
                    "method", "approach", "algorithm", "model", "architecture",
                    "preprocessing", "feature", "parameter", "hyperparameter",
                    "training", "validation", "cross-validation", "technique"
                ]
                if any(keyword in cell_content.lower() for keyword in methodology_keywords):
                    methodology_cells.append({
                        "index": i,
                        "type": cell_type,
                        "content": cell_content[:200] + "..." if len(cell_content) > 200 else cell_content
                    })
        
        return f"Found {len(methodology_cells)} methodology-relevant cells: {methodology_cells}"
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON format - {str(e)}"
    except Exception as e:
        return f"Error: Methodology analysis failed - {str(e)}"

# =============================================================================
# RESULTS WRITER AGENT TOOLS
# =============================================================================

@tool("write_results", return_direct=True)
def write_results(notebook_json: str) -> str:
    """
    Generate results section by analyzing output cells and training logs.
    
    Args:
        notebook_json (str): JSON string of parsed notebook data containing cells
        
    Returns:
        str: Generated results section
    """
    try:
        from backend.app.agents.papergen.agent_results_writer import results_writer
        import json
        
        json_data = json.loads(notebook_json)
        return results_writer(json_data)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON format - {str(e)}"
    except Exception as e:
        return f"Error: Results writing failed - {str(e)}"

@tool("analyze_results_cells", return_direct=True)
def analyze_results_cells(notebook_json: str) -> str:
    """
    Identify and analyze cells relevant to results section.
    
    Args:
        notebook_json (str): JSON string of parsed notebook data
        
    Returns:
        str: Analysis of results-relevant content
    """
    try:
        import json
        
        json_data = json.loads(notebook_json)
        results_cells = []
        
        for i, cell in enumerate(json_data.get("cells", [])):
            cell_type = cell.get("type")
            cell_content_raw = cell.get("content", "")
            
            # Handle different content formats
            if isinstance(cell_content_raw, str):
                cell_content = cell_content_raw.strip()
            elif isinstance(cell_content_raw, list):
                cell_content = "\n".join(cell_content_raw).strip()
            elif isinstance(cell_content_raw, dict):
                cell_content = cell_content_raw.get("text/plain", "").strip()
            else:
                cell_content = ""
            
            # Check for results-relevant content
            if cell_type == "code":
                if "model.fit" in cell_content or ".fit(" in cell_content or \
                   "model.evaluate" in cell_content or ".evaluate(" in cell_content:
                    results_cells.append({
                        "index": i,
                        "type": cell_type,
                        "content": cell_content[:200] + "..." if len(cell_content) > 200 else cell_content
                    })
            
            elif cell_type == "output_text":
                # Filter out noise
                if not cell_content.startswith("<") and cell_content:
                    # Look for training/evaluation logs
                    if "Epoch" in cell_content or "val_accuracy" in cell_content or \
                       ("/" in cell_content and ("accuracy" in cell_content or "loss" in cell_content)):
                        results_cells.append({
                            "index": i,
                            "type": cell_type,
                            "content": cell_content[:200] + "..." if len(cell_content) > 200 else cell_content
                        })
        
        return f"Found {len(results_cells)} results-relevant cells: {results_cells}"
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON format - {str(e)}"
    except Exception as e:
        return f"Error: Results analysis failed - {str(e)}"

# =============================================================================
# ILLUSTRATION CRITIC AGENT TOOLS
# =============================================================================

@tool("generate_illustrations", return_direct=True)
def generate_illustrations(paper_draft: str) -> str:
    """
    Analyze paper draft and generate illustrations for visualization needs.
    
    Args:
        paper_draft (str): Complete or partial paper draft text
        
    Returns:
        str: Results of illustration generation with file paths and captions
    """
    try:
        from backend.app.agents.papergen.agent_illustration_critic import agent_illustration_critic
        return agent_illustration_critic(paper_draft)
    except Exception as e:
        return f"Error: Illustration generation failed - {str(e)}"

@tool("analyze_visualization_needs", return_direct=True)
def analyze_visualization_needs(text_content: str) -> str:
    """
    Analyze text content to identify sections that would benefit from visualizations.
    
    Args:
        text_content (str): Text content to analyze for visualization opportunities
        
    Returns:
        str: Analysis of visualization needs and suggestions
    """
    try:
        import re
        
        visualization_indicators = []
        
        # Check for architecture descriptions
        if re.search(r'\b(layer|network|architecture|model)\b.*\b(conv|dense|lstm|sequential)\b', text_content, re.IGNORECASE):
            visualization_indicators.append("Neural network architecture diagram needed")
        
        # Check for process descriptions
        if re.search(r'\b(step|process|algorithm|procedure)\b.*\b(then|next|followed by)\b', text_content, re.IGNORECASE):
            visualization_indicators.append("Process flowchart would be helpful")
        
        # Check for comparisons
        if re.search(r'\b(compare|comparison|versus|vs|better|worse|accuracy|performance)\b', text_content, re.IGNORECASE):
            visualization_indicators.append("Comparison chart could enhance understanding")
        
        # Check for time-based data
        if re.search(r'\b(epoch|time|week|day|iteration|training)\b.*\b(progress|improvement|change)\b', text_content, re.IGNORECASE):
            visualization_indicators.append("Time series plot would be valuable")
        
        # Check for correlation/relationship mentions
        if re.search(r'\b(correlation|relationship|matrix|dependent|independent)\b', text_content, re.IGNORECASE):
            visualization_indicators.append("Correlation heatmap could be useful")
        
        if visualization_indicators:
            return f"Visualization opportunities identified: {'; '.join(visualization_indicators)}"
        else:
            return "No clear visualization needs identified in this text"
            
    except Exception as e:
        return f"Error: Visualization analysis failed - {str(e)}"

# =============================================================================
# FORMATTER AGENT TOOLS
# =============================================================================

@tool("format_complete_paper", return_direct=True)
def format_complete_paper(sections_json: str, template: Optional[str] = None) -> str:
    """
    Assemble and format complete research paper from individual sections.
    
    Args:
        sections_json (str): JSON string containing all paper sections
        template (str): Optional paper template to use
        
    Returns:
        str: Complete formatted research paper
    """
    try:
        from backend.app.agents.papergen.agent_formatter import format_paper
        import json
        
        sections_data = json.loads(sections_json)
        return format_paper(sections_data, template)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON format - {str(e)}"
    except Exception as e:
        return f"Error: Paper formatting failed - {str(e)}"

@tool("validate_paper_structure", return_direct=True)
def validate_paper_structure(paper_content: str) -> str:
    """
    Validate and clean formatted paper content.
    
    Args:
        paper_content (str): Raw formatted paper content
        
    Returns:
        str: Validation report with cleaned content and issues
    """
    try:
        from backend.app.agents.papergen.agent_formatter import clean_and_validate_paper
        import json
        
        result = clean_and_validate_paper(paper_content)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: Paper validation failed - {str(e)}"

@tool("generate_paper_metadata", return_direct=True)
def generate_paper_metadata(sections_json: str) -> str:
    """
    Generate metadata for assembled paper.
    
    Args:
        sections_json (str): JSON string containing all paper sections
        
    Returns:
        str: JSON string containing paper metadata
    """
    try:
        from backend.app.agents.papergen.agent_formatter import generate_paper_metadata
        import json
        
        sections_data = json.loads(sections_json)
        metadata = generate_paper_metadata(sections_data)
        return json.dumps(metadata, indent=2)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON format - {str(e)}"
    except Exception as e:
        return f"Error: Metadata generation failed - {str(e)}"

# =============================================================================
# ENHANCED NOTEBOOK PROCESSING TOOLS
# =============================================================================

@tool("extract_code_snippets", return_direct=True)
def extract_code_snippets(notebook_json: str, language: str = "python") -> str:
    """
    Extract all code snippets from notebook for analysis.
    
    Args:
        notebook_json (str): JSON string of parsed notebook data
        language (str): Programming language to filter by
        
    Returns:
        str: JSON string of extracted code snippets with metadata
    """
    try:
        import json
        
        notebook_data = json.loads(notebook_json)
        code_snippets = []
        
        for i, cell in enumerate(notebook_data.get("cells", [])):
            if cell.get("type") == "code":
                cell_content = cell.get("content", "")
                
                # Handle different content formats
                if isinstance(cell_content, list):
                    code_text = "\n".join(cell_content)
                else:
                    code_text = str(cell_content)
                
                if code_text.strip():
                    code_snippets.append({
                        "cell_index": i,
                        "code": code_text,
                        "language": language,
                        "line_count": len(code_text.split('\n')),
                        "char_count": len(code_text)
                    })
        
        return json.dumps({
            "total_code_cells": len(code_snippets),
            "total_lines": sum(snippet["line_count"] for snippet in code_snippets),
            "snippets": code_snippets
        }, indent=2)
        
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON format - {str(e)}"
    except Exception as e:
        return f"Error: Code extraction failed - {str(e)}"

@tool("extract_output_results", return_direct=True)  
def extract_output_results(notebook_json: str) -> str:
    """
    Extract all output results from notebook for analysis.
    
    Args:
        notebook_json (str): JSON string of parsed notebook data
        
    Returns:
        str: JSON string of extracted outputs with metadata
    """
    try:
        import json
        import re
        
        notebook_data = json.loads(notebook_json)
        outputs = []
        
        for i, cell in enumerate(notebook_data.get("cells", [])):
            if cell.get("type") == "output_text":
                content = cell.get("content", "")
                
                if isinstance(content, list):
                    output_text = "\n".join(content)
                else:
                    output_text = str(content)
                
                if output_text.strip():
                    # Categorize output type
                    output_type = "general"
                    if "epoch" in output_text.lower():
                        output_type = "training_log"
                    elif re.search(r'\d+/\d+.*accuracy', output_text.lower()):
                        output_type = "evaluation_result"
                    elif "error" in output_text.lower() or "exception" in output_text.lower():
                        output_type = "error"
                    
                    outputs.append({
                        "cell_index": i,
                        "content": output_text,
                        "output_type": output_type,
                        "char_count": len(output_text),
                        "line_count": len(output_text.split('\n'))
                    })
        
        return json.dumps({
            "total_output_cells": len(outputs),
            "output_types": {
                "training_logs": len([o for o in outputs if o["output_type"] == "training_log"]),
                "evaluation_results": len([o for o in outputs if o["output_type"] == "evaluation_result"]),
                "errors": len([o for o in outputs if o["output_type"] == "error"]),
                "general": len([o for o in outputs if o["output_type"] == "general"])
            },
            "outputs": outputs
        }, indent=2)
        
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON format - {str(e)}"
    except Exception as e:
        return f"Error: Output extraction failed - {str(e)}"