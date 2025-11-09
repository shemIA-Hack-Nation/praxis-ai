import nbformat
from langchain.tools import tool
import base64
import os
import hashlib
from pathlib import Path
import uuid

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
        
        # Return the full path to the saved image
        return str(image_path)
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
                "content": cell.source
            })
            cell_index += 1
        elif cell.cell_type == 'code':
            structured_cells.append({
                "type": "code",
                "content": cell.source
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