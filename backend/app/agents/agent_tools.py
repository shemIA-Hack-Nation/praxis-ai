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
    
    # Create a metadata file for this notebook
    metadata = {
        "notebook_path": os.path.abspath(notebook_path),
        "notebook_name": Path(notebook_path).stem,
        "notebook_id": f"{Path(notebook_path).stem}_{hashlib.md5(os.path.abspath(notebook_path).encode()).hexdigest()[:8]}",
        "images_folder": images_folder,
        "total_cells": cell_index,
        "parsed_at": str(Path(notebook_path).stat().st_mtime)
    }
    
    # Save metadata file
    metadata_path = Path(images_folder) / "notebook_metadata.json"
    with open(metadata_path, 'w') as f:
        import json
        json.dump(metadata, f, indent=2)
    
    return structured_cells