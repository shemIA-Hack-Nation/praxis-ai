import sys
import os
import json
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from backend.app.agents.agent_tools import parse_notebook
except ImportError:
    # Fallback: try direct import
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from agents.agent_tools import parse_notebook

def get_notebook_images_by_id(notebook_id: str):
    """
    Get all image paths for a specific notebook by its ID.
    
    Args:
        notebook_id (str): The notebook ID (e.g., "plant-recognition_a1b2c3d4")
        
    Returns:
        dict: Contains image paths and metadata for the notebook
    """
    images_folder = Path(f"data/notebook_images/{notebook_id}")
    
    if not images_folder.exists():
        return {"error": f"No images found for notebook ID: {notebook_id}"}
    
    # Read metadata
    metadata_path = images_folder / "notebook_metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    # Get all image files
    image_files = list(images_folder.glob("*.png"))
    
    return {
        "notebook_id": notebook_id,
        "images_folder": str(images_folder),
        "metadata": metadata,
        "image_files": [str(img) for img in image_files],
        "image_count": len(image_files)
    }

def get_notebook_parser_agent():
    """Returns the notebook parser tool function directly."""
    return parse_notebook

def run_notebook_parser(notebook_path: str):
    """Parse a notebook file and return structured cell data with image extraction."""
    parser_tool = get_notebook_parser_agent()
    
    # Call the tool with the notebook path
    result = parser_tool.run(notebook_path)
    
    # Count extracted images for summary
    image_count = sum(1 for cell in result if cell.get('type') == 'output_plot')
    
    if image_count > 0:
        print(f"\n📸 Extracted {image_count} images from notebook outputs")
        
        # Show first image info as example
        first_image = next((cell for cell in result if cell.get('type') == 'output_plot'), None)
        if first_image and 'content' in first_image:
            content = first_image['content']
            if isinstance(content, dict) and 'notebook_id' in content:
                print(f"📁 Images saved to: data/notebook_images/{content['notebook_id']}/")
                print(f"🔗 Notebook ID for linking: {content['notebook_id']}")
    
    return result

if __name__ == '__main__':
    import sys
    import os
    import json

    if len(sys.argv) > 1:
        notebook_path = sys.argv[1]
        if os.path.exists(notebook_path):
            print(f"📖 Parsing notebook: {notebook_path}")
            try:
                parsed_content = run_notebook_parser(os.path.abspath(notebook_path))
                
                # Show summary
                total_cells = len(parsed_content)
                code_cells = sum(1 for cell in parsed_content if cell.get('type') == 'code')
                markdown_cells = sum(1 for cell in parsed_content if cell.get('type') == 'markdown')
                output_cells = sum(1 for cell in parsed_content if cell.get('type') in ['output_text', 'output_plot'])
                
                print(f"\n✅ Parsing completed!")
                print(f"📊 Summary: {total_cells} total elements")
                print(f"   - {markdown_cells} markdown cells")
                print(f"   - {code_cells} code cells") 
                print(f"   - {output_cells} output elements")
                
                # Optionally show parsed content (commented out for brevity)
                # print("\nParsed Content:")
                # print(json.dumps(parsed_content, indent=2))
                
            except Exception as e:
                print(f"❌ An error occurred: {e}")
        else:
            print(f"Error: Notebook file not found at '{notebook_path}'")
    else:
        print("Usage: python -m backend.app.agents.papergen.agent_notebook_parser <path_to_notebook>.ipynb")

