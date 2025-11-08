import nbformat
from langchain.tools import tool
import base64

@tool("notebook_parser", return_direct=True)
def parse_notebook(notebook_path: str) -> list:
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
    except Exception as e:
        raise ValueError(f"Error reading or parsing the notebook file: {e}")

    structured_cells = []
    for cell in nb.cells:
        if cell.cell_type == 'markdown':
            structured_cells.append({
                "type": "markdown",
                "content": cell.source
            })
        elif cell.cell_type == 'code':
            structured_cells.append({
                "type": "code",
                "content": cell.source
            })
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
                        # Encode image data to base64
                        image_data = output.data['image/png']
                        if isinstance(image_data, str):
                            # Already base64 encoded
                            base64_image = image_data
                        else:
                            # If it's raw bytes
                            base64_image = base64.b64encode(image_data).decode('utf-8')
                        
                        structured_cells.append({
                            "type": "output_plot",
                            "content": base64_image
                        })
                    elif 'text/html' in output.data:
                         structured_cells.append({
                            "type": "output_text",
                            "content": output.data['text/html']
                        })
    return structured_cells