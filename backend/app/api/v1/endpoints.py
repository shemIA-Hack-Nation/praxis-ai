from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import os
import tempfile
import json
from pathlib import Path
import sys
from typing import Dict, Any, List
import time
import asyncio

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from backend.app.agents.agent_tools import parse_notebook
except ImportError:
    # Fallback import
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from agents.agent_tools import parse_notebook

router = APIRouter()

@router.post("/upload_notebook_stream/")
async def upload_notebook_stream(file: UploadFile = File(...)):
    """
    Upload and parse a Jupyter notebook file with streaming response.
    
    Args:
        file: The uploaded .ipynb file
        
    Returns:
        Streaming response with progressive cell parsing
    """
    # Validate file type
    if not file.filename or not file.filename.endswith('.ipynb'):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Please upload a .ipynb file."
        )
    
    async def stream_notebook_parsing():
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.ipynb', delete=False) as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_notebook_path = temp_file.name
            
            # Send initial status
            yield f"data: {json.dumps({'status': 'starting', 'message': 'Starting notebook analysis...', 'progress': 0})}\n\n"
            await asyncio.sleep(0.5)
            
            # Parse notebook with streaming
            try:
                from backend.app.agents.agent_tools import parse_notebook_streaming
                
                async for update in parse_notebook_streaming(temp_notebook_path):
                    yield f"data: {json.dumps(update)}\n\n"
                    await asyncio.sleep(0.1)  # Small delay for visual effect
                    
            except ImportError:
                # Fallback to regular parsing with simulated streaming
                print(f"Parsing notebook: {temp_notebook_path}")
                parsed_cells = parse_notebook.run(temp_notebook_path)
                
                total_cells = len(parsed_cells)
                
                for i, cell in enumerate(parsed_cells):
                    progress = int((i + 1) / total_cells * 80)  # 80% for parsing
                    
                    update = {
                        'status': 'parsing',
                        'message': f'Processing cell {i + 1} of {total_cells}...',
                        'progress': progress,
                        'cell': cell,
                        'cell_index': i,
                        'total_cells': total_cells
                    }
                    
                    yield f"data: {json.dumps(update)}\n\n"
                    await asyncio.sleep(0.2)  # Simulate processing time
                
                # Final summary
                code_cells = sum(1 for cell in parsed_cells if cell.get('type') == 'code')
                markdown_cells = sum(1 for cell in parsed_cells if cell.get('type') == 'markdown')
                output_cells = sum(1 for cell in parsed_cells if cell.get('type') in ['output_text', 'output_plot'])
                
                final_update = {
                    'status': 'completed',
                    'message': f'Successfully parsed {total_cells} cells',
                    'progress': 100,
                    'cells': parsed_cells,
                    'summary': {
                        'total_elements': total_cells,
                        'code_cells': code_cells,
                        'markdown_cells': markdown_cells,
                        'output_cells': output_cells
                    }
                }
                
                yield f"data: {json.dumps(final_update)}\n\n"
                
        except Exception as e:
            error_update = {
                'status': 'error',
                'message': f'Failed to parse notebook: {str(e)}',
                'progress': 0,
                'error': str(e)
            }
            yield f"data: {json.dumps(error_update)}\n\n"
        
        finally:
            # Clean up temporary file
            if 'temp_notebook_path' in locals() and os.path.exists(temp_notebook_path):
                os.unlink(temp_notebook_path)
    
    return StreamingResponse(
        stream_notebook_parsing(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )

@router.post("/upload_notebook/")
async def upload_notebook(file: UploadFile = File(...)):
    """
    Upload and parse a Jupyter notebook file (original non-streaming version).
    
    Args:
        file: The uploaded .ipynb file
        
    Returns:
        Parsed notebook data with cells, metadata, and summary
    """
    # Validate file type
    if not file.filename or not file.filename.endswith('.ipynb'):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Please upload a .ipynb file."
        )
    
    try:
        # Create temporary file to store the uploaded notebook
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.ipynb', delete=False) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_notebook_path = temp_file.name
        
        # Parse the notebook using our agent tool
        try:
            print(f"Parsing notebook: {temp_notebook_path}")
            parsed_cells = parse_notebook.run(temp_notebook_path)
            print(f"Parsed cells type: {type(parsed_cells)}")
            print(f"Parsed cells length: {len(parsed_cells) if hasattr(parsed_cells, '__len__') else 'No length'}")
            if hasattr(parsed_cells, '__getitem__') and len(parsed_cells) > 0:
                print(f"First cell: {parsed_cells[0]}")
        except Exception as e:
            print(f"Parse error: {str(e)}")
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse notebook: {str(e)}"
            )
        finally:
            # Clean up temporary file
            if os.path.exists(temp_notebook_path):
                os.unlink(temp_notebook_path)
        
        # Load the complete parsed data if available
        complete_data = None
        try:
            # Try to find the notebook ID from parsed cells
            notebook_id = None
            for cell in parsed_cells:
                if cell.get('type') == 'output_plot' and isinstance(cell.get('content'), dict):
                    notebook_id = cell['content'].get('notebook_id')
                    break
            
            if notebook_id:
                parsed_data_path = Path(f"data/notebook_images/{notebook_id}/notebook_parsed_data.json")
                if parsed_data_path.exists():
                    with open(parsed_data_path, 'r', encoding='utf-8') as f:
                        complete_data = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load complete parsed data: {e}")
        
        # Structure the response
        if complete_data:
            response_data = {
                "success": True,
                "notebook_metadata": complete_data["notebook_metadata"],
                "cells": complete_data["cells"],
                "summary": complete_data["summary"],
                "message": f"Successfully parsed notebook with {len(complete_data['cells'])} cells"
            }
        else:
            # Fallback to basic structure
            code_cells = sum(1 for cell in parsed_cells if cell.get('type') == 'code')
            markdown_cells = sum(1 for cell in parsed_cells if cell.get('type') == 'markdown')
            output_cells = sum(1 for cell in parsed_cells if cell.get('type') in ['output_text', 'output_plot'])
            
            response_data = {
                "success": True,
                "cells": parsed_cells,
                "summary": {
                    "total_elements": len(parsed_cells),
                    "code_cells": code_cells,
                    "markdown_cells": markdown_cells,
                    "output_cells": output_cells
                },
                "message": f"Successfully parsed notebook with {len(parsed_cells)} elements"
            }
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.post("/process_notebook/")
async def process_notebook_with_orchestrator(notebook_data: Dict[str, Any]):
    """
    Process parsed notebook data through the orchestrator pipeline.
    
    This endpoint will be called after notebook parsing to generate the research paper.
    Currently returns a placeholder response for future orchestrator integration.
    
    Args:
        notebook_data: The parsed notebook data from upload_notebook
        
    Returns:
        Processed data ready for report generation
    """
    try:
        # Placeholder for orchestrator agent integration
        # TODO: Integrate with orchestrator agent when ready
        
        orchestrator_result = {
            "success": True,
            "message": "Orchestrator processing placeholder",
            "processed_data": {
                "methodology_sections": [],
                "results_sections": [],
                "figures": [],
                "analysis_complete": False
            },
            "next_steps": [
                "Extract methodology from code cells",
                "Identify results and plots", 
                "Generate research paper sections",
                "Format final report"
            ]
        }
        
        return JSONResponse(content=orchestrator_result)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Orchestrator processing failed: {str(e)}"
        )

@router.get("/notebook/{notebook_id}")
async def get_notebook_data(notebook_id: str):
    """
    Retrieve parsed notebook data by ID.
    
    Args:
        notebook_id: The unique notebook identifier
        
    Returns:
        Complete notebook data including metadata, cells, and images
    """
    try:
        parsed_data_path = Path(f"data/notebook_images/{notebook_id}/notebook_parsed_data.json")
        metadata_path = Path(f"data/notebook_images/{notebook_id}/notebook_metadata.json")
        
        if not parsed_data_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Notebook data not found for ID: {notebook_id}"
            )
        
        with open(parsed_data_path, 'r', encoding='utf-8') as f:
            complete_data = json.load(f)
        
        return JSONResponse(content={
            "success": True,
            "data": complete_data
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve notebook data: {str(e)}"
        )

@router.get("/notebooks/")
async def list_notebooks():
    """
    List all available parsed notebooks.
    
    Returns:
        List of notebook metadata for all parsed notebooks
    """
    try:
        notebooks = []
        data_dir = Path("data/notebook_images")
        
        if data_dir.exists():
            for notebook_folder in data_dir.iterdir():
                if notebook_folder.is_dir():
                    metadata_path = notebook_folder / "notebook_metadata.json"
                    if metadata_path.exists():
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                            notebooks.append(metadata)
        
        return JSONResponse(content={
            "success": True,
            "notebooks": notebooks,
            "count": len(notebooks)
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list notebooks: {str(e)}"
        )

# Papergen Orchestrator endpoint with streaming agent workflow
@router.post("/generate_paper_stream/")
async def generate_paper_stream(file: UploadFile = File(...)):
    """
    Generate research paper using orchestrator workflow with streaming agent calls.
    
    Args:
        file: The uploaded .ipynb file
        
    Returns:
        Streaming response showing each agent call explicitly
    """
    # Validate file type
    if not file.filename or not file.filename.endswith('.ipynb'):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Please upload a .ipynb file."
        )
    
    async def stream_paper_generation():
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.ipynb', delete=False) as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_notebook_path = temp_file.name
            
            # Phase 1: Parse Notebook
            yield f"data: {json.dumps({'phase': 'notebook_parsing', 'agent': 'NotebookParser', 'status': 'starting', 'message': '🔍 Starting notebook analysis...', 'progress': 5})}\n\n"
            await asyncio.sleep(0.5)
            
            try:
                from backend.app.agents.agent_tools import parse_notebook
                parsed_cells = parse_notebook(temp_notebook_path)
                yield f"data: {json.dumps({'phase': 'notebook_parsing', 'agent': 'NotebookParser', 'status': 'completed', 'message': f'✅ Parsed {len(parsed_cells)} cells successfully', 'progress': 15, 'data': {'cell_count': len(parsed_cells)}})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'phase': 'notebook_parsing', 'agent': 'NotebookParser', 'status': 'error', 'message': f'❌ Notebook parsing failed: {str(e)}', 'progress': 15})}\n\n"
                return
            
            await asyncio.sleep(0.5)
            
            # Phase 2: Initialize Orchestrator
            yield f"data: {json.dumps({'phase': 'orchestrator_init', 'agent': 'PapergenOrchestrator', 'status': 'starting', 'message': '🎯 Initializing multi-agent orchestrator...', 'progress': 20})}\n\n"
            await asyncio.sleep(0.5)
            
            try:
                from backend.app.agents.papergen.orchestrator_papergen import create_papergen_orchestrator
                orchestrator = create_papergen_orchestrator()
                
                # Check which agent tools are available (consolidated tools approach)
                agents_status = {
                    'LiteraryAgent': orchestrator.synthesize_literary_sections is not None,
                    'IllustrationCritic': orchestrator.generate_illustrations is not None,
                    'MethodologyWriter': orchestrator.write_methodology is not None,
                    'ResultsWriter': orchestrator.write_results is not None,
                    'FormatterAgent': orchestrator.format_complete_paper is not None
                }
                
                available_agents = sum(agents_status.values())
                yield f"data: {json.dumps({'phase': 'orchestrator_init', 'agent': 'PapergenOrchestrator', 'status': 'completed', 'message': f'✅ Orchestrator ready with {available_agents}/5 agents', 'progress': 25, 'data': {'agents_status': agents_status}})}\n\n"
                
            except Exception as e:
                yield f"data: {json.dumps({'phase': 'orchestrator_init', 'agent': 'PapergenOrchestrator', 'status': 'error', 'message': f'❌ Orchestrator initialization failed: {str(e)}', 'progress': 25})}\n\n"
                return
            
            await asyncio.sleep(0.5)
            
            # Phase 3: Start Paper Generation Workflow
            yield f"data: {json.dumps({'phase': 'workflow_start', 'agent': 'PapergenOrchestrator', 'status': 'starting', 'message': '🚀 Starting multi-agent paper generation workflow...', 'progress': 30})}\n\n"
            await asyncio.sleep(0.5)
            
            # Load template
            template_path = Path("templates/research_paper_template.md")
            if template_path.exists():
                template_content = template_path.read_text()
            else:
                template_content = """
                # Research Paper Template
                
                [ABSTRACT]
                
                ## Introduction
                [INTRODUCTION]
                
                ## Methodology
                [METHODOLOGY]
                
                ## Results
                [RESULTS]
                
                ## Conclusion
                [CONCLUSION]
                """
            
            # Generate paper with detailed workflow streaming
            try:
                result = await orchestrator.generate_paper(
                    notebook_cells=parsed_cells,
                    template=template_content,
                    user_requirements={"style": "academic"}
                )
                
                # Stream each workflow stage based on messages
                current_progress = 35
                progress_per_stage = 60 / len(result.get('messages', []))
                
                for message in result.get('messages', []):
                    agent_name = message.get('agent', 'Unknown')
                    stage = message.get('stage', '')
                    msg_content = message.get('message', message.get('error', ''))
                    is_error = 'error' in message
                    
                    # Map stage names to user-friendly names
                    stage_names = {
                        'analysis': 'Notebook Analysis',
                        'task_assignment': 'Task Assignment', 
                        'methodology': 'Methodology Generation',
                        'results': 'Results Analysis',
                        'literary': 'Literary Writing',
                        'illustrations': 'Illustration Review',
                        'assembly': 'Paper Assembly',
                        'quality_check': 'Quality Validation'
                    }
                    
                    stage_display = stage_names.get(stage, stage.replace('_', ' ').title())
                    
                    # Determine agent display name
                    agent_display = {
                        'orchestrator': 'PapergenOrchestrator',
                        'methodology_writer': 'MethodologyWriter',
                        'results_writer': 'ResultsWriter', 
                        'literary_agent': 'LiteraryAgent',
                        'illustration_critic': 'IllustrationCritic',
                        'formatter': 'FormatterAgent'
                    }.get(agent_name, agent_name)
                    
                    status = 'error' if is_error else 'working'
                    
                    yield f"data: {json.dumps({'phase': 'agent_workflow', 'agent': agent_display, 'stage': stage_display, 'status': status, 'message': msg_content, 'progress': int(current_progress)})}\n\n"
                    
                    current_progress += progress_per_stage
                    await asyncio.sleep(0.3)
                
                # Final results
                if result.get('success'):
                    latex_file = result.get('latex_file', '')
                    pdf_file = result.get('pdf_file', '')
                    
                    # Create download URLs for the files
                    latex_url = f"/reports/{Path(latex_file).name}" if latex_file else ""
                    pdf_url = f"/reports/{Path(pdf_file).name}" if pdf_file else ""
                    
                    final_message = f"✅ Paper generation completed! Generated {len(result.get('sections', {}))} sections with {len(result.get('illustrations', []))} illustrations."
                    if pdf_file:
                        final_message += f" PDF compiled successfully."
                    elif latex_file:
                        final_message += f" LaTeX file generated (PDF compilation unavailable)."
                    
                    yield f"data: {json.dumps({'phase': 'completion', 'agent': 'PapergenOrchestrator', 'status': 'completed', 'message': final_message, 'progress': 100, 'data': {'paper_content': result.get('paper_content', ''), 'sections': result.get('sections', {}), 'illustrations': result.get('illustrations', []), 'latex_file': latex_file, 'pdf_file': pdf_file, 'latex_url': latex_url, 'pdf_url': pdf_url, 'metadata': result.get('metadata', {})}})}\n\n"
                else:
                    error_msg = f"❌ Paper generation failed. Errors: {len(result.get('errors', []))}"
                    yield f"data: {json.dumps({'phase': 'completion', 'agent': 'PapergenOrchestrator', 'status': 'error', 'message': error_msg, 'progress': 100, 'data': {'errors': result.get('errors', []), 'warnings': result.get('warnings', [])}})}\n\n"
                
            except Exception as e:
                yield f"data: {json.dumps({'phase': 'workflow_error', 'agent': 'PapergenOrchestrator', 'status': 'error', 'message': f'❌ Workflow execution failed: {str(e)}', 'progress': 100})}\n\n"
            
            # Cleanup
            try:
                os.unlink(temp_notebook_path)
            except:
                pass
                
        except Exception as e:
            yield f"data: {json.dumps({'phase': 'fatal_error', 'agent': 'System', 'status': 'error', 'message': f'❌ Fatal error: {str(e)}', 'progress': 0})}\n\n"
    
    return StreamingResponse(
        stream_paper_generation(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*"
        }
    )


# Industry Navigation endpoint (placeholder for future implementation)
# try:
#     from backend.app.agents.industrynav.orchestrator_sprint2 import run_industry_nav_crew
#     
#     @router.post("/run-industry-nav/")
#     async def run_industry_nav(file: UploadFile = File(...)):
#         """Run Industry Navigation analysis on uploaded paper content."""
#         # Implementation placeholder
#         pass
# except ImportError:
#     print("⚠️ Industry Navigation orchestrator not available")
