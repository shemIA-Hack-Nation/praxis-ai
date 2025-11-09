from typing import TypedDict, List, Dict, Any, Optional, Union
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
import json
import asyncio

class PapergenState(TypedDict):
    """State for the papergen orchestrator workflow"""
    # Input data
    notebook_cells: List[Dict[str, Any]]
    template_content: str
    user_requirements: Dict[str, Any]
    
    # Processing state
    current_stage: str
    progress: float
    task_assignments: Dict[str, Any]
    
    # Agent outputs
    methodology_draft: str
    results_draft: str
    literary_sections: Dict[str, str]  # abstract, introduction, conclusion
    illustrations: List[Dict[str, Any]]
    
    # Final output
    assembled_paper: str
    generation_complete: bool
    
    # Error handling
    errors: List[str]
    warnings: List[str]
    
    # Agent communication
    messages: List[Dict[str, Any]]

class PapergenOrchestrator:
    """
    Orchestrator Agent for managing the research paper generation workflow.
    Acts as Editor-in-Chief coordinating multiple specialist agents.
    
    Coordinates:
    - NotebookParser (already handled upstream)
    - MethodologyWriter Agent
    - ResultsWriter Agent  
    - LiteraryAgent
    - IllustrationCritic Agent
    - Formatter Agent
    """
    
    def __init__(self):
        self.workflow = self._build_workflow()
        
        # Agent instances (to be implemented)
        self.methodology_writer = None  # MethodologyWriter()
        self.results_writer = None      # ResultsWriter() 
        
        # Import and instantiate LiteraryAgent
        try:
            from .agent_literary import LiteraryAgent
        except ImportError:
            from agent_literary import LiteraryAgent
        self.literary_agent = LiteraryAgent()
        
        self.illustration_critic = None # IllustrationCritic()
        self.formatter_agent = None     # FormatterAgent()
        
    def _build_workflow(self):
        """Build the LangGraph workflow for paper generation"""
        
        workflow = StateGraph(PapergenState)
        
        # Add nodes for each stage
        workflow.add_node("analyze_notebook", self._analyze_notebook)
        workflow.add_node("assign_tasks", self._assign_tasks)
        workflow.add_node("methodology_writing", self._coordinate_methodology)
        workflow.add_node("results_writing", self._coordinate_results)
        workflow.add_node("literary_writing", self._coordinate_literary)
        workflow.add_node("illustration_review", self._coordinate_illustrations)
        workflow.add_node("final_assembly", self._coordinate_assembly)
        workflow.add_node("quality_check", self._quality_check)
        
        # Define the workflow flow
        workflow.set_entry_point("analyze_notebook")
        
        workflow.add_edge("analyze_notebook", "assign_tasks")
        workflow.add_edge("assign_tasks", "methodology_writing")
        workflow.add_edge("methodology_writing", "results_writing")
        workflow.add_edge("results_writing", "literary_writing")
        workflow.add_edge("literary_writing", "illustration_review")
        workflow.add_edge("illustration_review", "final_assembly")
        workflow.add_edge("final_assembly", "quality_check")
        workflow.add_edge("quality_check", END)
        
        return workflow.compile()
    
    def _analyze_notebook(self, state: PapergenState) -> PapergenState:
        """
        Analyze the parsed notebook cells to understand structure and content.
        Categorizes cells and identifies their roles in the research paper.
        """
        state["current_stage"] = "analyzing_notebook"
        state["progress"] = 10.0
        
        try:
            # Categorize cells by type and purpose
            code_cells = [cell for cell in state["notebook_cells"] if cell.get("type") == "code"]
            markdown_cells = [cell for cell in state["notebook_cells"] if cell.get("type") == "markdown"]
            output_text_cells = [cell for cell in state["notebook_cells"] if cell.get("type") == "output_text"]
            output_plot_cells = [cell for cell in state["notebook_cells"] if cell.get("type") == "output_plot"]
            
            # Analyze code cells for methodology content
            methodology_cells = self._identify_methodology_cells(code_cells)
            
            # Analyze outputs for results content
            results_cells = output_text_cells + output_plot_cells
            
            # Extract user objectives from markdown cells
            user_objectives = self._extract_user_objectives(markdown_cells)
            
            # Store analysis results
            state["task_assignments"] = {
                "methodology_cells": methodology_cells,
                "results_cells": results_cells, 
                "user_objectives": user_objectives,
                "total_cells": len(state["notebook_cells"]),
                "analysis": {
                    "code_count": len(code_cells),
                    "markdown_count": len(markdown_cells),
                    "output_text_count": len(output_text_cells),
                    "output_plot_count": len(output_plot_cells)
                }
            }
            
            state["messages"].append({
                "agent": "orchestrator",
                "stage": "analysis",
                "message": f"📊 Analysis complete: {len(code_cells)} code, {len(markdown_cells)} markdown, {len(results_cells)} outputs"
            })
            
        except Exception as e:
            error_msg = f"Notebook analysis failed: {str(e)}"
            state["errors"].append(error_msg)
            state["messages"].append({
                "agent": "orchestrator", 
                "stage": "analysis",
                "error": error_msg
            })
            
        return state
    
    def _identify_methodology_cells(self, code_cells: List[Dict]) -> List[Dict]:
        """Identify code cells relevant to methodology description"""
        methodology_cells = []
        
        methodology_keywords = [
            "model", "Sequential", "Dense", "Conv", "LSTM", "train", "compile",
            "fit", "sklearn", "algorithm", "pipeline", "preprocessing"
        ]
        
        for cell in code_cells:
            content = cell.get("content", "") + cell.get("source", "")
            if any(keyword in content for keyword in methodology_keywords):
                methodology_cells.append(cell)
                
        return methodology_cells
    
    def _extract_user_objectives(self, markdown_cells: List[Dict]) -> Dict[str, str]:
        """Extract user objectives and context from markdown cells"""
        objectives = {
            "title": "",
            "description": "",
            "goals": ""
        }
        
        for cell in markdown_cells:
            content = cell.get("content", "") + cell.get("source", "")
            
            # Simple heuristic to identify title (first h1 heading)
            if content.startswith("#") and not objectives["title"]:
                objectives["title"] = content.split("\n")[0].replace("#", "").strip()
            
            # Accumulate description from markdown content
            if content and not content.startswith("#"):
                objectives["description"] += content + "\n"
                
        return objectives
    
    def _assign_tasks(self, state: PapergenState) -> PapergenState:
        """
        Assign specific tasks to specialist agents based on template and content.
        Creates detailed task specifications for each agent.
        """
        state["current_stage"] = "assigning_tasks"
        state["progress"] = 20.0
        
        try:
            template = state["template_content"]
            assignments = state["task_assignments"]
            
            # Parse template for required sections
            required_sections = self._parse_template_sections(template)
            
            # Create task assignments for each agent
            task_specs = {
                "methodology_writer": {
                    "task": "Generate methodology section from code cells",
                    "input_cells": assignments["methodology_cells"],
                    "target_sections": [s for s in required_sections if "method" in s.lower()],
                    "requirements": "Translate code into formal methodology prose"
                },
                "results_writer": {
                    "task": "Generate results section from outputs and plots", 
                    "input_cells": assignments["results_cells"],
                    "target_sections": [s for s in required_sections if "result" in s.lower()],
                    "requirements": "Interpret plots and outputs into results narrative"
                },
                "literary_agent": {
                    "task": "Generate high-level narrative sections",
                    "input_data": {
                        "user_objectives": assignments["user_objectives"],
                        "methodology_preview": "Will receive from MethodologyWriter",
                        "results_preview": "Will receive from ResultsWriter"
                    },
                    "target_sections": ["abstract", "introduction", "conclusion"],
                    "requirements": "Synthesize coherent narrative from all components"
                },
                "illustration_critic": {
                    "task": "Review and enhance visual content",
                    "input_data": "Full draft from other agents",
                    "requirements": "Identify gaps in visual content and generate new illustrations"
                },
                "formatter": {
                    "task": "Assemble final formatted paper",
                    "input_data": "All section drafts and illustrations",
                    "requirements": "Format according to template with proper citations and structure"
                }
            }
            
            state["task_assignments"]["agent_tasks"] = task_specs
            
            state["messages"].append({
                "agent": "orchestrator",
                "stage": "task_assignment", 
                "message": f"🎯 Tasks assigned to {len(task_specs)} specialist agents"
            })
            
        except Exception as e:
            error_msg = f"Task assignment failed: {str(e)}"
            state["errors"].append(error_msg)
            state["messages"].append({
                "agent": "orchestrator",
                "stage": "task_assignment",
                "error": error_msg
            })
            
        return state
    
    def _parse_template_sections(self, template: str) -> List[str]:
        """Parse template to identify required sections"""
        import re
        
        # Find placeholders like [METHODOLOGY], [RESULTS], etc.
        placeholders = re.findall(r'\[([^\]]+)\]', template)
        
        # Common section patterns
        section_patterns = [
            "abstract", "introduction", "methodology", "method", 
            "results", "discussion", "conclusion", "references"
        ]
        
        sections = placeholders + section_patterns
        return list(set(s.lower() for s in sections))
    
    def _coordinate_methodology(self, state: PapergenState) -> PapergenState:
        """
        Coordinate with MethodologyWriter agent.
        
        TASK: @MethodologyWriter, use cells [c2, c3] to write the [METHODOLOGY] section.
        """
        state["current_stage"] = "writing_methodology"
        state["progress"] = 35.0
        
        try:
            task_spec = state["task_assignments"]["agent_tasks"]["methodology_writer"]
            
            # TODO: Implement actual MethodologyWriter agent call
            # if self.methodology_writer:
            #     methodology_draft = await self.methodology_writer.generate_section(
            #         cells=task_spec["input_cells"],
            #         requirements=task_spec["requirements"]
            #     )
            # else:
            
            # Placeholder implementation
            methodology_cells = task_spec["input_cells"]
            methodology_draft = self._generate_methodology_placeholder(methodology_cells)
            
            state["methodology_draft"] = methodology_draft
            
            state["messages"].append({
                "agent": "methodology_writer",
                "stage": "methodology",
                "message": f"📝 Methodology section generated from {len(methodology_cells)} code cells"
            })
            
        except Exception as e:
            error_msg = f"Methodology writing failed: {str(e)}"
            state["errors"].append(error_msg)
            state["messages"].append({
                "agent": "methodology_writer",
                "stage": "methodology", 
                "error": error_msg
            })
            
        return state
    
    def _generate_methodology_placeholder(self, cells: List[Dict]) -> str:
        """Generate placeholder methodology section"""
        methodology = "## Methodology\n\n"
        
        if cells:
            methodology += "We employed the following computational approach:\n\n"
            for i, cell in enumerate(cells):
                content = cell.get("content", "") + cell.get("source", "")
                if content:
                    methodology += f"**Step {i+1}:** Based on the implementation:\n```python\n{content[:200]}...\n```\n\n"
        else:
            methodology += "*Methodology will be generated from code cells by MethodologyWriter agent.*\n"
            
        return methodology
    
    def _coordinate_results(self, state: PapergenState) -> PapergenState:
        """
        Coordinate with ResultsWriter agent.
        
        TASK: @ResultsWriter, use cells [c4, c5, c6] to write the [RESULTS] section.
        """
        state["current_stage"] = "writing_results"
        state["progress"] = 50.0
        
        try:
            task_spec = state["task_assignments"]["agent_tasks"]["results_writer"]
            
            # TODO: Implement actual ResultsWriter agent call
            # if self.results_writer:
            #     results_draft = await self.results_writer.generate_section(
            #         cells=task_spec["input_cells"],
            #         requirements=task_spec["requirements"]
            #     )
            # else:
            
            # Placeholder implementation
            results_cells = task_spec["input_cells"]
            results_draft = self._generate_results_placeholder(results_cells)
            
            state["results_draft"] = results_draft
            
            state["messages"].append({
                "agent": "results_writer",
                "stage": "results",
                "message": f"📈 Results section generated from {len(results_cells)} output cells"
            })
            
        except Exception as e:
            error_msg = f"Results writing failed: {str(e)}"
            state["errors"].append(error_msg)
            state["messages"].append({
                "agent": "results_writer",
                "stage": "results",
                "error": error_msg
            })
            
        return state
    
    def _generate_results_placeholder(self, cells: List[Dict]) -> str:
        """Generate placeholder results section"""
        results = "## Results\n\n"
        
        plot_count = sum(1 for cell in cells if cell.get("type") == "output_plot")
        text_count = sum(1 for cell in cells if cell.get("type") == "output_text")
        
        if plot_count > 0:
            results += f"The analysis generated {plot_count} visualization(s) showing the model performance.\n\n"
            
        if text_count > 0:
            results += f"Numerical results from {text_count} output(s) demonstrate:\n\n"
            for cell in cells:
                if cell.get("type") == "output_text":
                    content = str(cell.get("content", ""))[:100]
                    results += f"- {content}...\n"
                    
        if not cells:
            results += "*Results will be generated from output cells by ResultsWriter agent.*\n"
            
        return results
    
    def _coordinate_literary(self, state: PapergenState) -> PapergenState:
        """
        Coordinate with LiteraryAgent for high-level narrative.
        
        TASK: @LiteraryAgent, use cell [c1] and wait for other drafts to write [ABSTRACT] and [INTRODUCTION].
        """
        state["current_stage"] = "writing_literary"
        state["progress"] = 65.0
        
        try:
            # Implement actual LiteraryAgent call
            if self.literary_agent:
                # Prepare synthesis data for LiteraryAgent
                synthesis_data = {
                    "method_draft": state["methodology_draft"],
                    "results_draft": state["results_draft"]
                }
                
                # Generate all literary sections using the actual agent
                literary_sections = self.literary_agent.synthesize_all_sections(synthesis_data)
                
                # Parse the sections to extract clean content
                parsed_sections = {}
                for section_key, section_content in literary_sections.items():
                    # Remove the **Section: X** header and get clean content
                    lines = section_content.split('\n')
                    clean_content = []
                    skip_header = True
                    
                    for line in lines:
                        if skip_header and line.strip().startswith('**Section:'):
                            skip_header = False
                            continue
                        if not skip_header:
                            clean_content.append(line)
                    
                    parsed_sections[section_key] = '\n'.join(clean_content).strip()
                
                state["literary_sections"] = parsed_sections
                
                state["messages"].append({
                    "agent": "literary_agent", 
                    "stage": "literary",
                    "message": f"✍️ Generated {len(parsed_sections)} literary sections using LiteraryAgent"
                })
                
            else:
                # Fallback to placeholder if agent not available
                user_objectives = state["task_assignments"]["user_objectives"]
                literary_sections = self._generate_literary_placeholder(user_objectives)
                state["literary_sections"] = literary_sections
                
                state["messages"].append({
                    "agent": "literary_agent", 
                    "stage": "literary",
                    "message": f"⚠️ Used placeholder - LiteraryAgent not initialized"
                })
            
        except Exception as e:
            error_msg = f"Literary writing failed: {str(e)}"
            state["errors"].append(error_msg)
            state["messages"].append({
                "agent": "literary_agent",
                "stage": "literary",
                "error": error_msg
            })
            
            # Fallback to placeholder on error
            try:
                user_objectives = state["task_assignments"]["user_objectives"]
                literary_sections = self._generate_literary_placeholder(user_objectives)
                state["literary_sections"] = literary_sections
            except:
                state["literary_sections"] = {
                    "abstract": "Error: Could not generate abstract",
                    "introduction": "Error: Could not generate introduction", 
                    "conclusion": "Error: Could not generate conclusion"
                }
            
        return state
    
    def _generate_literary_placeholder(self, objectives: Dict) -> Dict[str, str]:
        """Generate placeholder literary sections"""
        title = objectives.get("title", "Research Analysis")
        description = objectives.get("description", "")
        
        return {
            "abstract": f"## Abstract\n\n{title}\n\nThis paper presents an analysis of... *[Generated by LiteraryAgent]*\n",
            "introduction": f"## Introduction\n\n{description}\n\n*[Generated by LiteraryAgent based on user objectives]*\n",
            "conclusion": "## Conclusion\n\nIn conclusion, our analysis demonstrates... *[Generated by LiteraryAgent]*\n"
        }
    
    def _coordinate_illustrations(self, state: PapergenState) -> PapergenState:
        """
        Coordinate with IllustrationCritic agent.
        
        TASK: @IllustrationCritic, review draft and identify missing visuals.
        """
        state["current_stage"] = "reviewing_illustrations"
        state["progress"] = 80.0
        
        try:
            # TODO: Implement actual IllustrationCritic call
            # if self.illustration_critic:
            #     illustration_plan = await self.illustration_critic.review_draft(
            #         draft_content=combined_draft,
            #         existing_plots=existing_plots
            #     )
            # else:
            
            # Count existing plots from notebook
            existing_plots = [cell for cell in state["notebook_cells"] 
                            if cell.get("type") == "output_plot"]
            
            illustrations = []
            for plot in existing_plots:
                illustrations.append({
                    "type": "existing",
                    "source": "notebook", 
                    "path": plot.get("content", {}).get("image_path", ""),
                    "description": "Plot from notebook analysis"
                })
            
            # Placeholder for generated illustrations
            if state["methodology_draft"]:
                illustrations.append({
                    "type": "generated", 
                    "description": "Architecture diagram for methodology",
                    "path": "architecture_diagram.png",
                    "prompt": "Generate diagram showing model architecture"
                })
                
            state["illustrations"] = illustrations
            
            state["messages"].append({
                "agent": "illustration_critic",
                "stage": "illustrations", 
                "message": f"🎨 Reviewed illustrations: {len(existing_plots)} existing, {len(illustrations)-len(existing_plots)} recommended"
            })
            
        except Exception as e:
            error_msg = f"Illustration review failed: {str(e)}"
            state["errors"].append(error_msg)
            state["messages"].append({
                "agent": "illustration_critic",
                "stage": "illustrations",
                "error": error_msg
            })
            
        return state
    
    def _coordinate_assembly(self, state: PapergenState) -> PapergenState:
        """
        Coordinate with Formatter agent for final assembly.
        
        TASK: @Formatter, assemble final paper from all sections and illustrations.
        """
        state["current_stage"] = "assembling_paper"
        state["progress"] = 90.0
        
        try:
            # TODO: Implement actual Formatter call
            # if self.formatter_agent:
            #     assembled_paper = await self.formatter_agent.assemble_paper(
            #         template=state["template_content"],
            #         sections=all_sections,
            #         illustrations=state["illustrations"]
            #     )
            # else:
            
            # Placeholder assembly
            assembled_content = self._assemble_paper_placeholder(state)
            state["assembled_paper"] = assembled_content
            
            state["messages"].append({
                "agent": "formatter",
                "stage": "assembly",
                "message": "📄 Paper assembly complete - all sections integrated"
            })
            
        except Exception as e:
            error_msg = f"Paper assembly failed: {str(e)}"
            state["errors"].append(error_msg)
            state["messages"].append({
                "agent": "formatter",
                "stage": "assembly", 
                "error": error_msg
            })
            
        return state
    
    def _assemble_paper_placeholder(self, state: PapergenState) -> str:
        """Assemble paper from all sections"""
        paper_parts = []
        
        # Add literary sections
        if "abstract" in state["literary_sections"]:
            paper_parts.append(state["literary_sections"]["abstract"])
        if "introduction" in state["literary_sections"]:
            paper_parts.append(state["literary_sections"]["introduction"])
            
        # Add methodology
        if state["methodology_draft"]:
            paper_parts.append(state["methodology_draft"])
            
        # Add results
        if state["results_draft"]:
            paper_parts.append(state["results_draft"])
            
        # Add conclusion
        if "conclusion" in state["literary_sections"]:
            paper_parts.append(state["literary_sections"]["conclusion"])
            
        # Add illustrations section
        if state["illustrations"]:
            illustrations_section = "\n## Figures\n\n"
            for i, ill in enumerate(state["illustrations"]):
                illustrations_section += f"**Figure {i+1}:** {ill.get('description', 'Illustration')}\n"
                if ill.get('path'):
                    illustrations_section += f"![Figure {i+1}]({ill['path']})\n\n"
            paper_parts.append(illustrations_section)
        
        return "\n\n".join(paper_parts)
    
    def _quality_check(self, state: PapergenState) -> PapergenState:
        """
        Perform final quality check on generated paper.
        Validates completeness and consistency.
        """
        state["current_stage"] = "quality_check"
        state["progress"] = 100.0
        
        try:
            quality_issues = []
            
            # Check if all sections are present
            if not state["assembled_paper"]:
                quality_issues.append("No assembled paper content")
            if not state["methodology_draft"]:
                quality_issues.append("Missing methodology section")
            if not state["results_draft"]:
                quality_issues.append("Missing results section")
            if not state["literary_sections"]:
                quality_issues.append("Missing literary sections")
                
            # Check for minimum content length
            if len(state["assembled_paper"]) < 500:
                quality_issues.append("Paper content too short")
                
            # Store quality issues as warnings
            state["warnings"].extend(quality_issues)
            
            # Determine if generation is complete
            has_critical_errors = len([e for e in state["errors"] if "failed" in e]) > 0
            
            if not has_critical_errors and state["assembled_paper"]:
                state["generation_complete"] = True
                state["messages"].append({
                    "agent": "orchestrator",
                    "stage": "quality_check",
                    "message": f"✅ Quality check completed - Paper generation successful! ({len(quality_issues)} warnings)"
                })
            else:
                state["messages"].append({
                    "agent": "orchestrator", 
                    "stage": "quality_check",
                    "message": f"⚠️ Quality check found {len(quality_issues)} issues"
                })
                
        except Exception as e:
            error_msg = f"Quality check failed: {str(e)}"
            state["errors"].append(error_msg)
            state["messages"].append({
                "agent": "orchestrator",
                "stage": "quality_check", 
                "error": error_msg
            })
            
        return state
    
    async def generate_paper(self, notebook_cells: List[Dict], template: str, user_requirements: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Main entry point for paper generation workflow.
        
        Args:
            notebook_cells: Parsed notebook cells from NotebookParser agent
            template: Research paper template content with placeholders
            user_requirements: Optional user specifications and preferences
            
        Returns:
            Dict containing generated paper content, metadata, and status
        """
        
        # Initialize state
        initial_state = PapergenState(
            notebook_cells=notebook_cells,
            template_content=template,
            user_requirements=user_requirements or {},
            current_stage="initializing",
            progress=0.0,
            task_assignments={},
            methodology_draft="",
            results_draft="",
            literary_sections={},
            illustrations=[],
            assembled_paper="",
            generation_complete=False,
            errors=[],
            warnings=[],
            messages=[]
        )
        
        print("🚀 Starting papergen orchestrator workflow...")
        
        # Run the workflow
        try:
            final_state = self.workflow.invoke(initial_state)
            
            return {
                "success": final_state["generation_complete"],
                "paper_content": final_state["assembled_paper"],
                "sections": {
                    "abstract": final_state["literary_sections"].get("abstract", ""),
                    "introduction": final_state["literary_sections"].get("introduction", ""),
                    "methodology": final_state["methodology_draft"],
                    "results": final_state["results_draft"],
                    "conclusion": final_state["literary_sections"].get("conclusion", "")
                },
                "illustrations": final_state["illustrations"],
                "errors": final_state["errors"],
                "warnings": final_state["warnings"],
                "messages": final_state["messages"],
                "metadata": {
                    "stages_completed": final_state["current_stage"],
                    "progress": final_state["progress"],
                    "total_notebook_cells": len(notebook_cells),
                    "task_assignments": final_state["task_assignments"]
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Orchestrator workflow execution failed: {str(e)}",
                "paper_content": "",
                "sections": {},
                "illustrations": [],
                "errors": [str(e)],
                "warnings": [],
                "messages": [{"agent": "orchestrator", "error": str(e)}],
                "metadata": {"workflow_failed": True}
            }

# Factory function for easy instantiation
def create_papergen_orchestrator() -> PapergenOrchestrator:
    """Create and return a configured PapergenOrchestrator instance"""
    return PapergenOrchestrator()
