"""
Pipeline manager that orchestrates the multi-agent research analysis pipeline.
"""
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List

from app.core.config import settings
from app.agents.industrynav.agent_research_expander import ResearchExpanderAgent
from app.agents.industrynav.agent_paper_parser import PaperParserAgent
from app.agents.industrynav.agent_novelty_assessor import NoveltyAssessorAgent
from app.agents.industrynav.agent_report_generator import ReportGeneratorAgent
from app.agents.industrynav.agent_pdf_parser import PDFParser
from app.agents.industrynav.agent_augmenter import AugmentationAgent

logger = logging.getLogger(__name__)


class PipelineManager:
    """Manages the research analysis pipeline."""
    
    def __init__(self):
        """Initialize pipeline manager with all agents (lazy initialization)."""
        # Don't initialize agents here to avoid slow startup
        # They will be initialized on first use
        self.pdf_parser = None
        self.research_expander = None
        self.paper_parser = None
        self.novelty_assessor = None
        self.report_generator = None
        self.augmentation_agent = None
        self.progress_callback = None
    
    def set_progress_callback(self, callback):
        """Set progress callback for real-time updates."""
        self.progress_callback = callback
        # Also set callback on research expander when initialized
        if self.research_expander:
            self.research_expander.set_progress_callback(callback)
    
    def _ensure_agents_initialized(self):
        """Lazy initialization of agents."""
        if self.pdf_parser is None:
            self.pdf_parser = PDFParser()
        if self.research_expander is None:
            self.research_expander = ResearchExpanderAgent()
            # Set progress callback if available
            if self.progress_callback:
                self.research_expander.set_progress_callback(self.progress_callback)
        if self.paper_parser is None:
            self.paper_parser = PaperParserAgent()
        if self.novelty_assessor is None:
            self.novelty_assessor = NoveltyAssessorAgent()
        if self.report_generator is None:
            self.report_generator = ReportGeneratorAgent()
        if self.augmentation_agent is None:
            self.augmentation_agent = AugmentationAgent()
    
    async def process_paper(self, paper_path: str) -> Dict[str, str]:
        """
        Process a research paper through the entire pipeline.
        
        Args:
            paper_path: Path to the research paper PDF file
        
        Returns:
            Dictionary with 'paper' and 'analysis' keys containing markdown content
        """
        try:
            # Initialize agents on first use
            self._ensure_agents_initialized()
            
            logger.info(f"Starting pipeline processing for paper: {paper_path}")
            
            # Helper function for progress notifications
            async def notify_progress(event: str, data: Dict[str, Any]):
                """Notify progress if callback is set."""
                if self.progress_callback:
                    try:
                        if asyncio.iscoroutinefunction(self.progress_callback):
                            await self.progress_callback(event, data)
                        else:
                            self.progress_callback(event, data)
                    except Exception as e:
                        logger.warning(f"Error in progress callback: {str(e)}")
            
            # Step 1: Parse uploaded paper and extract content
            logger.info("Step 1: Parsing research paper and extracting content...")
            await notify_progress("extraction_started", {"message": "Step 1: Extracting content from PDF..."})
            
            paper_data = await self.pdf_parser.parse_paper(paper_path)
            await notify_progress("pdf_parsed", {
                "message": "PDF parsed successfully",
                "text_length": paper_data.get("text_length", 0)
            })
            
            research_context = await self.pdf_parser.extract_research_context(paper_data)
            await notify_progress("content_extracted", {
                "message": "Content extracted from paper",
                "title": research_context.get("title", "")[:50],
                "has_abstract": bool(research_context.get("abstract")),
                "publication_date": research_context.get("publication_date"),
                "authors_count": len(research_context.get("authors", [])),
                "keywords_count": len(research_context.get("keywords", []))
            })
            
            logger.info(f"Extracted content - Title: {research_context.get('title', 'Unknown')[:50]}")
            logger.info(f"Publication date: {research_context.get('publication_date', 'Unknown')}")
            logger.info(f"Authors: {len(research_context.get('authors', []))} authors found")
            
            # Step 2: Expand research corpus (only after content extraction)
            logger.info("Step 2: Expanding research corpus based on extracted content...")
            paper_corpus = await self.research_expander.expand_research(
                research_context,
                max_papers=settings.MAX_PAPERS_IN_CORPUS,
                min_papers=settings.MIN_PAPERS_IN_CORPUS,
                enable_full_text=settings.ENABLE_FULL_TEXT_SCRAPING
            )
            
            # Step 3: Parse papers and build indexes
            logger.info("Step 3: Parsing papers and building indexes...")
            await notify_progress("indexing_started", {
                "message": "Step 3: Generating embeddings and building indexes...",
                "papers_count": len(paper_corpus)
            })
            
            parsed_data = await self.paper_parser.parse_and_index(
                paper_corpus,
                target_paper=research_context
            )
            
            await notify_progress("indexing_complete", {
                "message": "Indexes built successfully",
                "papers_indexed": len(parsed_data.get("papers", []))
            })
            
            # Step 4: Assess novelty
            logger.info("Step 4: Assessing novelty...")
            await notify_progress("novelty_assessment_started", {
                "message": "Step 4: Assessing novelty based on similarity and date context..."
            })
            
            novelty_result = await self.novelty_assessor.assess_novelty(
                target_paper=research_context,
                parsed_data=parsed_data
            )
            
            await notify_progress("novelty_assessment_complete", {
                "message": "Novelty assessment complete",
                "novelty_score": novelty_result.get("score", 0)
            })
            
            # Step 5: Generate reports
            logger.info("Step 5: Generating reports...")
            await notify_progress("report_generation_started", {
                "message": "Step 5: Generating final reports..."
            })
            
            reports = await self.report_generator.generate_reports(
                paper_data=paper_data,
                research_context=research_context,
                paper_corpus=paper_corpus,
                parsed_data=parsed_data,
                novelty_result=novelty_result
            )
            
            await notify_progress("report_generation_complete", {
                "message": "Initial analysis complete - Knowledge graph ready",
                "nodes_count": reports.get("knowledge_graph", {}).get("stats", {}).get("total_nodes", 0),
                "links_count": reports.get("knowledge_graph", {}).get("stats", {}).get("total_links", 0),
                "initial_novelty_score": novelty_result.get("score", 0)
            })
            
            # Mark initial results as ready (this event is sent before returning)
            # The WebSocket will receive this to initialize the augmentation panel
            await notify_progress("initial_results_ready", {
                "message": "Initial knowledge graph is ready",
                "papers_count": len(paper_corpus),
                "novelty_score": novelty_result.get("score", 0),
                "knowledge_graph": reports.get("knowledge_graph")
            })
            
            # Store initial results for augmentation
            initial_reports = reports.copy()
            
            # Step 6: Augmentation (runs in background, updates graph in real-time)
            if settings.ENABLE_AUGMENTATION:
                # Start augmentation as background task
                asyncio.create_task(self._augment_and_update(
                    research_context=research_context,
                    paper_data=paper_data,
                    initial_paper_corpus=paper_corpus,
                    initial_parsed_data=parsed_data,
                    initial_novelty_result=novelty_result,
                    initial_reports=initial_reports,
                    notify_progress=notify_progress
                ))
            else:
                # No augmentation, send final completion
                await notify_progress("processing_complete", {
                    "message": "All processing complete",
                    "total_papers": len(paper_corpus),
                    "final_novelty_score": novelty_result.get("score", 0)
                })
            
            logger.info("Pipeline processing completed successfully - returning initial results")
            # Return initial results immediately
            return initial_reports
        
        except Exception as e:
            logger.error(f"Error in pipeline processing: {str(e)}", exc_info=True)
            raise
    
    async def _augment_and_update(
        self,
        research_context: Dict[str, Any],
        paper_data: Dict[str, Any],
        initial_paper_corpus: List[Dict[str, Any]],
        initial_parsed_data: Dict[str, Any],
        initial_novelty_result: Dict[str, Any],
        initial_reports: Dict[str, Any],
        notify_progress: callable
    ):
        """
        Augment analysis in background and send real-time updates.
        
        This runs as a background task after initial results are returned.
        """
        try:
            self._ensure_agents_initialized()  # Ensure augmentation agent is initialized
            logger.info("Starting background augmentation process...")
            
            await notify_progress("augmentation_started", {
                "message": "Augmentation: Identifying gaps and interesting areas...",
                "initial_papers": len(initial_paper_corpus)
            })
            
            # Run augmentation
            augmentation_result = await self.augmentation_agent.augment_analysis(
                research_context=research_context,
                initial_paper_corpus=initial_paper_corpus,
                initial_parsed_data=initial_parsed_data,
                initial_novelty_result=initial_novelty_result,
                progress_callback=notify_progress
            )
            
            # Use augmented results if we got additional papers
            if augmentation_result.get("additional_papers_count", 0) > 0:
                logger.info(f"Augmentation found {augmentation_result['additional_papers_count']} additional papers")
                
                # Send update with new papers being added
                await notify_progress("augmentation_papers_found", {
                    "message": f"Found {augmentation_result['additional_papers_count']} additional papers",
                    "additional_papers": augmentation_result["additional_papers_count"],
                    "new_papers": augmentation_result["augmented_corpus"][len(initial_paper_corpus):]
                })
                
                # Send intermediate novelty score update during parsing
                await notify_progress("augmentation_novelty", {
                    "message": "Re-assessing novelty with augmented corpus...",
                    "papers_count": len(augmentation_result["augmented_corpus"]),
                    "novelty_score": augmentation_result["augmented_novelty_result"].get("score", 0)
                })
                
                # Re-generate reports with augmented data
                await notify_progress("augmentation_report_generation", {
                    "message": "Updating knowledge graph with augmented data..."
                })
                
                final_reports = await self.report_generator.generate_reports(
                    paper_data=paper_data,
                    research_context=research_context,
                    paper_corpus=augmentation_result["augmented_corpus"],
                    parsed_data=augmentation_result["augmented_parsed_data"],
                    novelty_result=augmentation_result["augmented_novelty_result"]
                )
                
                # Send updated knowledge graph
                await notify_progress("knowledge_graph_updated", {
                    "message": "Knowledge graph updated with augmented data",
                    "knowledge_graph": final_reports["knowledge_graph"],
                    "nodes_count": final_reports.get("knowledge_graph", {}).get("stats", {}).get("total_nodes", 0),
                    "links_count": final_reports.get("knowledge_graph", {}).get("stats", {}).get("total_links", 0),
                    "total_papers": len(augmentation_result["augmented_corpus"])
                })
                
                # Send final novelty assessment
                final_novelty_score = augmentation_result["augmented_novelty_result"].get("score", 0)
                augmented_analysis = await self.report_generator._generate_analysis(
                    research_context=research_context,
                    paper_corpus=augmentation_result["augmented_corpus"],
                    parsed_data=augmentation_result["augmented_parsed_data"],
                    novelty_result=augmentation_result["augmented_novelty_result"]
                )
                await notify_progress("final_novelty_assessment", {
                    "message": "Final novelty assessment complete",
                    "novelty_score": final_novelty_score,
                    "novelty_result": augmentation_result["augmented_novelty_result"],
                    "augmented_novelty_analysis": augmented_analysis
                })
                
                await notify_progress("augmentation_complete", {
                    "message": f"Augmentation complete: {augmentation_result['additional_papers_count']} additional papers processed",
                    "total_papers": len(augmentation_result["augmented_corpus"]),
                    "final_novelty_score": final_novelty_score,
                    "initial_novelty_score": initial_novelty_result.get("score", 0)
                })
            else:
                logger.info("No additional papers found during augmentation, but novelty was optimized")
                # Even without new papers, we may have optimized the novelty score
                optimized_score = augmentation_result.get("augmented_novelty_result", {}).get("score", initial_novelty_result.get("score", 0))
                
                # Update reports with optimized novelty if score changed
                if optimized_score != initial_novelty_result.get("score", 0):
                    # Re-generate reports with optimized novelty
                    await notify_progress("augmentation_report_generation", {
                        "message": "Updating knowledge graph with optimized novelty score..."
                    })
                    
                    final_reports = await self.report_generator.generate_reports(
                        paper_data=paper_data,
                        research_context=research_context,
                        paper_corpus=initial_paper_corpus,
                        parsed_data=initial_parsed_data,
                        novelty_result=augmentation_result.get("augmented_novelty_result", initial_novelty_result)
                    )
                    
                    # Send updated knowledge graph
                    await notify_progress("knowledge_graph_updated", {
                        "message": "Knowledge graph updated with optimized novelty assessment",
                        "knowledge_graph": final_reports["knowledge_graph"],
                        "nodes_count": final_reports.get("knowledge_graph", {}).get("stats", {}).get("total_nodes", 0),
                        "links_count": final_reports.get("knowledge_graph", {}).get("stats", {}).get("total_links", 0),
                        "total_papers": len(initial_paper_corpus)
                    })
                    
                    # Send final novelty assessment
                    augmented_analysis = await self.report_generator._generate_analysis(
                        research_context=research_context,
                        paper_corpus=initial_paper_corpus,
                        parsed_data=initial_parsed_data,
                        novelty_result=augmentation_result.get("augmented_novelty_result", initial_novelty_result)
                    )
                    await notify_progress("final_novelty_assessment", {
                        "message": "Final novelty assessment complete",
                        "novelty_score": optimized_score,
                        "novelty_result": augmentation_result.get("augmented_novelty_result", initial_novelty_result),
                        "augmented_novelty_analysis": augmented_analysis
                    })
                
                await notify_progress("augmentation_complete", {
                    "message": "Novelty optimization complete",
                    "total_papers": len(initial_paper_corpus),
                    "final_novelty_score": optimized_score,
                    "initial_novelty_score": initial_novelty_result.get("score", 0),
                    "additional_papers": 0
                })
            
            # Send final completion
            await notify_progress("processing_complete", {
                "message": "All processing complete",
                "total_papers": len(augmentation_result.get("augmented_corpus", initial_paper_corpus)),
                "final_novelty_score": augmentation_result.get("augmented_novelty_result", initial_novelty_result).get("score", 0)
            })
            
        except Exception as e:
            logger.error(f"Error in augmentation process: {str(e)}", exc_info=True)
            try:
                await notify_progress("augmentation_error", {
                    "error": str(e),
                    "message": "Augmentation encountered an error, but continuing with initial results"
                })
                # Even on error, try to optimize novelty if possible
                try:
                    optimized_novelty_result = await self.novelty_assessor.assess_novelty(
                        target_paper=research_context,
                        parsed_data=initial_parsed_data
                    )
                    if optimized_novelty_result.get("score", 0) != initial_novelty_result.get("score", 0):
                        await notify_progress("final_novelty_assessment", {
                            "message": "Novelty score optimized despite augmentation error",
                            "novelty_score": optimized_novelty_result.get("score", 0),
                            "novelty_result": optimized_novelty_result
                        })
                except Exception as novelty_error:
                    logger.warning(f"Error optimizing novelty after augmentation error: {str(novelty_error)}")
            except Exception as notify_error:
                logger.error(f"Error sending augmentation error notification: {str(notify_error)}")
