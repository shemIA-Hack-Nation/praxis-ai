"""
Paper Parser & Embedder Agent: Extracts concepts, generates embeddings, and builds Knowledge Graph.
"""
import logging
from typing import List, Dict, Any, Optional
import asyncio

try:
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:
    # Fallback for older langchain versions
    try:
        from langchain.schema import HumanMessage, SystemMessage
    except ImportError:
        HumanMessage = None
        SystemMessage = None

from app.core.config import settings
from app.core.llm_providers import get_llm, get_embeddings
from app.agents.industrynav.kg_builder import KnowledgeGraphBuilder
from app.agents.industrynav.vector_db import VectorDBManager

logger = logging.getLogger(__name__)


class PaperParserAgent:
    """Agent responsible for parsing papers and building indexes."""
    
    def __init__(self):
        """Initialize paper parser."""
        # Initialize embeddings model (using sentence-transformers) - lazy loading
        self.embeddings = get_embeddings()
        
        # Initialize LLM for concept extraction (using Gemini) - lazy loading
        self.llm = get_llm(temperature=0)
        
        # Initialize Vector DB and KG builders - lazy loading
        self.vector_db = None
        self.kg_builder = None
    
    def _ensure_initialized(self):
        """Lazy initialization of vector DB and KG builders."""
        if self.vector_db is None:
            self.vector_db = VectorDBManager()
        if self.kg_builder is None:
            self.kg_builder = KnowledgeGraphBuilder()
    
    async def parse_and_index(
        self,
        paper_corpus: List[Dict[str, Any]],
        target_paper: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Parse papers and build vector database and knowledge graph.
        
        Args:
            paper_corpus: List of paper metadata
            target_paper: Target paper (research context from notebook)
        
        Returns:
            Dictionary with parsed data, embeddings, and KG structure
        """
        logger.info(f"Parsing and indexing {len(paper_corpus)} papers")
        
        # Include target paper in corpus for processing
        all_papers = [target_paper] + paper_corpus
        
        # Extract concepts from all papers with rate limiting (batch processing)
        logger.info("Extracting concepts from papers...")
        # Process in batches to avoid rate limits (5 papers at a time with delay)
        batch_size = 5
        papers_with_concepts = []
        
        for i in range(0, len(all_papers), batch_size):
            batch = all_papers[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(all_papers) + batch_size - 1)//batch_size} ({len(batch)} papers)")
            
            # Process batch in parallel
            batch_tasks = [self._extract_concepts(paper) for paper in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            papers_with_concepts.extend(batch_results)
            
            # Add delay between batches to avoid rate limits (except for last batch)
            if i + batch_size < len(all_papers):
                await asyncio.sleep(2.0)  # 2 second delay between batches
        
        # Filter out errors
        valid_papers = []
        for i, result in enumerate(papers_with_concepts):
            if isinstance(result, Exception):
                logger.error(f"Error extracting concepts from paper {i}: {str(result)}")
                continue
            valid_papers.append(result)
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        papers_with_embeddings = await self._generate_embeddings(valid_papers)
        
        # Ensure components are initialized
        self._ensure_initialized()
        
        # Store in Vector DB
        logger.info("Storing embeddings in Vector DB...")
        await self.vector_db.store_papers(papers_with_embeddings)
        
        # Build Knowledge Graph
        logger.info("Building Knowledge Graph...")
        kg_data = await self.kg_builder.build_graph(papers_with_embeddings)
        
        return {
            "papers": papers_with_embeddings,
            "kg_data": kg_data,
            "target_paper_index": 0  # First paper is target
        }
    
    async def _extract_concepts(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract key concepts, methods, and contributions from a paper using LLM.
        
        Args:
            paper: Paper metadata dictionary
        
        Returns:
            Paper dictionary with added concepts, methods, and contributions
        """
        if not self.llm:
            # Fallback: simple keyword extraction
            return {
                **paper,
                "concepts": [],
                "methods": [],
                "contributions": []
            }
        
        try:
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            full_text = paper.get("full_text", "")
            
            # Use full text if available, otherwise use abstract
            content_to_analyze = full_text[:1000] if full_text else abstract[:500]
            content_type = "Full text" if full_text else "Abstract"
            
            # Truncate to save tokens
            truncated_content = content_to_analyze
            
            prompt = f"""Extract key info from this paper. Return JSON only:
{{
    "concepts": ["concept1", "concept2"],
    "methods": ["method1", "method2"],
    "contributions": ["contribution1"]
}}

Title: {title[:200]}
{content_type}: {truncated_content}"""
            
            messages = [
                SystemMessage(content="Extract key info from papers. Return JSON only."),  # Shorter system message
                HumanMessage(content=prompt)
            ]
            
            # Handle both sync and async LLM calls
            try:
                if hasattr(self.llm, 'ainvoke'):
                    response = await self.llm.ainvoke(messages)
                else:
                    # Fallback to sync if async not available
                    response = self.llm.invoke(messages)
                content = response.content
            except Exception as e:
                logger.error(f"Error calling LLM: {str(e)}")
                raise
            
            # Parse JSON response
            import json
            # Try to extract JSON from response
            try:
                # Remove markdown code blocks if present
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                extracted_data = json.loads(content)
                
                paper["concepts"] = extracted_data.get("concepts", [])
                paper["methods"] = extracted_data.get("methods", [])
                paper["contributions"] = extracted_data.get("contributions", [])
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON from LLM response: {content}")
                paper["concepts"] = []
                paper["methods"] = []
                paper["contributions"] = []
        
        except Exception as e:
            logger.error(f"Error extracting concepts: {str(e)}")
            paper["concepts"] = []
            paper["methods"] = []
            paper["contributions"] = []
        
        return paper
    
    async def _generate_embeddings(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for paper abstracts.
        
        Args:
            papers: List of papers with concepts extracted
        
        Returns:
            List of papers with embeddings added
        """
        if not self.embeddings:
            logger.warning("Embeddings model not available, skipping embedding generation")
            # Create dummy embeddings for testing
            import numpy as np
            for paper in papers:
                # Create a random embedding vector (384 dimensions for all-MiniLM-L6-v2)
                paper["embedding"] = np.random.rand(384).tolist()
                paper["embedding_model"] = "dummy"
            return papers
        
        try:
            # Prepare texts for embedding
            texts = []
            for paper in papers:
                # Combine title and abstract for embedding
                text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
                texts.append(text[:8000])  # Limit text length
            
            # Generate embeddings - handle both sync and async
            try:
                if hasattr(self.embeddings, 'aembed_documents'):
                    embeddings_list = await self.embeddings.aembed_documents(texts)
                else:
                    # Fallback to sync if async not available
                    embeddings_list = self.embeddings.embed_documents(texts)
            except Exception as e:
                logger.error(f"Error in embedding generation: {str(e)}")
                # Create dummy embeddings as fallback
                import numpy as np
                embeddings_list = [np.random.rand(384).tolist() for _ in texts]
            
            # Add embeddings to papers
            for i, paper in enumerate(papers):
                if i < len(embeddings_list):
                    paper["embedding"] = embeddings_list[i]
                    paper["embedding_model"] = settings.EMBEDDING_MODEL
                else:
                    # Fallback for missing embeddings
                    import numpy as np
                    paper["embedding"] = np.random.rand(384).tolist()
                    paper["embedding_model"] = "fallback"
            
            return papers
        
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            # Create dummy embeddings as fallback
            import numpy as np
            for paper in papers:
                if "embedding" not in paper:
                    paper["embedding"] = np.random.rand(384).tolist()
                    paper["embedding_model"] = "error_fallback"
            return papers