"""
Research Expander Agent: Builds corpus of related papers from arXiv and Semantic Scholar.
Optimized version with LLM-based term extraction and real-time visualization.
"""
import logging
from typing import List, Dict, Any, Optional, Callable
import asyncio

import arxiv
import requests

try:
    from semanticscholar import SemanticScholar
    HAS_SEMANTIC_SCHOLAR = True
except ImportError:
    HAS_SEMANTIC_SCHOLAR = False
    SemanticScholar = None

from app.core.config import settings
from app.services.paper_scraper import PaperScraper
from app.agents.industrynav.search_term_extractor import SearchTermExtractor

logger = logging.getLogger(__name__)


class ResearchExpanderAgent:
    """Agent responsible for expanding research corpus."""
    
    def __init__(self):
        """Initialize research expander."""
        self.semantic_scholar = None
        if HAS_SEMANTIC_SCHOLAR and settings.SEMANTIC_SCHOLAR_API_KEY:
            try:
                self.semantic_scholar = SemanticScholar(api_key=settings.SEMANTIC_SCHOLAR_API_KEY)
            except Exception as e:
                logger.warning(f"Failed to initialize Semantic Scholar: {str(e)}")
                self.semantic_scholar = None
        
        # LLM-based search term extractor
        self.search_extractor = SearchTermExtractor()
        self.progress_callback: Optional[Callable] = None
    
    def set_progress_callback(self, callback: Callable):
        """Set callback for progress updates (for real-time visualization)."""
        self.progress_callback = callback
    
    async def _notify_progress(self, event: str, data: Dict[str, Any]):
        """Notify progress callback if set."""
        if self.progress_callback:
            try:
                if asyncio.iscoroutinefunction(self.progress_callback):
                    await self.progress_callback(event, data)
                else:
                    self.progress_callback(event, data)
            except Exception as e:
                logger.warning(f"Error in progress callback: {str(e)}")
    
    async def expand_research(
        self,
        research_context: Dict[str, Any],
        max_papers: int = 50,
        min_papers: int = 20,
        enable_full_text: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Expand research corpus by finding related papers.
        
        Args:
            research_context: Research context from notebook (title, abstract, methods)
            max_papers: Maximum number of papers to retrieve
            min_papers: Minimum number of papers to retrieve
        
        Returns:
            List of paper metadata dictionaries
        """
        title = research_context.get('title', 'Unknown')
        logger.info(f"Expanding research corpus for: {title}")
        
        # Verify we have content to work with
        if not research_context.get("title") and not research_context.get("abstract"):
            logger.warning("No title or abstract found in research context, using fallback extraction")
        
        await self._notify_progress("search_term_extraction_started", {"message": "Extracting search terms with LLM from paper content..."})
        
        # Step 1: Extract precise search terms using LLM (from already extracted paper content)
        search_terms_dict = await self.search_extractor.extract_search_terms(research_context)
        logger.info(f"LLM-extracted search terms: keywords={search_terms_dict.get('keywords', [])}, methods={search_terms_dict.get('methods', [])}, categories={search_terms_dict.get('categories', [])}")
        
        await self._notify_progress("search_term_extraction_complete", {
            "keywords": search_terms_dict.get("keywords", []),
            "methods": search_terms_dict.get("methods", []),
            "categories": search_terms_dict.get("categories", []),
            "message": f"Extracted {len(search_terms_dict.get('keywords', []))} keywords, {len(search_terms_dict.get('methods', []))} methods"
        })
        
        papers = []
        
        # Use optimized query terms
        query_terms = search_terms_dict.get("query_terms", [])
        keywords = search_terms_dict.get("keywords", [])
        methods = search_terms_dict.get("methods", [])
        
        # If LLM extraction failed, use rule-based fallback
        if not query_terms and not keywords and not methods:
            logger.warning("LLM extraction failed, using rule-based fallback")
            search_terms = self._extract_search_terms(research_context)
            query_terms = search_terms
            keywords = search_terms[:10]
        else:
            # Combine methods and keywords for query
            if not query_terms:
                query_terms = (methods[:3] + keywords[:5])[:8]
        
        # Ensure we have at least some terms
        if not query_terms:
            # Last resort: extract from title and abstract
            title = research_context.get("title", "")
            abstract = research_context.get("abstract", "")
            if title:
                query_terms = [w for w in title.split() if len(w) > 3][:5]
            if not query_terms and abstract:
                query_terms = [w for w in abstract.split() if len(w) > 4][:5]
        
        if not query_terms:
            logger.error("No search terms available, cannot search arXiv")
            await self._notify_progress("search_error", {
                "error": "No search terms extracted from paper",
                "message": "Failed to extract searchable terms from paper content"
            })
            return []
        
        await self._notify_progress("search_started", {
            "message": "Searching arXiv with optimized query...",
            "query_terms": query_terms[:5]
        })
        
        # Search multiple sources in parallel
        tasks = []
        
        publication_date = research_context.get("publication_date")
        categories = search_terms_dict.get("categories", [])
        
        if settings.ARXIV_API_ENABLED:
            tasks.append(self._optimized_search_arxiv(
                search_terms_dict, 
                research_context, 
                max_results=max_papers, 
                target_year=publication_date
            ))
        
        if self.semantic_scholar:
            tasks.append(self._search_semantic_scholar(query_terms, max_results=max_papers, target_year=publication_date))
        
        # Execute searches in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error in search: {str(result)}")
                continue
            if isinstance(result, list):
                papers.extend(result)
        
        # Deduplicate papers by title
        papers = self._deduplicate_papers(papers)
        
        # Filter and rank papers by date relevance and similarity
        publication_date = research_context.get("publication_date")
        papers = self._filter_and_rank_papers(papers, research_context, publication_date)
        
        # Limit to max_papers
        papers = papers[:max_papers]
        
        # Ensure we have at least min_papers
        if len(papers) < min_papers:
            logger.warning(f"Only found {len(papers)} papers, less than minimum {min_papers}")
        
        # Follow citations if enabled (get papers that cite the target paper)
        if settings.ENABLE_CITATION_FOLLOWING and self.semantic_scholar:
            try:
                logger.info("Following citations to find related papers...")
                cited_papers = await self._get_cited_papers(research_context, max_results=10)
                if cited_papers:
                    papers.extend(cited_papers)
                    logger.info(f"Found {len(cited_papers)} papers through citation following")
            except Exception as e:
                logger.warning(f"Error following citations: {str(e)}")
        
        # Deduplicate again after adding cited papers
        papers = self._deduplicate_papers(papers)
        
        # Scrape full text if enabled
        if enable_full_text and papers:
            logger.info("Scraping full text for papers...")
            try:
                async with PaperScraper() as scraper:
                    papers = await scraper.scrape_papers_batch(papers, max_workers=5)
                logger.info(f"Scraped full text for {len([p for p in papers if p.get('full_text')])} papers")
            except Exception as e:
                logger.error(f"Error scraping full text: {str(e)}")
                # Continue without full text if scraping fails
        
        logger.info(f"Found {len(papers)} papers for research corpus")
        return papers
    
    async def _get_cited_papers(self, research_context: Dict[str, Any], max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Get papers that cite the target paper (citation following).
        
        Args:
            research_context: Target paper context
            max_results: Maximum number of cited papers to retrieve
        
        Returns:
            List of papers that cite the target paper
        """
        if not self.semantic_scholar:
            return []
        
        try:
            title = research_context.get("title", "")
            if not title:
                return []
            
            # Run in executor since Semantic Scholar API is sync
            loop = asyncio.get_event_loop()
            
            def get_citations():
                try:
                    # Search for the target paper in Semantic Scholar
                    search_results = self.semantic_scholar.search_paper(title, limit=1)
                    if not search_results:
                        return []
                    
                    target_paper_id = search_results[0].paperId if hasattr(search_results[0], 'paperId') else None
                    if not target_paper_id:
                        return []
                    
                    # Get papers that cite this paper
                    cited_papers = []
                    try:
                        # Get citations (papers that cite the target)
                        citations = self.semantic_scholar.get_paper_citations(target_paper_id, limit=max_results)
                        
                        for citation in citations:
                            if hasattr(citation, 'citingPaper') and citation.citingPaper:
                                paper = citation.citingPaper
                                cited_papers.append({
                                    "title": getattr(paper, 'title', ''),
                                    "abstract": getattr(paper, 'abstract', '') or '',
                                    "year": getattr(paper, 'year', None),
                                    "authors": [author.name for author in getattr(paper, 'authors', [])] if hasattr(paper, 'authors') else [],
                                    "url": getattr(paper, 'url', ''),
                                    "arxiv_id": None,
                                    "citations": getattr(paper, 'citationCount', 0) or 0,
                                    "source": "citation_following"
                                })
                    except Exception as e:
                        logger.warning(f"Error getting citations: {str(e)}")
                    
                    return cited_papers
                except Exception as e:
                    logger.warning(f"Error in citation following: {str(e)}")
                    return []
            
            cited_papers = await loop.run_in_executor(None, get_citations)
            return cited_papers
            
        except Exception as e:
            logger.warning(f"Error in citation following: {str(e)}")
            return []
    
    def _extract_search_terms(self, research_context: Dict[str, Any]) -> List[str]:
        """Extract search terms from research context (improved with keywords)."""
        terms = []
        
        # Use extracted keywords if available (best quality)
        keywords = research_context.get("keywords", [])
        if keywords:
            terms.extend([kw.lower() for kw in keywords if len(kw) > 2])
        
        # Extract from title
        title = research_context.get("title", "")
        if title:
            # Remove common words and extract key terms
            words = title.lower().split()
            stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "using", "based", "novel", "new", "approach"}
            title_terms = [w for w in words if w not in stop_words and len(w) > 3]
            terms.extend(title_terms)
        
        # Extract from abstract (key phrases)
        abstract = research_context.get("abstract", "")
        if abstract:
            # Extract first sentence and key phrases
            sentences = abstract.split('.')
            if sentences:
                first_sentence = sentences[0].lower()
                # Extract meaningful words (length > 4)
                abstract_terms = [w.strip('.,;:()[]') for w in first_sentence.split() if len(w.strip('.,;:()[]')) > 4]
                terms.extend(abstract_terms[:5])
        
        # Extract from methods section (technical terms)
        methods = research_context.get("methods", "")
        if methods and len(terms) < 10:
            # Extract technical terms (capitalized words, acronyms)
            import re
            tech_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', methods[:500])
            terms.extend([t.lower() for t in tech_terms[:3]])
        
        # Remove duplicates and limit
        seen = set()
        unique_terms = []
        for term in terms:
            term_clean = term.strip('.,;:()[]').lower()
            if term_clean and term_clean not in seen and len(term_clean) > 2:
                seen.add(term_clean)
                unique_terms.append(term_clean)
        
        return unique_terms[:15]  # Return top 15 terms
    
    async def _optimized_search_arxiv(
        self, 
        search_terms_dict: Dict[str, Any],
        research_context: Dict[str, Any],
        max_results: int = 50,
        target_year: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Optimized arXiv search with categories, date ranges, and focused queries."""
        try:
            from datetime import datetime, timedelta
            
            # Build optimized query
            methods = search_terms_dict.get("methods", [])[:2]
            keywords = search_terms_dict.get("keywords", [])[:5]
            query_terms = search_terms_dict.get("query_terms", keywords)
            
            # Construct query: (method1 AND method2) OR (keyword1 OR keyword2 OR ...)
            query_parts = []
            if methods:
                methods_query = " AND ".join([f'"{m}"' for m in methods])
                query_parts.append(f"({methods_query})")
            if query_terms:
                keywords_query = " OR ".join([f'"{k}"' for k in query_terms[:5]])
                query_parts.append(f"({keywords_query})")
            
            query = " OR ".join(query_parts) if len(query_parts) > 1 else (query_parts[0] if query_parts else " ".join(query_terms[:3]))
            
            logger.info(f"Optimized arXiv query: {query}")
            
            # Calculate date range
            date_from = None
            date_to = None
            if target_year:
                try:
                    target_year_int = int(target_year)
                    from_year = max(2010, target_year_int - 3)
                    to_year = min(datetime.now().year, target_year_int + 3)
                    date_from = datetime(from_year, 1, 1)
                    date_to = datetime(to_year, 12, 31)
                except:
                    pass
            
            # Run arXiv search in thread pool
            loop = asyncio.get_event_loop()
            
            search_params = {
                "query": query,
                "max_results": min(max_results, 100),  # Hard limit
                "sort_by": arxiv.SortCriterion.Relevance
            }
            
            search = await loop.run_in_executor(
                None,
                lambda: arxiv.Search(**search_params)
            )
            
            # Get results
            results = await loop.run_in_executor(
                None,
                lambda: list(search.results())
            )
            
            papers = []
            for i, paper in enumerate(results):
                # Extract arXiv ID
                arxiv_id = None
                if paper.entry_id:
                    arxiv_id = paper.entry_id.split('/')[-1].replace('.pdf', '')
                
                # Extract categories
                paper_categories = []
                if hasattr(paper, 'primary_category'):
                    paper_categories.append(paper.primary_category)
                if hasattr(paper, 'categories'):
                    paper_categories.extend(paper.categories)
                
                paper_data = {
                    "title": paper.title,
                    "abstract": paper.summary,
                    "url": paper.entry_id,
                    "arxiv_id": arxiv_id,
                    "year": paper.published.year if paper.published else None,
                    "published": paper.published.isoformat() if paper.published else None,
                    "authors": [author.name for author in paper.authors],
                    "categories": paper_categories,
                    "citations": None,
                    "source": "arxiv"
                }
                
                papers.append(paper_data)
                
                # Notify progress for each paper
                await self._notify_progress("paper_found", {
                    "paper": {
                        "title": paper_data["title"][:60],
                        "year": paper_data["year"],
                        "authors": paper_data["authors"][:2],
                        "categories": paper_categories[:2]
                    },
                    "count": len(papers),
                    "total": len(results)
                })
            
            logger.info(f"Found {len(papers)} papers from optimized arXiv search")
            await self._notify_progress("search_complete", {"papers_found": len(papers)})
            return papers
        
        except Exception as e:
            logger.error(f"Error in optimized arXiv search: {str(e)}")
            await self._notify_progress("search_error", {"error": str(e)})
            return []
    
    async def _search_arxiv(self, search_terms: List[str], max_results: int = 25, target_year: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search arXiv for related papers (truly async) - legacy method."""
        # Use optimized search instead
        search_terms_dict = {"query_terms": search_terms, "keywords": search_terms, "methods": []}
        return await self._optimized_search_arxiv(search_terms_dict, {}, max_results, target_year)
    
    async def _search_semantic_scholar(self, search_terms: List[str], max_results: int = 25, target_year: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search Semantic Scholar for related papers."""
        try:
            if not self.semantic_scholar:
                return []
            
            # Combine terms into query
            query = " ".join(search_terms[:5])
            
            # Search Semantic Scholar
            results = self.semantic_scholar.search_paper(query, limit=max_results)
            
            papers = []
            for paper in results:
                if paper and hasattr(paper, 'title'):
                    papers.append({
                        "title": paper.title,
                        "abstract": getattr(paper, 'abstract', '') or '',
                        "url": getattr(paper, 'url', '') or '',
                        "year": getattr(paper, 'year', None),
                        "authors": [author.get('name', '') for author in getattr(paper, 'authors', [])] if hasattr(paper, 'authors') else [],
                        "citations": getattr(paper, 'citationCount', 0) or 0,
                        "source": "semantic_scholar"
                    })
            
            logger.info(f"Found {len(papers)} papers from Semantic Scholar")
            return papers
        
        except Exception as e:
            logger.error(f"Error searching Semantic Scholar: {str(e)}")
            return []
    
    def _deduplicate_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate papers based on title similarity."""
        seen_titles = set()
        unique_papers = []
        
        for paper in papers:
            title_lower = paper["title"].lower().strip()
            # Simple deduplication: exact title match
            if title_lower not in seen_titles:
                seen_titles.add(title_lower)
                unique_papers.append(paper)
        
        return unique_papers
    
    def _filter_and_rank_papers(self, papers: List[Dict[str, Any]], research_context: Dict[str, Any], target_year: Optional[str] = None) -> List[Dict[str, Any]]:
        """Filter and rank papers by date relevance and similarity to target paper."""
        from datetime import datetime
        
        target_year_int = None
        if target_year:
            try:
                target_year_int = int(target_year)
            except:
                pass
        
        current_year = datetime.now().year
        
        def score_paper(paper: Dict[str, Any]) -> float:
            score = 0.0
            
            # Date-based scoring (critical for novelty assessment)
            paper_year = paper.get("year")
            if paper_year and target_year_int:
                year_diff = abs(paper_year - target_year_int)
                
                # Papers from same year or ±1 year get highest boost (for comparison)
                if year_diff == 0:
                    score += 20  # Same year - most relevant for novelty
                elif year_diff == 1:
                    score += 15  # Adjacent year
                elif year_diff <= 3:
                    score += 10  # Within 3 years
                elif year_diff <= 5:
                    score += 5   # Within 5 years
                
                # For new papers (target is recent), prioritize recent papers
                if target_year_int and target_year_int >= current_year - 2:
                    if paper_year >= current_year - 3:
                        score += 5  # Recent papers boost for new research
            
            # Citations boost (but less important than date relevance)
            citations = paper.get("citations", 0) or 0
            score += min(citations / 200, 5)  # Cap at 5 points
            
            # Title similarity boost (simple keyword matching)
            target_title = research_context.get("title", "").lower()
            paper_title = paper.get("title", "").lower()
            if target_title and paper_title:
                target_words = set(target_title.split())
                paper_words = set(paper_title.split())
                common_words = target_words.intersection(paper_words)
                if len(common_words) > 0:
                    score += min(len(common_words) * 2, 10)  # Up to 10 points for title similarity
            
            return score
        
        # Sort by score (descending)
        papers.sort(key=score_paper, reverse=True)
        
        # If target year is known, ensure we have papers from similar time periods
        if target_year_int:
            # Separate papers by time period
            same_period = [p for p in papers if p.get("year") and abs(p.get("year") - target_year_int) <= 2]
            other_papers = [p for p in papers if p not in same_period]
            
            # Prioritize same period papers but include others
            ranked_papers = same_period[:max(20, len(papers) // 2)] + other_papers
            return ranked_papers[:len(papers)]
        
        return papers
