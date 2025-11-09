"""
Optimized Research Expander: Uses LLM-extracted search terms and optimized arxiv search.
"""
import logging
from typing import List, Dict, Any, Optional, Callable, AsyncGenerator
import asyncio
from datetime import datetime, timedelta

import arxiv

from app.core.config import settings
from app.agents.industrynav.search_term_extractor import SearchTermExtractor

logger = logging.getLogger(__name__)


class OptimizedResearchExpander:
    """Optimized research expander with LLM-based term extraction and efficient search."""
    
    def __init__(self):
        """Initialize optimized research expander."""
        self.search_extractor = SearchTermExtractor()
        self.progress_callback: Optional[Callable] = None
    
    def set_progress_callback(self, callback: Callable):
        """Set callback for progress updates (for real-time visualization)."""
        self.progress_callback = callback
    
    async def _notify_progress(self, event: str, data: Dict[str, Any]):
        """Notify progress callback if set."""
        if self.progress_callback:
            try:
                await self.progress_callback(event, data)
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
        Expand research corpus with optimized search.
        
        Args:
            research_context: Research context with title, abstract, methods, etc.
            max_papers: Maximum number of papers to retrieve
            min_papers: Minimum number of papers to retrieve
            enable_full_text: Whether to scrape full text
        
        Returns:
            List of paper metadata dictionaries
        """
        logger.info(f"Optimized research expansion for: {research_context.get('title', 'Unknown')}")
        
        await self._notify_progress("extraction_started", {"message": "Extracting search terms..."})
        
        # Step 1: Extract precise search terms using LLM
        search_terms = await self.search_extractor.extract_search_terms(research_context)
        logger.info(f"Extracted search terms: {search_terms.get('query_terms', [])}")
        
        await self._notify_progress("extraction_complete", {
            "keywords": search_terms.get("keywords", []),
            "methods": search_terms.get("methods", []),
            "categories": search_terms.get("categories", [])
        })
        
        # Step 2: Optimized arxiv search
        await self._notify_progress("search_started", {"message": "Searching arXiv..."})
        
        papers = await self._optimized_arxiv_search(
            search_terms,
            research_context,
            max_papers=max_papers
        )
        
        await self._notify_progress("search_complete", {
            "papers_found": len(papers),
            "message": f"Found {len(papers)} papers"
        })
        
        # Step 3: Filter and rank
        publication_date = research_context.get("publication_date")
        papers = self._filter_and_rank_papers(papers, research_context, publication_date)
        papers = papers[:max_papers]
        
        # Ensure minimum papers
        if len(papers) < min_papers:
            logger.warning(f"Only found {len(papers)} papers, less than minimum {min_papers}")
        
        await self._notify_progress("ranking_complete", {
            "papers_ranked": len(papers),
            "top_papers": [{"title": p.get("title", "")[:50], "year": p.get("year")} for p in papers[:5]]
        })
        
        logger.info(f"Optimized expansion complete: {len(papers)} papers")
        return papers
    
    async def _optimized_arxiv_search(
        self,
        search_terms: Dict[str, Any],
        research_context: Dict[str, Any],
        max_papers: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Perform optimized arXiv search with categories, date ranges, and focused queries.
        
        Args:
            search_terms: Extracted search terms
            research_context: Research context
            max_papers: Maximum papers to retrieve
        
        Returns:
            List of paper dictionaries
        """
        papers = []
        
        # Get query terms
        query_terms = search_terms.get("query_terms", [])
        if not query_terms:
            logger.warning("No query terms extracted, using fallback")
            query_terms = search_terms.get("keywords", [])[:5]
        
        # Build optimized query
        # Use AND for methods, OR for keywords
        methods = search_terms.get("methods", [])[:2]  # Top 2 methods
        keywords = query_terms[:5]  # Top 5 keywords
        
        # Construct query: (method1 AND method2) OR (keyword1 OR keyword2 OR ...)
        query_parts = []
        if methods:
            methods_query = " AND ".join([f'"{m}"' for m in methods])
            query_parts.append(f"({methods_query})")
        if keywords:
            keywords_query = " OR ".join([f'"{k}"' for k in keywords])
            query_parts.append(f"({keywords_query})")
        
        query = " OR ".join(query_parts) if len(query_parts) > 1 else (query_parts[0] if query_parts else " ".join(query_terms[:3]))
        
        logger.info(f"Optimized query: {query}")
        
        # Get categories
        categories = search_terms.get("categories", [])
        
        # Calculate date range
        publication_date = research_context.get("publication_date")
        date_from, date_to = self._calculate_date_range(publication_date)
        
        await self._notify_progress("arxiv_search_started", {
            "query": query,
            "categories": categories,
            "date_range": {"from": date_from, "to": date_to}
        })
        
        try:
            # Run arXiv search with optimizations
            loop = asyncio.get_event_loop()
            
            # Build search with optimizations
            search_params = {
                "query": query,
                "max_results": min(max_papers, 100),  # Hard limit
                "sort_by": arxiv.SortCriterion.Relevance
            }
            
            # If categories available, we can't directly filter in arxiv.py,
            # but we can add category terms to query
            if categories:
                category_query = " OR ".join([f"cat:{cat}" for cat in categories])
                query = f"({query}) AND ({category_query})"
                search_params["query"] = query
            
            search = await loop.run_in_executor(
                None,
                lambda: arxiv.Search(**search_params)
            )
            
            # Process results as they come (for real-time visualization)
            results = await loop.run_in_executor(
                None,
                lambda: list(search.results())
            )
            
            for i, paper in enumerate(results):
                # Extract arXiv ID
                arxiv_id = None
                if paper.entry_id:
                    arxiv_id = paper.entry_id.split('/')[-1].replace('.pdf', '')
                
                # Extract categories from primary_category or categories
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
                
                # Notify progress for each paper found
                await self._notify_progress("paper_found", {
                    "paper": {
                        "title": paper_data["title"][:50],
                        "year": paper_data["year"],
                        "authors": paper_data["authors"][:3]
                    },
                    "count": len(papers),
                    "total": len(results)
                })
            
            logger.info(f"Found {len(papers)} papers from optimized arXiv search")
            return papers
        
        except Exception as e:
            logger.error(f"Error in optimized arXiv search: {str(e)}")
            await self._notify_progress("search_error", {"error": str(e)})
            return []
    
    def _calculate_date_range(self, publication_date: Optional[str]) -> tuple:
        """
        Calculate date range for search based on publication date.
        
        Args:
            publication_date: Publication year of target paper
        
        Returns:
            Tuple of (from_date, to_date) as strings
        """
        current_date = datetime.now()
        
        if publication_date:
            try:
                target_year = int(publication_date)
                # Search ±3 years around target year, but not before 2010
                from_year = max(2010, target_year - 3)
                to_year = min(current_date.year, target_year + 3)
                
                date_from = f"{from_year}-01-01"
                date_to = f"{to_year}-12-31"
                
                return date_from, date_to
            except:
                pass
        
        # Default: last 5 years
        date_from = (current_date - timedelta(days=5*365)).strftime("%Y-%m-%d")
        date_to = current_date.strftime("%Y-%m-%d")
        
        return date_from, date_to
    
    def _filter_and_rank_papers(self, papers: List[Dict[str, Any]], research_context: Dict[str, Any], target_year: Optional[str] = None) -> List[Dict[str, Any]]:
        """Filter and rank papers by date relevance and similarity."""
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
            
            # Date-based scoring
            paper_year = paper.get("year")
            if paper_year and target_year_int:
                year_diff = abs(paper_year - target_year_int)
                
                if year_diff == 0:
                    score += 20
                elif year_diff == 1:
                    score += 15
                elif year_diff <= 3:
                    score += 10
                elif year_diff <= 5:
                    score += 5
            
            # Category match boost
            target_categories = research_context.get("categories", [])
            paper_categories = paper.get("categories", [])
            if target_categories and paper_categories:
                matching_categories = set(target_categories) & set(paper_categories)
                score += len(matching_categories) * 3
            
            # Title similarity
            target_title = research_context.get("title", "").lower()
            paper_title = paper.get("title", "").lower()
            if target_title and paper_title:
                target_words = set(target_title.split())
                paper_words = set(paper_title.split())
                common_words = target_words.intersection(paper_words)
                if len(common_words) > 0:
                    score += min(len(common_words) * 2, 10)
            
            return score
        
        papers.sort(key=score_paper, reverse=True)
        return papers

