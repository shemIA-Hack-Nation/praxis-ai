"""
Wrapper for arxivscraper library for optimized paper scraping.
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

try:
    import arxivscraper
    HAS_ARXIVSCRAPER = True
except ImportError:
    HAS_ARXIVSCRAPER = False
    arxivscraper = None

logger = logging.getLogger(__name__)


class ArxivScraperWrapper:
    """Wrapper for arxivscraper with optimized parameters."""
    
    def __init__(self):
        """Initialize arxivscraper wrapper."""
        if not HAS_ARXIVSCRAPER:
            logger.warning("arxivscraper not installed. Install with: pip install arxivscraper")
    
    async def search_papers(
        self,
        query: str,
        categories: Optional[List[str]] = None,
        max_results: int = 50,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search arXiv papers using arxivscraper with optimized parameters.
        
        Args:
            query: Search query
            categories: arXiv categories (e.g., ['cs.AI', 'cs.CV'])
            max_results: Maximum number of results
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
        
        Returns:
            List of paper dictionaries
        """
        if not HAS_ARXIVSCRAPER:
            logger.warning("arxivscraper not available, falling back to arxiv library")
            return []
        
        try:
            import asyncio
            
            # Build scraper configuration
            scraper_config = {
                'query': query,
                'max_results': max_results,
            }
            
            # Add categories if provided
            if categories:
                scraper_config['categories'] = categories
            
            # Add date range if provided
            if date_from:
                scraper_config['date_from'] = date_from
            if date_to:
                scraper_config['date_to'] = date_to
            
            # Run scraper in executor (it's synchronous)
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self._run_scraper(scraper_config)
            )
            
            # Convert to our format
            papers = []
            for result in results:
                papers.append({
                    "title": result.get("title", ""),
                    "abstract": result.get("abstract", ""),
                    "url": result.get("url", ""),
                    "arxiv_id": result.get("arxiv_id", ""),
                    "year": result.get("year"),
                    "published": result.get("published"),
                    "authors": result.get("authors", []),
                    "categories": result.get("categories", []),
                    "citations": None,
                    "source": "arxiv"
                })
            
            logger.info(f"Found {len(papers)} papers using arxivscraper")
            return papers
        
        except Exception as e:
            logger.error(f"Error using arxivscraper: {str(e)}")
            return []
    
    def _run_scraper(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run arxivscraper synchronously."""
        try:
            # Create scraper
            scraper = arxivscraper.Scraper(
                category=config.get('categories', ['cs.AI'])[0] if config.get('categories') else 'cs.AI',
                date_from=config.get('date_from'),
                date_to=config.get('date_to'),
                filters={
                    'categories': config.get('categories', []),
                }
            )
            
            # Fetch results
            output = scraper.scrape()
            
            # Convert to list of dicts
            results = []
            for entry in output:
                results.append({
                    "title": entry.get("title", ""),
                    "abstract": entry.get("abstract", ""),
                    "url": entry.get("url", ""),
                    "arxiv_id": entry.get("arxiv_id", ""),
                    "year": entry.get("year"),
                    "published": entry.get("published"),
                    "authors": entry.get("authors", []),
                    "categories": entry.get("categories", []),
                })
            
            return results[:config.get('max_results', 50)]
        
        except Exception as e:
            logger.error(f"Error running arxivscraper: {str(e)}")
            return []

