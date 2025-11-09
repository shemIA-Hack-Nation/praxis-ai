"""
Paper Scraper Service: Downloads and extracts full text from research papers.
"""
import logging
import asyncio
import aiohttp
import aiofiles
from typing import Dict, Any, Optional, List
from pathlib import Path
import re
import tempfile
import os

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

from app.core.config import settings

logger = logging.getLogger(__name__)


class PaperScraper:
    """Service for scraping and extracting full text from research papers."""
    
    def __init__(self):
        """Initialize paper scraper."""
        self.temp_dir = Path(tempfile.gettempdir()) / "praxis_papers"
        self.temp_dir.mkdir(exist_ok=True)
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'Praxis-AI/1.0'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def scrape_arxiv_paper(self, arxiv_id: str) -> Dict[str, Any]:
        """
        Scrape full text from an arXiv paper.
        
        Args:
            arxiv_id: arXiv paper ID (e.g., "2301.12345" or "2301.12345v1")
        
        Returns:
            Dictionary with full text and metadata
        """
        try:
            # Clean arxiv_id (remove version if present)
            arxiv_id = arxiv_id.split('v')[0]
            
            # Construct PDF URL
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            
            # Download PDF
            pdf_path = await self._download_pdf(pdf_url, f"arxiv_{arxiv_id}")
            
            if not pdf_path:
                logger.warning(f"Failed to download PDF for arXiv {arxiv_id}")
                return {}
            
            # Extract text from PDF
            full_text = await self._extract_pdf_text(pdf_path)
            
            # Clean up
            if pdf_path.exists():
                pdf_path.unlink()
            
            return {
                "arxiv_id": arxiv_id,
                "full_text": full_text,
                "pdf_url": pdf_url,
                "text_length": len(full_text) if full_text else 0
            }
        
        except Exception as e:
            logger.error(f"Error scraping arXiv paper {arxiv_id}: {str(e)}")
            return {}
    
    async def scrape_semantic_scholar_paper(self, paper_id: str, pdf_url: Optional[str] = None) -> Dict[str, Any]:
        """
        Scrape full text from a Semantic Scholar paper.
        
        Args:
            paper_id: Semantic Scholar paper ID
            pdf_url: Optional direct PDF URL
        
        Returns:
            Dictionary with full text and metadata
        """
        try:
            if not pdf_url:
                # Try to get PDF URL from Semantic Scholar
                pdf_url = await self._get_semantic_scholar_pdf_url(paper_id)
            
            if not pdf_url:
                logger.warning(f"No PDF URL found for paper {paper_id}")
                return {}
            
            # Download PDF
            pdf_path = await self._download_pdf(pdf_url, f"ss_{paper_id}")
            
            if not pdf_path:
                logger.warning(f"Failed to download PDF for paper {paper_id}")
                return {}
            
            # Extract text from PDF
            full_text = await self._extract_pdf_text(pdf_path)
            
            # Clean up
            if pdf_path.exists():
                pdf_path.unlink()
            
            return {
                "paper_id": paper_id,
                "full_text": full_text,
                "pdf_url": pdf_url,
                "text_length": len(full_text) if full_text else 0
            }
        
        except Exception as e:
            logger.error(f"Error scraping Semantic Scholar paper {paper_id}: {str(e)}")
            return {}
    
    async def _download_pdf(self, url: str, filename: str) -> Optional[Path]:
        """Download PDF from URL."""
        session_to_use = self.session
        should_close = False
        
        if not session_to_use:
            session_to_use = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={'User-Agent': 'Praxis-AI/1.0'}
            )
            should_close = True
        
        try:
            async with session_to_use.get(url) as response:
                if response.status == 200:
                    pdf_path = self.temp_dir / f"{filename}.pdf"
                    async with aiofiles.open(pdf_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            await f.write(chunk)
                    return pdf_path
                else:
                    logger.warning(f"Failed to download PDF: HTTP {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error downloading PDF from {url}: {str(e)}")
            return None
        finally:
            if should_close and session_to_use:
                await session_to_use.close()
    
    async def _extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text from PDF file."""
        try:
            # Run PDF extraction in executor since it's synchronous
            loop = asyncio.get_event_loop()
            
            # Try pdfplumber first (better for academic papers)
            if HAS_PDFPLUMBER:
                return await loop.run_in_executor(None, self._extract_with_pdfplumber_sync, pdf_path)
            elif HAS_PYPDF2:
                return await loop.run_in_executor(None, self._extract_with_pypdf2_sync, pdf_path)
            else:
                logger.warning("No PDF parsing library available")
                return ""
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def _extract_with_pdfplumber_sync(self, pdf_path: Path) -> str:
        """Extract text using pdfplumber (synchronous)."""
        # Suppress pdfminer warnings about invalid color values (harmless PDF parsing issues)
        import logging
        pdfminer_logger = logging.getLogger('pdfminer')
        original_level = pdfminer_logger.level
        pdfminer_logger.setLevel(logging.ERROR)  # Only show errors, not warnings
        
        text_parts = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
        except Exception as e:
            logger.error(f"Error with pdfplumber: {str(e)}")
        finally:
            # Restore original logging level
            pdfminer_logger.setLevel(original_level)
        return "\n\n".join(text_parts)
    
    def _extract_with_pypdf2_sync(self, pdf_path: Path) -> str:
        """Extract text using PyPDF2 (synchronous)."""
        text_parts = []
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
        except Exception as e:
            logger.error(f"Error with PyPDF2: {str(e)}")
        return "\n\n".join(text_parts)
    
    async def _get_semantic_scholar_pdf_url(self, paper_id: str) -> Optional[str]:
        """Get PDF URL from Semantic Scholar paper ID."""
        # This would require Semantic Scholar API or web scraping
        # For now, return None - can be enhanced later
        return None
    
    async def scrape_papers_batch(self, papers: List[Dict[str, Any]], max_workers: int = 5) -> List[Dict[str, Any]]:
        """
        Scrape multiple papers in parallel.
        
        Args:
            papers: List of paper dictionaries with arxiv_id or paper_id
            max_workers: Maximum number of concurrent downloads
        
        Returns:
            List of papers with full_text added
        """
        semaphore = asyncio.Semaphore(max_workers)
        
        async def scrape_with_semaphore(paper: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                arxiv_id = paper.get("arxiv_id")
                paper_id = paper.get("paper_id")
                source = paper.get("source")
                
                if source == "arxiv" and arxiv_id:
                    # Extract arxiv_id from URL if needed
                    if not arxiv_id and paper.get("url"):
                        # Extract from URL like https://arxiv.org/abs/2301.12345
                        match = re.search(r'arxiv\.org/abs/(\d+\.\d+)', paper.get("url", ""))
                        if match:
                            arxiv_id = match.group(1)
                    
                    if arxiv_id:
                        scraped_data = await self.scrape_arxiv_paper(arxiv_id)
                        paper.update(scraped_data)
                
                elif source == "semantic_scholar" and paper_id:
                    pdf_url = paper.get("pdf_url")
                    scraped_data = await self.scrape_semantic_scholar_paper(paper_id, pdf_url)
                    paper.update(scraped_data)
                
                return paper
        
        tasks = [scrape_with_semaphore(paper) for paper in papers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out errors
        scraped_papers = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error in batch scraping: {str(result)}")
                continue
            scraped_papers.append(result)
        
        return scraped_papers
