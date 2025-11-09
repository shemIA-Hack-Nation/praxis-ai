"""
PDF Parser: Extracts content from uploaded research paper PDFs.
"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import asyncio

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

logger = logging.getLogger(__name__)


class PDFParser:
    """Parser for research paper PDF files."""
    
    def __init__(self):
        """Initialize PDF parser."""
        pass
    
    async def parse_paper(self, paper_path: str) -> Dict[str, Any]:
        """
        Parse a research paper PDF file.
        
        Args:
            paper_path: Path to the PDF file
        
        Returns:
            Dictionary containing paper structure and content
        """
        try:
            logger.info(f"Parsing research paper: {paper_path}")
            
            # Extract full text from PDF
            full_text = await self._extract_pdf_text(Path(paper_path))
            
            if not full_text:
                raise ValueError("Failed to extract text from PDF")
            
            # Extract basic metadata from filename
            filename = Path(paper_path).stem
            
            return {
                "filename": filename,
                "full_text": full_text,
                "text_length": len(full_text)
            }
        
        except Exception as e:
            logger.error(f"Error parsing paper: {str(e)}")
            raise
    
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
        text_parts = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
        except Exception as e:
            logger.error(f"Error with pdfplumber: {str(e)}")
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
    
    async def extract_research_context(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract comprehensive research context from paper (title, abstract, methods, results, dates, authors).
        
        Args:
            paper_data: Parsed paper data
        
        Returns:
            Dictionary with research context including publication date
        """
        full_text = paper_data.get("full_text", "")
        lines = full_text.split('\n')
        
        # Extract title (improved extraction)
        title = self._extract_title(lines, full_text)
        
        # Extract authors
        authors = self._extract_authors(lines)
        
        # Extract publication date
        publication_date = self._extract_publication_date(lines, full_text)
        
        # Extract abstract (improved)
        abstract = self._extract_abstract(lines, full_text)
        
        # Extract methods section
        methods = self._extract_section(lines, ["method", "methodology", "approach", "experimental"], 
                                        stop_keywords=["results", "discussion", "conclusion", "evaluation"])
        
        # Extract results section
        results = self._extract_section(lines, ["result", "finding", "experiment", "evaluation"],
                                       stop_keywords=["discussion", "conclusion", "related work", "future work"])
        
        # Extract introduction/key concepts
        introduction = self._extract_section(lines, ["introduction", "background", "related work"],
                                           stop_keywords=["method", "methodology", "approach", "experimental"])
        
        # Extract keywords/concepts
        keywords = self._extract_keywords(lines, full_text)
        
        return {
            "title": title,
            "authors": authors,
            "publication_date": publication_date,
            "abstract": abstract,
            "introduction": introduction,
            "methods": methods,
            "results": results,
            "keywords": keywords,
            "full_content": full_text,
            "filename": paper_data.get("filename", "")
        }
    
    def _extract_title(self, lines: list, full_text: str) -> str:
        """Extract paper title with improved accuracy, skipping watermarks."""
        import re
        
        # Skip watermarks and headers
        skip_patterns = [
            r"provided.*attribution.*provided",
            r"google.*hereby.*grants.*permission",
            r"copyright.*\d{4}",
            r"arxiv.*\d{4}",
            r"preprint",
            r"submitted.*to",
            r"under.*review",
        ]
        
        # Strategy 1: Find first substantial line that's not a watermark
        for i, line in enumerate(lines[:20]):
            line_clean = line.strip()
            # Skip if too short or too long
            if len(line_clean) < 15 or len(line_clean) > 300:
                continue
            
            # Skip if it matches watermark patterns
            line_lower = line_clean.lower()
            if any(re.search(pattern, line_lower) for pattern in skip_patterns):
                continue
            
            # Skip common headers
            if any(skip in line_lower for skip in ["abstract", "introduction", "author", "university", "department", "institute", "email", "@"]):
                continue
            
            # Skip if it's all caps and short (likely header)
            if line_clean.isupper() and len(line_clean.split()) < 5:
                continue
            
            # If it looks like a title (has multiple words, reasonable length)
            words = line_clean.split()
            if len(words) >= 3 and len(words) <= 30:
                # Clean up: remove extra spaces
                title = ' '.join(words)
                return title[:200]
        
        # Strategy 2: Look for title-like patterns (capitalized words)
        for i, line in enumerate(lines[:15]):
            line_clean = line.strip()
            if 20 <= len(line_clean) <= 200:
                # Check if it has title-like capitalization
                words = line_clean.split()
                if len(words) >= 3:
                    # Count capitalized words (title case)
                    capitalized = sum(1 for w in words if w and w[0].isupper())
                    if capitalized >= len(words) * 0.3:  # At least 30% capitalized
                        # Skip watermarks
                        if not any(re.search(pattern, line_clean.lower()) for pattern in skip_patterns):
                            return ' '.join(words)[:200]
        
        return "Research Paper"
    
    def _extract_authors(self, lines: list) -> list:
        """Extract author names from first page."""
        import re
        authors = []
        
        # Look for author patterns (usually after title, before abstract)
        # Common patterns: "Author1, Author2" or "Author1 and Author2" or "Author1 Author2"
        for i, line in enumerate(lines[:30]):
            line_clean = line.strip()
            if len(line_clean) < 5:
                continue
            
            # Skip watermarks
            if any(skip in line_clean.lower() for skip in ["provided", "attribution", "copyright", "arxiv", "preprint"]):
                continue
            
            # Pattern 1: Lines with commas (likely author list)
            if "," in line_clean and len(line_clean) < 200:
                # Split by comma
                parts = line_clean.split(",")
                for part in parts[:8]:  # Check up to 8 parts
                    part_clean = part.strip()
                    # Check if it looks like a name (has capital letters, reasonable length)
                    if 3 <= len(part_clean) <= 60:
                        # Remove email addresses
                        if "@" in part_clean:
                            part_clean = part_clean.split("@")[0].strip()
                        # Remove common suffixes
                        for suffix in ["university", "department", "institute", "lab", "laboratory", "email", "1", "2", "3"]:
                            if suffix in part_clean.lower():
                                part_clean = part_clean.split(suffix)[0].strip()
                        # Check if it has name-like structure (capitalized words)
                        words = part_clean.split()
                        if words and all(w[0].isupper() if w else False for w in words[:2]):  # First 2 words capitalized
                            if part_clean and part_clean not in authors:
                                authors.append(part_clean)
            
            # Pattern 2: Lines with "and" (Author1 and Author2)
            elif " and " in line_clean.lower() and len(line_clean) < 200:
                # Split by "and"
                parts = re.split(r'\s+and\s+', line_clean, flags=re.IGNORECASE)
                for part in parts[:5]:
                    part_clean = part.strip()
                    if 3 <= len(part_clean) <= 60:
                        words = part_clean.split()
                        if words and all(w[0].isupper() if w else False for w in words[:2]):
                            if part_clean and part_clean not in authors:
                                authors.append(part_clean)
            
            # Pattern 3: Lines with multiple capitalized words (potential author names)
            elif len(line_clean.split()) >= 2 and len(line_clean.split()) <= 10:
                words = line_clean.split()
                # Check if most words start with capital (name-like)
                capitalized = sum(1 for w in words if w and w[0].isupper())
                if capitalized >= len(words) * 0.7:  # 70% capitalized
                    # Skip if it's clearly not a name
                    if not any(skip in line_clean.lower() for skip in ["abstract", "introduction", "section", "figure", "table"]):
                        # Check if it has name-like structure
                        if all(3 <= len(w) <= 20 for w in words[:3]):  # Reasonable word lengths
                            if line_clean not in authors:
                                authors.append(line_clean)
            
            if len(authors) >= 5:  # Found enough authors
                break
        
        return authors[:5]  # Limit to 5 authors
    
    def _extract_publication_date(self, lines: list, full_text: str) -> Optional[str]:
        """Extract publication date from paper."""
        import re
        from datetime import datetime
        
        # Look for date patterns in first 50 lines
        date_patterns = [
            r'\b(19|20)\d{2}\b',  # 4-digit year
            r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b',  # Month Day, Year
            r'\b\d{1,2}[/-]\d{1,2}[/-](19|20)\d{2}\b',  # MM/DD/YYYY or DD/MM/YYYY
        ]
        
        search_text = '\n'.join(lines[:50])
        
        for pattern in date_patterns:
            matches = re.findall(pattern, search_text, re.IGNORECASE)
            if matches:
                # Try to parse the most recent date (likely publication date)
                for match in reversed(matches):
                    if isinstance(match, tuple):
                        year = match[0] + match[1] if len(match) > 1 else match[0]
                    else:
                        year = match
                    
                    try:
                        year_int = int(year) if len(year) == 4 else int("20" + year[-2:])
                        if 1900 <= year_int <= datetime.now().year + 1:
                            return str(year_int)
                    except:
                        continue
        
        # Fallback: look for arXiv date pattern
        arxiv_match = re.search(r'ar[Xx]iv:.*?(\d{4})\.(\d{4,5})', full_text)
        if arxiv_match:
            return arxiv_match.group(1)  # Return year
        
        return None
    
    def _extract_abstract(self, lines: list, full_text: str) -> str:
        """Extract abstract with improved accuracy."""
        abstract = ""
        abstract_started = False
        abstract_keywords = ["abstract"]
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            line_clean = line.strip()
            
            # Check if this is the abstract header
            if any(keyword == line_lower for keyword in abstract_keywords) and len(line_clean) < 20:
                abstract_started = True
                continue
            
            if abstract_started:
                # Stop at next major section
                if any(section in line_lower for section in ["introduction", "1.", "keywords", "index terms", "i. introduction"]):
                    if len(abstract) > 100:
                        break
                
                # Collect abstract content
                if line_clean:
                    # Skip very short lines that are likely section markers
                    if len(line_clean) > 3:
                        abstract += line_clean + " "
                        if len(abstract) > 1500:  # Limit abstract length
                            break
                elif len(abstract) > 200:  # Stop if we have content and hit empty line
                    break
        
        # If no abstract found, use first substantial paragraph
        if not abstract or len(abstract) < 100:
            for line in lines[5:30]:  # Skip title area
                if len(line.strip()) > 100:
                    abstract = line.strip()[:1000]
                    break
        
        return abstract.strip()[:1500] if abstract else ""
    
    def _extract_section(self, lines: list, start_keywords: list, stop_keywords: list, max_length: int = 3000) -> str:
        """Extract a section of the paper."""
        section_text = ""
        section_started = False
        
        for line in lines:
            line_lower = line.lower().strip()
            line_clean = line.strip()
            
            # Check if section starts
            if not section_started:
                if any(keyword in line_lower for keyword in start_keywords) and len(line_clean) < 50:
                    section_started = True
                    continue
            
            if section_started:
                # Stop at next major section
                if any(stop in line_lower for stop in stop_keywords):
                    if len(section_text) > 200:
                        break
                
                # Collect section content
                if line_clean and not line_clean.startswith(('Figure', 'Table', 'Fig.', 'Tab.')):
                    section_text += line_clean + " "
                    if len(section_text) > max_length:
                        break
        
        return section_text.strip()[:max_length]
    
    def _extract_keywords(self, lines: list, full_text: str) -> list:
        """Extract keywords from paper."""
        keywords = []
        
        # Look for keywords section
        keywords_started = False
        for line in lines:
            line_lower = line.lower().strip()
            if "keyword" in line_lower and len(line.strip()) < 30:
                keywords_started = True
                continue
            
            if keywords_started:
                line_clean = line.strip()
                if line_clean:
                    # Extract keywords (comma or semicolon separated)
                    if "," in line_clean:
                        keywords.extend([k.strip() for k in line_clean.split(",")])
                    elif ";" in line_clean:
                        keywords.extend([k.strip() for k in line_clean.split(";")])
                    else:
                        keywords.append(line_clean)
                    
                    if len(keywords) >= 10:  # Limit keywords
                        break
                elif len(keywords) > 0:
                    break
        
        # Clean and filter keywords
        cleaned_keywords = []
        for kw in keywords[:10]:
            kw_clean = kw.strip().strip(".,;:").lower()
            if 2 < len(kw_clean) < 50 and kw_clean not in ["keywords", "index terms"]:
                cleaned_keywords.append(kw_clean)
        
        return cleaned_keywords
