"""
LLM-based Search Term Extractor: Extracts precise search terms from research papers.
"""
import logging
from typing import Dict, Any, List, Optional
import json
import re

from app.core.llm_providers import get_llm

logger = logging.getLogger(__name__)

# arXiv category mapping
ARXIV_CATEGORIES = {
    "cs.AI": "Artificial Intelligence",
    "cs.CV": "Computer Vision",
    "cs.CL": "Computation and Language",
    "cs.LG": "Machine Learning",
    "cs.NE": "Neural and Evolutionary Computing",
    "stat.ML": "Machine Learning (Statistics)",
    "cs.IR": "Information Retrieval",
    "cs.CY": "Computers and Society",
    "cs.DS": "Data Structures and Algorithms",
    "math.OC": "Optimization and Control",
}


class SearchTermExtractor:
    """Extracts precise search terms using LLM for optimized paper search."""
    
    def __init__(self):
        """Initialize search term extractor."""
        self.llm = get_llm(temperature=0.3)  # Lower temperature for more deterministic extraction
    
    async def extract_search_terms(self, research_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured search terms from research context using LLM.
        
        Args:
            research_context: Research context with title, abstract, methods, etc.
        
        Returns:
            Dictionary with:
            - keywords: List of core keywords (5-10)
            - methods: List of specific methods/models
            - categories: List of arXiv categories
            - authors: List of authors
            - query_terms: Combined query terms for search
        """
        try:
            title = research_context.get("title", "")
            abstract = research_context.get("abstract", "")
            methods = research_context.get("methods", "")
            authors = research_context.get("authors", [])
            keywords = research_context.get("keywords", [])
            
            # Build prompt for LLM
            prompt = self._build_extraction_prompt(title, abstract, methods, keywords)
            
            # Extract using LLM
            if self.llm:
                try:
                    from langchain_core.messages import HumanMessage, SystemMessage
                    
                    messages = [
                        SystemMessage(content="Extract search terms. Return JSON only."),  # Shorter system message
                        HumanMessage(content=prompt)
                    ]
                    
                    if hasattr(self.llm, 'ainvoke'):
                        response = await self.llm.ainvoke(messages)
                    else:
                        response = self.llm.invoke(messages)
                    
                    content = response.content if hasattr(response, 'content') else str(response)
                    
                    # Parse JSON response
                    extracted = self._parse_llm_response(content)
                    
                    # Merge with existing data
                    if keywords:
                        extracted["keywords"] = list(set(extracted.get("keywords", []) + keywords))
                    if authors:
                        extracted["authors"] = list(set(extracted.get("authors", []) + authors))
                    
                    # Infer arXiv categories
                    extracted["categories"] = self._infer_categories(extracted, research_context)
                    
                    # Build optimized query
                    extracted["query_terms"] = self._build_query_terms(extracted)
                    
                    return extracted
                except Exception as e:
                    logger.warning(f"Error using LLM for extraction: {str(e)}, falling back to rule-based")
            
            # Fallback to rule-based extraction
            return self._rule_based_extraction(research_context)
        
        except Exception as e:
            logger.error(f"Error extracting search terms: {str(e)}")
            return self._rule_based_extraction(research_context)
    
    def _build_extraction_prompt(self, title: str, abstract: str, methods: str, keywords: List[str]) -> str:
        """Build prompt for LLM extraction (optimized for token usage)."""
        # Truncate inputs to save tokens
        truncated_title = title[:150]
        truncated_abstract = abstract[:400]  # Reduced from 500
        truncated_methods = methods[:200]  # Reduced from 300
        keywords_str = ', '.join(keywords[:5]) if keywords else 'None'  # Limit to 5 keywords
        
        return f"""Extract search terms. Return JSON only:
{{
    "keywords": ["term1", "term2"],
    "methods": ["method1", "method2"],
    "techniques": ["tech1"],
    "domains": ["domain1"]
}}

Title: {truncated_title}
Abstract: {truncated_abstract}
Methods: {truncated_methods}
Keywords: {keywords_str}"""

    def _parse_llm_response(self, content: str) -> Dict[str, Any]:
        """Parse LLM response into structured data."""
        try:
            import re
            
            # The LLM might return the full conversation, extract just the JSON
            content = content.strip()
            
            # Strategy 1: Find JSON object in the response
            # Look for { ... } pattern with "keywords" or "concepts" or "methods"
            json_patterns = [
                r'\{[^{}]*"(?:keywords|concepts|methods)"[^{}]*\}',  # Simple JSON
                r'\{.*?"(?:keywords|concepts|methods)".*?\}',  # JSON with nested structures
            ]
            
            json_match = None
            for pattern in json_patterns:
                json_match = re.search(pattern, content, re.DOTALL)
                if json_match:
                    content = json_match.group(0)
                    break
            
            # If no match, try to find any JSON object
            if not json_match:
                json_match = re.search(r'\{.*?\}', content, re.DOTALL)
                if json_match:
                    content = json_match.group(0)
            
            # Remove markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            # Remove any text before the first {
            if "{" in content:
                content = content[content.index("{"):]
            # Remove any text after the last }
            if "}" in content:
                content = content[:content.rindex("}") + 1]
            
            # Clean up: remove any leading/trailing non-JSON text
            content = content.strip()
            
            # Parse JSON
            data = json.loads(content)
            
            # Ensure all fields are lists
            result = {
                "keywords": data.get("keywords", []),
                "methods": data.get("methods", []) + data.get("techniques", []),
                "domains": data.get("domains", []),
                "authors": [],
                "categories": []
            }
            
            # Limit and clean
            result["keywords"] = [k.strip().lower() for k in result["keywords"][:10] if k]
            result["methods"] = [m.strip() for m in result["methods"][:10] if m]
            result["domains"] = [d.strip() for d in result["domains"][:5] if d]
            
            return result
        
        except Exception as e:
            logger.warning(f"Error parsing LLM response: {str(e)}")
            return {
                "keywords": [],
                "methods": [],
                "domains": [],
                "authors": [],
                "categories": []
            }
    
    def _infer_categories(self, extracted: Dict[str, Any], research_context: Dict[str, Any]) -> List[str]:
        """Infer arXiv categories from extracted terms and context."""
        categories = []
        
        # Keyword-based inference
        keywords = [k.lower() for k in extracted.get("keywords", [])]
        methods = [m.lower() for m in extracted.get("methods", [])]
        all_terms = keywords + methods
        
        # Map terms to categories
        term_to_category = {
            "neural network": ["cs.NE", "cs.LG"],
            "transformer": ["cs.CL", "cs.LG", "cs.AI"],
            "cnn": ["cs.CV", "cs.LG"],
            "rnn": ["cs.CL", "cs.LG"],
            "lstm": ["cs.CL", "cs.LG"],
            "bert": ["cs.CL", "cs.AI"],
            "gpt": ["cs.CL", "cs.AI"],
            "computer vision": ["cs.CV"],
            "image": ["cs.CV"],
            "object detection": ["cs.CV"],
            "natural language": ["cs.CL"],
            "nlp": ["cs.CL"],
            "language model": ["cs.CL", "cs.AI"],
            "machine learning": ["cs.LG", "stat.ML"],
            "deep learning": ["cs.LG", "cs.NE"],
            "reinforcement learning": ["cs.LG", "cs.AI"],
            "optimization": ["math.OC", "cs.DS"],
            "graph": ["cs.DS", "cs.AI"],
            "recommendation": ["cs.IR"],
        }
        
        for term in all_terms:
            for key, cats in term_to_category.items():
                if key in term:
                    categories.extend(cats)
        
        # Remove duplicates and limit
        categories = list(set(categories))[:3]  # Top 3 categories
        
        return categories
    
    def _build_query_terms(self, extracted: Dict[str, Any]) -> List[str]:
        """Build optimized query terms for search."""
        query_terms = []
        
        # Prioritize methods and specific techniques
        methods = extracted.get("methods", [])
        if methods:
            query_terms.extend(methods[:3])
        
        # Add top keywords
        keywords = extracted.get("keywords", [])
        if keywords:
            query_terms.extend(keywords[:5])
        
        # Remove duplicates and limit
        query_terms = list(dict.fromkeys(query_terms))[:8]  # Preserve order, limit to 8
        
        return query_terms
    
    def _rule_based_extraction(self, research_context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback rule-based extraction."""
        keywords = research_context.get("keywords", [])
        authors = research_context.get("authors", [])
        title = research_context.get("title", "")
        methods = research_context.get("methods", "")
        
        # Extract from title
        title_words = [w.strip('.,;:()[]') for w in title.lower().split() 
                      if len(w) > 3 and w not in ["the", "and", "for", "with", "using", "based", "novel", "new"]]
        
        # Extract methods (capitalized terms)
        method_terms = []
        if methods:
            method_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', methods[:500])
            method_terms = [m.lower() for m in method_terms[:5]]
        
        # Combine
        all_keywords = list(set(keywords + title_words[:5]))
        
        return {
            "keywords": all_keywords[:10],
            "methods": method_terms,
            "domains": [],
            "authors": authors[:5],
            "categories": self._infer_categories({"keywords": all_keywords, "methods": method_terms}, research_context),
            "query_terms": (method_terms[:3] + all_keywords[:5])[:8]
        }

