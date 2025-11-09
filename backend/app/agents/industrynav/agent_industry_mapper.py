"""
Industry Mapper Agent: Maps research to industry applications and market opportunities.
"""
import logging
from typing import List, Dict, Any, Optional

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
from app.core.llm_providers import get_llm

logger = logging.getLogger(__name__)


class IndustryMapperAgent:
    """Agent responsible for mapping research to industry applications."""
    
    def __init__(self):
        """Initialize industry mapper."""
        self.llm = get_llm(temperature=0.3)
        
        # Industry taxonomy (can be expanded)
        self.industry_domains = [
            "Healthcare & Biotechnology",
            "Finance & Fintech",
            "Manufacturing & Industry 4.0",
            "Energy & Sustainability",
            "Transportation & Logistics",
            "Agriculture & Food Tech",
            "Education & EdTech",
            "Entertainment & Media",
            "Security & Defense",
            "Retail & E-commerce",
            "Real Estate & Construction",
            "Telecommunications",
            "Automotive",
            "Aerospace",
            "Software & IT Services"
        ]
    
    async def map_industry(
        self,
        research_context: Dict[str, Any],
        parsed_data: Dict[str, Any],
        novelty_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Map research to industry applications.
        
        Args:
            research_context: Research context from notebook
            parsed_data: Parsed data with papers and KG
            novelty_result: Novelty assessment results
        
        Returns:
            Dictionary with industry applications and opportunities
        """
        logger.info("Mapping research to industry applications")
        
        # Extract key information
        title = research_context.get("title", "")
        abstract = research_context.get("abstract", "")
        methods = research_context.get("methods", "")
        concepts = parsed_data.get("papers", [{}])[0].get("concepts", [])
        
        # Identify relevant industries
        relevant_industries = await self._identify_industries(
            title, abstract, methods, concepts
        )
        
        # Generate use cases
        use_cases = await self._generate_use_cases(
            title, abstract, methods, relevant_industries
        )
        
        # Identify market opportunities
        market_opportunities = await self._identify_market_opportunities(
            title, abstract, use_cases, novelty_result
        )
        
        # Find interdisciplinary connections
        interdisciplinary = await self._find_interdisciplinary_connections(
            parsed_data, concepts
        )
        
        return {
            "relevant_industries": relevant_industries,
            "use_cases": use_cases,
            "market_opportunities": market_opportunities,
            "interdisciplinary_connections": interdisciplinary
        }
    
    async def _identify_industries(
        self,
        title: str,
        abstract: str,
        methods: str,
        concepts: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Identify relevant industry domains.
        
        Args:
            title: Paper title
            abstract: Paper abstract
            methods: Research methods
            concepts: Key concepts
        
        Returns:
            List of relevant industries with relevance scores
        """
        if not self.llm:
            # Fallback: simple keyword matching
            return self._identify_industries_keyword(title, abstract, methods, concepts)
        
        try:
            prompt = f"""Analyze the following research and identify the top 3-5 most relevant industry domains from this list:
{', '.join(self.industry_domains)}

Research Title: {title}
Abstract: {abstract[:500]}
Key Concepts: {', '.join(concepts[:10])}

For each relevant industry, provide:
1. Industry name
2. Relevance score (1-10)
3. Brief explanation of why it's relevant

Respond in JSON format:
{{
    "industries": [
        {{
            "name": "Industry Name",
            "score": 8,
            "explanation": "Brief explanation"
        }}
    ]
}}"""
            
            messages = [
                SystemMessage(content="You are an expert industry analyst. Identify relevant industries for research work."),
                HumanMessage(content=prompt)
            ]
            
            # Handle both sync and async LLM calls
            try:
                if hasattr(self.llm, 'ainvoke'):
                    response = await self.llm.ainvoke(messages)
                else:
                    response = self.llm.invoke(messages)
                content = response.content
            except Exception as e:
                logger.error(f"Error calling LLM: {str(e)}")
                raise
            
            # Parse JSON response
            import json
            try:
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                result = json.loads(content)
                industries = result.get("industries", [])
                
                # Sort by score
                industries.sort(key=lambda x: x.get("score", 0), reverse=True)
                return industries[:5]
            
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON from LLM response: {content}")
                return self._identify_industries_keyword(title, abstract, methods, concepts)
        
        except Exception as e:
            logger.error(f"Error identifying industries: {str(e)}")
            return self._identify_industries_keyword(title, abstract, methods, concepts)
    
    def _identify_industries_keyword(
        self,
        title: str,
        abstract: str,
        methods: str,
        concepts: List[str]
    ) -> List[Dict[str, Any]]:
        """Fallback keyword-based industry identification."""
        text = f"{title} {abstract} {methods}".lower()
        
        industry_keywords = {
            "Healthcare & Biotechnology": ["health", "medical", "biotech", "clinical", "patient", "diagnosis", "treatment"],
            "Finance & Fintech": ["finance", "financial", "banking", "trading", "investment", "cryptocurrency", "blockchain"],
            "Manufacturing & Industry 4.0": ["manufacturing", "production", "factory", "industrial", "automation"],
            "Energy & Sustainability": ["energy", "solar", "wind", "renewable", "sustainability", "climate", "carbon"],
            "Transportation & Logistics": ["transport", "logistics", "shipping", "delivery", "vehicle", "traffic"],
            "Agriculture & Food Tech": ["agriculture", "farming", "crop", "food", "agricultural"],
            "Software & IT Services": ["software", "application", "system", "platform", "digital", "cloud"]
        }
        
        scores = {}
        for industry, keywords in industry_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                scores[industry] = min(score * 2, 10)
        
        # Sort by score
        industries = [
            {"name": industry, "score": score, "explanation": f"Relevant keywords found in research"}
            for industry, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return industries[:5]
    
    async def _generate_use_cases(
        self,
        title: str,
        abstract: str,
        methods: str,
        industries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate use cases for relevant industries."""
        if not self.llm or not industries:
            return []
        
        try:
            industry_names = [ind["name"] for ind in industries[:3]]
            
            prompt = f"""Generate 2-3 specific use cases for each of these industries: {', '.join(industry_names)}

Research Title: {title}
Abstract: {abstract[:500]}
Methods: {methods[:300]}

For each use case, provide:
1. Industry
2. Use case name
3. Description
4. Potential impact

Respond in JSON format:
{{
    "use_cases": [
        {{
            "industry": "Industry Name",
            "name": "Use Case Name",
            "description": "Description",
            "impact": "High/Medium/Low"
        }}
    ]
}}"""
            
            messages = [
                SystemMessage(content="You are an expert business analyst. Generate practical use cases for research."),
                HumanMessage(content=prompt)
            ]
            
            # Handle both sync and async LLM calls
            try:
                if hasattr(self.llm, 'ainvoke'):
                    response = await self.llm.ainvoke(messages)
                else:
                    response = self.llm.invoke(messages)
                content = response.content
            except Exception as e:
                logger.error(f"Error calling LLM: {str(e)}")
                raise
            
            import json
            try:
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                result = json.loads(content)
                return result.get("use_cases", [])
            
            except json.JSONDecodeError:
                logger.warning("Failed to parse use cases JSON")
                return []
        
        except Exception as e:
            logger.error(f"Error generating use cases: {str(e)}")
            return []
    
    async def _identify_market_opportunities(
        self,
        title: str,
        abstract: str,
        use_cases: List[Dict[str, Any]],
        novelty_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify market opportunities."""
        opportunities = []
        
        # High novelty research has higher market potential
        novelty_score = novelty_result.get("score", 5.0)
        
        if novelty_score >= 8:
            opportunities.append({
                "type": "Innovation Opportunity",
                "description": "High novelty research with potential for breakthrough applications",
                "market_size": "Large",
                "timeline": "2-5 years"
            })
        elif novelty_score >= 6:
            opportunities.append({
                "type": "Improvement Opportunity",
                "description": "Moderate novelty with potential for incremental improvements",
                "market_size": "Medium",
                "timeline": "1-3 years"
            })
        
        # Add opportunities based on use cases
        if use_cases:
            high_impact_cases = [uc for uc in use_cases if uc.get("impact") == "High"]
            if high_impact_cases:
                opportunities.append({
                    "type": "High-Impact Applications",
                    "description": f"{len(high_impact_cases)} high-impact use cases identified",
                    "market_size": "Variable",
                    "timeline": "1-4 years"
                })
        
        return opportunities
    
    async def _find_interdisciplinary_connections(
        self,
        parsed_data: Dict[str, Any],
        concepts: List[str]
    ) -> List[Dict[str, Any]]:
        """Find interdisciplinary connections in the knowledge graph."""
        kg_data = parsed_data.get("kg_data", {})
        edges = kg_data.get("edges", [])
        
        # Find connections between different domains
        connections = []
        
        # This is a simplified version - in practice, you'd analyze the KG structure
        # to find weak links between different research domains
        
        if concepts:
            connections.append({
                "type": "Concept Integration",
                "description": f"Research integrates {len(concepts)} key concepts from multiple domains",
                "potential": "High"
            })
        
        return connections
