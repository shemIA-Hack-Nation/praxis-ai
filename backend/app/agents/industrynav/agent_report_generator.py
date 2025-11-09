"""
Report Generator Agent: Generates markdown reports for paper and analysis.
"""
import logging
from typing import Dict, Any, List
from datetime import datetime

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


class ReportGeneratorAgent:
    """Agent responsible for generating final reports."""
    
    def __init__(self):
        """Initialize report generator."""
        self.llm = get_llm(temperature=0.7)
    
    async def generate_reports(
        self,
        paper_data: Dict[str, Any],
        research_context: Dict[str, Any],
        paper_corpus: List[Dict[str, Any]],
        parsed_data: Dict[str, Any],
        novelty_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate novelty analysis and knowledge graph.
        
        Args:
            paper_data: Parsed paper data (PDF content)
            research_context: Research context
            paper_corpus: Paper corpus
            parsed_data: Parsed data with KG
            novelty_result: Novelty assessment
        
        Returns:
            Dictionary with 'novelty_analysis' and 'knowledge_graph' data
        """
        logger.info("Generating novelty analysis and knowledge graph")
        
        # Generate novelty analysis
        analysis = await self._generate_analysis(
            research_context, paper_corpus, parsed_data, novelty_result
        )
        
        # Get knowledge graph data
        kg_data = parsed_data.get("kg_data", {})
        
        # Enhance knowledge graph with paper relationships
        enhanced_kg = self._enhance_knowledge_graph(
            research_context, paper_corpus, kg_data, novelty_result, parsed_data
        )
        
        return {
            "novelty_analysis": analysis,
            "knowledge_graph": enhanced_kg
        }
    
    async def _generate_paper(
        self,
        paper_data: Dict[str, Any],
        research_context: Dict[str, Any],
        paper_corpus: List[Dict[str, Any]],
        parsed_data: Dict[str, Any]
    ) -> str:
        """
        Generate research paper summary in markdown format.
        
        Args:
            paper_data: Parsed paper data (PDF content)
            research_context: Research context
            paper_corpus: Related papers
            parsed_data: Parsed data
        
        Returns:
            Markdown formatted research paper summary
        """
        if not self.llm:
            # Fallback: simple template-based paper
            return self._generate_paper_template(research_context, paper_data)
        
        try:
            title = research_context.get("title", "Research Paper")
            abstract = research_context.get("abstract", "")
            methods = research_context.get("methods", "")
            full_content = paper_data.get("full_text", "")[:1500]  # Reduced from 3000 to save tokens
            
            # Truncate inputs to save tokens
            truncated_title = title[:100]
            truncated_abstract = abstract[:300]  # Reduced from full abstract
            truncated_methods = methods[:500]  # Reduced from 1000
            truncated_content = full_content[:1000]  # Already limited above
            
            prompt = f"""Generate research summary in markdown:

Title: {truncated_title}
Abstract: {truncated_abstract}
Methods: {truncated_methods}
Content: {truncated_content}
Related Papers: {len(paper_corpus)}

Sections: Title, Abstract, Introduction, Methodology, Results, Discussion, Conclusion, References.
Use markdown. Be concise."""
            
            messages = [
                SystemMessage(content="Generate research summary in markdown. Be concise."),  # Shorter system message
                HumanMessage(content=prompt)
            ]
            
            # Handle both sync and async LLM calls
            try:
                if hasattr(self.llm, 'ainvoke'):
                    response = await self.llm.ainvoke(messages)
                else:
                    response = self.llm.invoke(messages)
                paper_content = response.content
            except Exception as e:
                logger.error(f"Error calling LLM: {str(e)}")
                raise
            
            # Clean up markdown if needed
            if "```markdown" in paper_content:
                paper_content = paper_content.split("```markdown")[1].split("```")[0].strip()
            elif "```" in paper_content and paper_content.startswith("```"):
                paper_content = paper_content.split("```")[1].split("```")[0].strip()
            
            return paper_content
        
        except Exception as e:
            logger.error(f"Error generating paper: {str(e)}")
            return self._generate_paper_template(research_context, paper_data)
    
    def _generate_paper_template(
        self,
        research_context: Dict[str, Any],
        paper_data: Dict[str, Any]
    ) -> str:
        """Generate paper summary using template (fallback)."""
        title = research_context.get("title", "Research Paper")
        abstract = research_context.get("abstract", "")
        methods = research_context.get("methods", "")
        full_text = paper_data.get("full_text", "")[:1000]
        
        paper = f"""# {title}

## Abstract

{abstract}

## Introduction

This research paper presents a comprehensive analysis of the topic.

## Methodology

{methods[:500] if methods else "Methodology details extracted from the paper."}

## Key Findings

[Key findings extracted from the research paper]

## Discussion

[Discussion of findings and implications based on the paper content]

## Conclusion

[Summary and future work directions]

## References

[References would be generated from related paper corpus]
"""
        return paper
    
    async def _generate_analysis(
        self,
        research_context: Dict[str, Any],
        paper_corpus: List[Dict[str, Any]],
        parsed_data: Dict[str, Any],
        novelty_result: Dict[str, Any]
    ) -> str:
        """
        Generate novelty analysis report in markdown format.
        
        Args:
            research_context: Research context
            paper_corpus: Paper corpus
            parsed_data: Parsed data
            novelty_result: Novelty assessment
        
        Returns:
            Markdown formatted analysis report
        """
        # Build analysis report
        analysis_parts = []
        
        # Header
        analysis_parts.append("# Research Landscape Analysis\n")
        analysis_parts.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Novelty Assessment
        analysis_parts.append("## Novelty Assessment\n")
        score = novelty_result.get("score", 0)
        analysis_parts.append(f"**Novelty Score: {score}/10**\n")
        analysis_parts.append(f"*{self._get_novelty_rating(score)}*\n\n")
        
        justification = novelty_result.get("justification", "")
        analysis_parts.append(f"{justification}\n\n")
        
        # Semantic Distance
        semantic_distance = novelty_result.get("semantic_distance", 0)
        analysis_parts.append(f"**Semantic Distance:** {semantic_distance:.3f}\n")
        analysis_parts.append("*Average distance to K-nearest neighbors in embedding space*\n\n")
        
        # Similar Papers
        similar_papers = novelty_result.get("similar_papers", [])
        if similar_papers:
            analysis_parts.append("### Most Similar Papers\n\n")
            for i, paper in enumerate(similar_papers[:5], 1):
                title = paper.get("title", "Unknown")
                year = paper.get("year", "N/A")
                similarity = paper.get("similarity_score", 0)
                analysis_parts.append(f"{i}. **{title}** ({year}) - Similarity: {similarity:.3f}\n")
            analysis_parts.append("\n")
        
        # Knowledge Graph Gaps
        kg_gaps = novelty_result.get("kg_gaps", [])
        if kg_gaps:
            analysis_parts.append("## Knowledge Graph Analysis\n\n")
            analysis_parts.append("### Research Gaps Identified\n\n")
            for gap in kg_gaps:
                gap_type = gap.get("type", "Unknown")
                description = gap.get("description", "")
                analysis_parts.append(f"- **{gap_type.replace('_', ' ').title()}:** {description}\n")
            analysis_parts.append("\n")
        
        # Publication Date Context
        publication_date = research_context.get("publication_date")
        if publication_date:
            current_year = datetime.now().year
            try:
                year = int(publication_date)
                year_diff = current_year - year
                if year_diff <= 2:
                    analysis_parts.append(f"**Publication Context:** Recent publication ({publication_date}) - Novelty assessment considers temporal context.\n\n")
                elif year_diff > 5:
                    analysis_parts.append(f"**Publication Context:** Historical publication ({publication_date}) - Novelty assessed in historical context.\n\n")
                else:
                    analysis_parts.append(f"**Publication Context:** Published in {publication_date}.\n\n")
            except:
                pass
        
        # Research Hypothesis
        analysis_parts.append("## Research Hypothesis\n\n")
        hypothesis = self._generate_hypothesis(research_context, kg_gaps, parsed_data)
        analysis_parts.append(f"{hypothesis}\n\n")
        
        # Research Landscape
        analysis_parts.append("## Research Landscape\n\n")
        analysis_parts.append(f"**Papers Analyzed:** {len(paper_corpus)}\n")
        
        kg_stats = parsed_data.get("kg_data", {}).get("stats", {})
        if kg_stats:
            analysis_parts.append(f"**Knowledge Graph Nodes:** {kg_stats.get('num_nodes', 0)}\n")
            analysis_parts.append(f"**Knowledge Graph Edges:** {kg_stats.get('num_edges', 0)}\n")
            analysis_parts.append(f"**Concepts Identified:** {kg_stats.get('num_concepts', 0)}\n")
            analysis_parts.append(f"**Methods Identified:** {kg_stats.get('num_methods', 0)}\n\n")
        
        # Recommendations
        analysis_parts.append("## Recommendations\n\n")
        recommendations = self._generate_recommendations(novelty_result, research_context)
        for rec in recommendations:
            analysis_parts.append(f"- {rec}\n")
        analysis_parts.append("\n")
        
        return "".join(analysis_parts)
    
    def _get_novelty_rating(self, score: float) -> str:
        """Get novelty rating text."""
        if score >= 8:
            return "Highly Novel - Significant contribution with high uniqueness"
        elif score >= 6:
            return "Moderately Novel - Good contribution with some unique aspects"
        elif score >= 4:
            return "Somewhat Novel - Incremental contribution building on existing work"
        else:
            return "Low Novelty - Closely related to existing research"
    
    def _generate_hypothesis(
        self,
        research_context: Dict[str, Any],
        kg_gaps: List[Dict[str, Any]],
        parsed_data: Dict[str, Any]
    ) -> str:
        """Generate research hypothesis from gaps."""
        if kg_gaps:
            gap = kg_gaps[0]
            gap_type = gap.get("type", "")
            if gap_type == "novel_combinations":
                combinations = gap.get("combinations", [])
                if combinations:
                    method, concept = combinations[0]
                    return f"**Hypothesis:** The application of {method} to {concept} represents a novel research direction with potential for significant impact. This combination addresses a gap in the current literature and warrants further investigation."
        
        # Default hypothesis
        title = research_context.get("title", "this research")
        return f"**Hypothesis:** {title} addresses an important research question with potential applications across multiple domains. Further validation and extension of this work could lead to significant scientific and practical contributions."
    
    def _generate_recommendations(
        self,
        novelty_result: Dict[str, Any],
        research_context: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for future work based on novelty assessment."""
        recommendations = []
        
        score = novelty_result.get("score", 5)
        if score >= 8:
            recommendations.append("Consider submitting to top-tier conferences or journals given the high novelty score.")
            recommendations.append("The high novelty suggests this work could have significant impact in the field.")
        elif score >= 6:
            recommendations.append("Expand the research to address identified knowledge gaps for increased impact.")
            recommendations.append("Consider building upon the unique aspects identified in the novelty analysis.")
        else:
            recommendations.append("Continue refining the research approach and expanding the literature review.")
            recommendations.append("Focus on differentiating aspects to increase novelty score.")
        
        # Date-based recommendations
        publication_date = research_context.get("publication_date")
        if publication_date:
            try:
                year = int(publication_date)
                year_diff = datetime.now().year - year
                if year_diff <= 2:
                    recommendations.append("As a recent publication, monitor citations and follow-up research to validate novelty claims.")
                elif year_diff > 5:
                    recommendations.append("For historical context, consider how this work influenced subsequent research in the field.")
            except:
                pass
        
        semantic_distance = novelty_result.get("semantic_distance", 0)
        if semantic_distance < 0.1:
            recommendations.append("The low semantic distance suggests exploring more diverse research directions.")
        
        if not recommendations:
            recommendations.append("Continue refining the research approach and expanding the literature review.")
            recommendations.append("Consider interdisciplinary collaborations to enhance impact.")
        
        return recommendations
    
    def _enhance_knowledge_graph(
        self,
        research_context: Dict[str, Any],
        paper_corpus: List[Dict[str, Any]],
        kg_data: Dict[str, Any],
        novelty_result: Dict[str, Any],
        parsed_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enhance knowledge graph with paper relationships and similarity links.
        
        Args:
            research_context: Target paper context
            paper_corpus: Retrieved papers
            kg_data: Existing knowledge graph data
            novelty_result: Novelty assessment results
        
        Returns:
            Enhanced knowledge graph in D3.js format
        """
        # Get existing nodes and edges
        nodes = kg_data.get("nodes", {})
        edges = kg_data.get("edges", [])
        
        # Add target paper as a special node
        target_paper_id = "target_paper"
        nodes[target_paper_id] = {
            "id": target_paper_id,
            "type": "TargetPaper",
            "title": research_context.get("title", "Target Paper"),
            "abstract": research_context.get("abstract", "")[:200],
            "year": research_context.get("publication_date"),
            "novelty_score": novelty_result.get("score", 0),
            "label": research_context.get("title", "Target Paper")[:50],
            "group": 0  # Special group for target paper
        }
        
        # Convert nodes to list format for D3.js
        d3_nodes = []
        node_id_map = {}  # Map old IDs to new sequential IDs
        
        # Add target paper first
        d3_nodes.append({
            "id": 0,
            "name": nodes[target_paper_id]["title"][:50],
            "type": "TargetPaper",
            "year": nodes[target_paper_id].get("year"),
            "novelty_score": nodes[target_paper_id].get("novelty_score", 0),
            "group": 0,
            "title": nodes[target_paper_id]["title"],
            "abstract": nodes[target_paper_id].get("abstract", "")[:200]
        })
        node_id_map[target_paper_id] = 0
        
        # Add paper nodes
        paper_index = 1
        for node_id, node_data in nodes.items():
            if node_id == target_paper_id:
                continue
            
            if node_data.get("type") == "Paper":
                d3_nodes.append({
                    "id": paper_index,
                    "name": node_data.get("title", "")[:50],
                    "type": "Paper",
                    "year": node_data.get("year"),
                    "group": 1,  # Retrieved papers
                    "title": node_data.get("title", ""),
                    "abstract": node_data.get("abstract", "")[:200]
                })
                node_id_map[node_id] = paper_index
                paper_index += 1
        
        # Add concept nodes
        concept_index = paper_index
        for node_id, node_data in nodes.items():
            if node_data.get("type") == "Concept":
                d3_nodes.append({
                    "id": concept_index,
                    "name": node_data.get("name", ""),
                    "type": "Concept",
                    "group": 2,  # Concepts
                    "title": node_data.get("name", "")
                })
                node_id_map[node_id] = concept_index
                concept_index += 1
        
        # Add method nodes
        method_index = concept_index
        for node_id, node_data in nodes.items():
            if node_data.get("type") == "Method":
                d3_nodes.append({
                    "id": method_index,
                    "name": node_data.get("name", ""),
                    "type": "Method",
                    "group": 3,  # Methods
                    "title": node_data.get("name", "")
                })
                node_id_map[node_id] = method_index
                method_index += 1
        
        # Convert edges to D3.js format
        d3_links = []
        for edge in edges:
            source_id = edge.get("source")
            target_id = edge.get("target")
            
            if source_id in node_id_map and target_id in node_id_map:
                d3_links.append({
                    "source": node_id_map[source_id],
                    "target": node_id_map[target_id],
                    "type": edge.get("relationship", "RELATED"),
                    "value": edge.get("properties", {}).get("frequency", 1),
                    "similarity": edge.get("properties", {}).get("similarity")
                })
        
        # Add similarity links between target paper and similar papers
        similar_papers = novelty_result.get("similar_papers", [])
        for i, similar_paper in enumerate(similar_papers[:10]):  # Top 10 similar papers
            # Find matching paper node by title (more flexible matching)
            similar_title = similar_paper.get("title", "").lower()
            for node in d3_nodes:
                if node.get("type") == "Paper":
                    node_title = node.get("title", "").lower()
                    # More flexible matching: check if titles share significant words
                    if similar_title and node_title:
                        # Check if at least 3 words match or if one title contains the other
                        similar_words = set(similar_title.split()[:5])  # First 5 words
                        node_words = set(node_title.split()[:5])
                        shared_words = similar_words & node_words
                        
                        # Match if: significant word overlap OR one title contains the other
                        if len(shared_words) >= 2 or similar_title[:30] in node_title or node_title[:30] in similar_title:
                            similarity = similar_paper.get("similarity_score", 0)
                            # Avoid duplicate links
                            link_exists = any(
                                link.get("source") == 0 and link.get("target") == node["id"]
                                for link in d3_links
                            )
                            if not link_exists:
                                d3_links.append({
                                    "source": 0,  # Target paper
                                    "target": node["id"],
                                    "type": "SIMILAR_TO",
                                    "value": similarity if similarity > 0 else 0.5,
                                    "similarity": similarity if similarity > 0 else 0.5
                                })
                                break
        
        # Also add links based on shared concepts/methods between target and retrieved papers
        target_concepts = set(research_context.get("keywords", []))
        target_methods = set(research_context.get("methods", "").split() if isinstance(research_context.get("methods"), str) else [])
        
        # Get target paper's extracted concepts from parsed_data
        parsed_papers = parsed_data.get("papers", [])
        if parsed_papers:
            target_parsed = parsed_papers[0]  # First paper is target
            target_concepts.update(target_parsed.get("concepts", []))
            target_methods.update(target_parsed.get("methods", []))
        
        # Link target to retrieved papers based on shared concepts/methods
        for node in d3_nodes:
            if node.get("type") == "Paper" and node["id"] != 0:  # Skip target paper itself
                # Find corresponding paper in corpus
                for paper in paper_corpus:
                    paper_title = paper.get("title", "").lower()
                    node_title = node.get("title", "").lower()
                    
                    # Match paper by title
                    if paper_title and node_title and (paper_title[:30] in node_title or node_title[:30] in paper_title):
                        paper_concepts = set(paper.get("concepts", []))
                        paper_methods = set(paper.get("methods", []))
                        
                        # Calculate shared concepts/methods
                        shared_concepts = target_concepts & paper_concepts
                        shared_methods = target_methods & paper_methods
                        
                        # Create link if there's significant overlap
                        if shared_concepts or shared_methods:
                            similarity_score = len(shared_concepts) + len(shared_methods) * 2
                            
                            # Avoid duplicate links
                            link_exists = any(
                                link.get("source") == 0 and link.get("target") == node["id"]
                                for link in d3_links
                            )
                            
                            if not link_exists and similarity_score > 0:
                                d3_links.append({
                                    "source": 0,  # Target paper
                                    "target": node["id"],
                                    "type": "RELATED_TO",
                                    "value": similarity_score,
                                    "similarity": similarity_score,
                                    "shared_concepts": list(shared_concepts),
                                    "shared_methods": list(shared_methods)
                                })
                        break
        
        # Add links from target paper to its concepts/methods
        target_concepts = research_context.get("keywords", [])
        target_methods = research_context.get("methods", "")
        
        # Link target to concepts
        for concept_name in target_concepts[:10]:
            for node in d3_nodes:
                if node.get("type") == "Concept" and concept_name.lower() in node.get("name", "").lower():
                    d3_links.append({
                        "source": 0,
                        "target": node["id"],
                        "type": "DISCUSSES",
                        "value": 1
                    })
                    break
        
        return {
            "nodes": d3_nodes,
            "links": d3_links,
            "stats": {
                "total_nodes": len(d3_nodes),
                "total_links": len(d3_links),
                "target_paper": research_context.get("title", "Target Paper"),
                "retrieved_papers": len(paper_corpus),
                "novelty_score": novelty_result.get("score", 0)
            }
        }
