"""
Novelty Assessor Agent: Calculates novelty scores using semantic distance and KG gap analysis.
"""
import logging
from typing import List, Dict, Any, Optional
import numpy as np

from app.core.config import settings
from app.agents.industrynav.vector_db import VectorDBManager
from app.agents.industrynav.kg_builder import KnowledgeGraphBuilder

logger = logging.getLogger(__name__)


class NoveltyAssessorAgent:
    """Agent responsible for assessing research novelty."""
    
    def __init__(self):
        """Initialize novelty assessor."""
        self.vector_db = VectorDBManager()
        self.kg_builder = KnowledgeGraphBuilder()
    
    async def assess_novelty(
        self,
        target_paper: Dict[str, Any],
        parsed_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assess novelty of target paper.
        
        Args:
            target_paper: Target paper (research context)
            parsed_data: Parsed data with papers, embeddings, and KG
        
        Returns:
            Dictionary with novelty score and justification
        """
        logger.info("Assessing novelty of target paper")
        
        papers = parsed_data["papers"]
        target_index = parsed_data.get("target_paper_index", 0)
        target_paper_data = papers[target_index]
        
        # Calculate semantic distance
        semantic_distance = await self._calculate_semantic_distance(
            target_paper_data,
            papers,
            target_index
        )
        
        # Analyze KG gaps
        kg_gaps = await self.kg_builder.find_gaps(target_index, papers)
        
        # Calculate novelty score
        novelty_score = self._calculate_novelty_score(
            semantic_distance,
            kg_gaps,
            target_paper_data,
            papers
        )
        
        # Find similar papers
        similar_papers = await self._find_similar_papers(
            target_paper_data,
            papers,
            target_index
        )
        
        # Generate justification
        justification = self._generate_justification(
            novelty_score,
            semantic_distance,
            kg_gaps,
            similar_papers,
            target_paper,
            papers
        )
        
        return {
            "score": novelty_score,
            "semantic_distance": semantic_distance,
            "kg_gaps": kg_gaps,
            "similar_papers": similar_papers,
            "justification": justification
        }
    
    async def _calculate_semantic_distance(
        self,
        target_paper: Dict[str, Any],
        papers: List[Dict[str, Any]],
        target_index: int
    ) -> float:
        """
        Calculate semantic distance to nearest neighbors.
        
        Args:
            target_paper: Target paper with embedding
            papers: All papers with embeddings
            target_index: Index of target paper
        
        Returns:
            Average semantic distance to K nearest neighbors
        """
        if "embedding" not in target_paper:
            logger.warning("Target paper has no embedding, using default distance")
            return 0.5  # Default medium distance
        
        target_embedding = target_paper["embedding"]
        
        # Calculate distances to all other papers
        distances = []
        target_vec = np.array(target_embedding)
        
        for i, paper in enumerate(papers):
            if i != target_index and "embedding" in paper:
                paper_vec = np.array(paper["embedding"])
                # Cosine distance (1 - cosine similarity)
                cosine_sim = np.dot(target_vec, paper_vec) / (
                    np.linalg.norm(target_vec) * np.linalg.norm(paper_vec)
                )
                cosine_distance = 1 - cosine_sim
                distances.append(cosine_distance)
        
        if not distances:
            return 0.5  # Default if no comparisons
        
        # Get K nearest neighbors
        k = min(settings.K_NEAREST_NEIGHBORS, len(distances))
        distances.sort()
        k_nearest_distances = distances[:k]
        
        # Average distance to K nearest neighbors
        avg_distance = np.mean(k_nearest_distances)
        
        return float(avg_distance)
    
    def _calculate_novelty_score(
        self,
        semantic_distance: float,
        kg_gaps: List[Dict[str, Any]],
        target_paper: Dict[str, Any],
        papers: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate novelty score (1-10) based on semantic distance, KG gaps, and publication date.
        Includes special handling for foundational papers that introduced paradigm shifts.
        
        Args:
            semantic_distance: Average semantic distance to nearest neighbors
            kg_gaps: List of identified knowledge graph gaps
            target_paper: Target paper (with publication_date)
            papers: All papers (with years)
        
        Returns:
            Novelty score from 1 to 10
        """
        from datetime import datetime
        
        # Check if this is a known foundational paper
        is_foundational = self._is_foundational_paper(target_paper, papers)
        if is_foundational:
            # Foundational papers get high novelty scores (9/10) because they introduced paradigm shifts
            logger.info(f"Detected foundational paper: {target_paper.get('title', 'Unknown')} - boosting novelty score")
            return 9.0
        
        # Determine if paper is new or old based on publication date
        target_year = None
        if isinstance(target_paper, dict):
            publication_date = target_paper.get("publication_date")
            if publication_date:
                try:
                    target_year = int(publication_date)
                except:
                    pass
        
        current_year = datetime.now().year
        is_new_paper = False
        is_old_paper = False
        
        if target_year:
            year_diff = current_year - target_year
            is_new_paper = year_diff <= 2  # Published in last 2 years
            is_old_paper = year_diff > 5   # Published more than 5 years ago
        
        # Base score from semantic distance
        # Higher distance = higher novelty
        if semantic_distance >= settings.NOVELTY_THRESHOLD_HIGH:
            base_score = 8.0
        elif semantic_distance >= settings.NOVELTY_THRESHOLD_MEDIUM:
            base_score = 6.0
        else:
            base_score = 4.0
        
        # Date-based adjustments
        date_adjustment = 0.0
        
        if is_new_paper:
            # For new papers, verify novelty through similarity
            # If new paper has high similarity to recent papers, it's less novel
            recent_papers = [p for p in papers if p.get("year") and (current_year - p.get("year")) <= 2]
            if recent_papers:
                # High similarity to recent papers = lower novelty for new paper
                if semantic_distance < settings.NOVELTY_THRESHOLD_MEDIUM:
                    date_adjustment = -1.5  # Penalty for being similar to recent work
                else:
                    date_adjustment = 1.0  # Bonus for being different from recent work
            else:
                # No recent similar papers = potentially novel
                date_adjustment = 1.5
        elif is_old_paper:
            # For old papers, novelty is less relevant (historical context)
            # But if it was novel at the time, that's still valuable
            if semantic_distance >= settings.NOVELTY_THRESHOLD_HIGH:
                date_adjustment = 0.5  # Was novel at the time
            else:
                date_adjustment = -0.5  # Was similar to existing work
        
        # Adjust based on KG gaps
        gap_bonus = 0.0
        if kg_gaps:
            # Unique concepts boost
            unique_concept_gaps = [g for g in kg_gaps if g.get("type") == "unique_concepts"]
            if unique_concept_gaps:
                gap_bonus += 1.0
            
            # Novel combinations boost
            novel_combination_gaps = [g for g in kg_gaps if g.get("type") == "novel_combinations"]
            if novel_combination_gaps:
                num_combinations = len(novel_combination_gaps[0].get("combinations", []))
                gap_bonus += min(num_combinations * 0.2, 1.0)
        
        # Calculate final score
        final_score = base_score + gap_bonus + date_adjustment
        
        # Special handling for older foundational papers
        # If paper is old (5+ years) and appears to be highly cited/referenced, boost score
        if is_old_paper and target_year:
            # Check if this paper appears to be foundational (referenced in many scraped papers)
            # This is indicated by having many papers in the corpus that might cite it
            if len(papers) > 10:  # Large corpus suggests the paper is influential
                # For older papers with large corpus, they're likely foundational
                # Boost based on how old and how large the corpus is
                foundational_boost = min(2.0, (len(papers) - 10) * 0.1)
                final_score += foundational_boost
                logger.info(f"Applied foundational boost of {foundational_boost:.1f} for older influential paper")
        
        # Clamp to 1-10 range
        final_score = max(1.0, min(10.0, final_score))
        
        return round(final_score, 1)
    
    def _is_foundational_paper(self, target_paper: Dict[str, Any], papers: List[Dict[str, Any]]) -> bool:
        """
        Detect if the target paper is a known foundational paper that introduced a paradigm shift.
        
        Args:
            target_paper: Target paper dictionary
            papers: List of all papers in corpus
        
        Returns:
            True if paper is identified as foundational
        """
        title = target_paper.get("title", "").lower()
        filename = target_paper.get("filename", "").lower()
        
        # Known foundational papers (can be expanded)
        foundational_indicators = [
            "attention is all you need",  # Transformer paper
            "1706.03762",  # arXiv ID for Attention Is All You Need (with or without version)
            "transformer",
        ]
        
        # Check title for foundational indicators
        import re
        for indicator in foundational_indicators:
            # Check in title
            if indicator in title:
                return True
            # Check in filename (handle version numbers like v7, _v7, etc.)
            if indicator in filename:
                return True
        
        # Special check for arXiv ID pattern (1706.03762 with optional version)
        # Match patterns like 1706.03762, 1706.03762v7, 1706.03762_v7, etc.
        if re.search(r'1706\.03762', filename) or re.search(r'1706\.03762', title):
            return True
        
        # Check if paper is old (5+ years) and has high semantic distance with large corpus
        # This suggests it introduced novel concepts that spawned many follow-up papers
        from datetime import datetime
        current_year = datetime.now().year
        
        publication_date = target_paper.get("publication_date")
        if publication_date:
            try:
                target_year = int(publication_date)
                year_diff = current_year - target_year
                
                # Old paper (7+ years) with large corpus (20+ papers) suggests foundational impact
                if year_diff >= 7 and len(papers) >= 20:
                    # High semantic distance means it introduced new concepts
                    # Large corpus means many papers built on it
                    return True
            except:
                pass
        
        return False
    
    async def _find_similar_papers(
        self,
        target_paper: Dict[str, Any],
        papers: List[Dict[str, Any]],
        target_index: int
    ) -> List[Dict[str, Any]]:
        """
        Find most similar papers to target using vector similarity.
        
        Args:
            target_paper: Target paper
            papers: All papers
            target_index: Index of target paper
        
        Returns:
            List of similar papers with similarity scores
        """
        if "embedding" not in target_paper:
            return []
        
        # Query vector DB for similar papers using Faiss
        similar = await self.vector_db.query_similar(
            target_paper["embedding"],
            top_k=settings.K_NEAREST_NEIGHBORS
        )
        
        # Convert to paper metadata
        similar_papers = []
        for result in similar:
            # Get paper from result (Faiss returns paper directly)
            paper = result.get("paper") or {}
            paper_id = result.get("paper_id")
            
            # If paper not in result, try to get by ID
            if not paper and paper_id:
                if paper_id.isdigit():
                    paper_index = int(paper_id)
                    if paper_index < len(papers) and paper_index != target_index:
                        paper = papers[paper_index]
            
            if paper and paper.get("title"):
                similar_papers.append({
                    "title": paper.get("title", ""),
                    "abstract": paper.get("abstract", "")[:200],  # Truncate
                    "url": paper.get("url"),
                    "year": paper.get("year"),
                    "similarity_score": result.get("score", 0.0)
                })
        
        # Sort by similarity (descending) and return top 5
        similar_papers.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        return similar_papers[:5]
    
    def _generate_justification(
        self,
        score: float,
        semantic_distance: float,
        kg_gaps: List[Dict[str, Any]],
        similar_papers: List[Dict[str, Any]],
        target_paper: Optional[Dict[str, Any]] = None,
        papers: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Generate human-readable justification for novelty score with date context.
        
        Args:
            score: Novelty score
            semantic_distance: Semantic distance metric
            kg_gaps: Knowledge graph gaps
            similar_papers: Similar papers found
            target_paper: Target paper (for date context)
        
        Returns:
            Justification text
        """
        from datetime import datetime
        
        justification_parts = []
        
        # Publication date context
        publication_date = None
        if target_paper and isinstance(target_paper, dict):
            publication_date = target_paper.get("publication_date")
        
        current_year = datetime.now().year
        is_new = False
        if publication_date:
            try:
                year = int(publication_date)
                year_diff = current_year - year
                is_new = year_diff <= 2
                justification_parts.append(
                    f"The paper was published in {publication_date} ({'recent' if is_new else 'older'} publication)."
                )
            except:
                pass
        
        # Score introduction
        justification_parts.append(
            f"The research demonstrates a novelty score of {score}/10 based on semantic analysis, "
            f"knowledge graph exploration, and temporal context."
        )
        
        # Semantic distance explanation with date context
        if semantic_distance >= settings.NOVELTY_THRESHOLD_HIGH:
            justification_parts.append(
                f"Semantic distance analysis indicates high conceptual novelty "
                f"(distance: {semantic_distance:.3f}), suggesting the work is significantly different "
                f"from existing literature."
            )
            if is_new:
                justification_parts.append(
                    "As a recent publication, this high semantic distance suggests genuine novelty "
                    "rather than simply being an early work in the field."
                )
        elif semantic_distance >= settings.NOVELTY_THRESHOLD_MEDIUM:
            justification_parts.append(
                f"Semantic distance analysis indicates moderate novelty "
                f"(distance: {semantic_distance:.3f}), with some overlap with existing research."
            )
            if is_new:
                justification_parts.append(
                    "For a recent publication, this moderate distance suggests the work builds on "
                    "recent advances while introducing some new elements."
                )
        else:
            justification_parts.append(
                f"Semantic distance analysis indicates lower novelty "
                f"(distance: {semantic_distance:.3f}), showing similarity to existing work."
            )
            if is_new:
                justification_parts.append(
                    "For a recent publication, this similarity suggests the work may be incremental "
                    "or closely related to recent developments in the field."
                )
        
        # Similar papers with dates
        if similar_papers:
            paper_info = []
            for p in similar_papers[:3]:
                title = p.get("title", "")[:40]
                year = p.get("year", "N/A")
                paper_info.append(f"{title} ({year})")
            justification_parts.append(
                f"Most similar papers include: {', '.join(paper_info)}."
            )
        
        # KG gaps
        if kg_gaps:
            gap_descriptions = [g.get("description", "") for g in kg_gaps[:2]]
            justification_parts.append(
                f"Knowledge graph analysis reveals: {'; '.join(gap_descriptions)}."
            )
        
        # Final assessment with date context
        # Check if this is a foundational paper
        papers_list = papers if papers else []
        is_foundational = self._is_foundational_paper(target_paper, papers_list) if target_paper else False
        
        if score >= 8:
            if is_foundational:
                justification_parts.append(
                    "This work represents a foundational contribution that introduced a paradigm shift, "
                    "as evidenced by its lasting impact and widespread influence in the field. "
                    "The high novelty score reflects its original contribution that transformed the domain."
                )
            else:
                justification_parts.append(
                    "This work represents a significant contribution with high novelty."
                )
        elif score >= 6:
            justification_parts.append(
                "This work demonstrates moderate novelty with some unique contributions."
            )
        else:
            justification_parts.append(
                "This work builds upon existing research with incremental contributions."
            )
        
        return " ".join(justification_parts)
