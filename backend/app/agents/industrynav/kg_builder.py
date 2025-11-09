"""
Knowledge Graph Builder for constructing and managing research knowledge graphs.
"""
import logging
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict

from app.core.config import settings

logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """Builds and manages knowledge graphs from research papers."""
    
    def __init__(self):
        """Initialize KG builder."""
        self.neo4j_driver = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize graph database client."""
        try:
            if settings.GRAPH_DB_PROVIDER == "neo4j":
                self._init_neo4j()
            else:
                logger.warning(f"Unknown graph DB provider: {settings.GRAPH_DB_PROVIDER}, using in-memory graph")
                self._init_memory()
        except Exception as e:
            logger.error(f"Error initializing graph DB: {str(e)}, using in-memory graph")
            self._init_memory()
    
    def _init_neo4j(self):
        """Initialize Neo4j driver."""
        try:
            from neo4j import GraphDatabase
            if settings.NEO4J_URI and settings.NEO4J_USER and settings.NEO4J_PASSWORD:
                self.neo4j_driver = GraphDatabase.driver(
                    settings.NEO4J_URI,
                    auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
                )
                logger.info("Neo4j driver initialized")
            else:
                logger.warning("Neo4j credentials not provided, using in-memory graph")
                self._init_memory()
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j: {str(e)}")
            self._init_memory()
    
    def _init_memory(self):
        """Initialize in-memory graph storage."""
        self.nodes = {}  # node_id -> node_data
        self.edges = []  # [(source, target, relationship, properties)]
        logger.info("Using in-memory graph storage")
    
    async def build_graph(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build knowledge graph from papers.
        
        Args:
            papers: List of papers with concepts and methods extracted
        
        Returns:
            Dictionary with graph structure and statistics
        """
        logger.info(f"Building knowledge graph from {len(papers)} papers")
        
        if self.neo4j_driver:
            return await self._build_neo4j_graph(papers)
        else:
            return await self._build_memory_graph(papers)
    
    async def _build_neo4j_graph(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build graph in Neo4j."""
        # Neo4j implementation would go here
        logger.info("Neo4j graph building not fully implemented, using memory")
        return await self._build_memory_graph(papers)
    
    async def _build_memory_graph(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build graph in memory with paper-to-paper relationships."""
        # Clear existing graph
        self.nodes = {}
        self.edges = []
        
        # Create paper nodes
        paper_nodes = {}
        for i, paper in enumerate(papers):
            paper_id = f"paper_{i}"
            paper_nodes[i] = paper_id
            self.nodes[paper_id] = {
                "type": "Paper",
                "title": paper.get("title", ""),
                "abstract": paper.get("abstract", ""),
                "year": paper.get("year"),
                "paper_index": i,
                "authors": paper.get("authors", []),
                "arxiv_id": paper.get("arxiv_id"),
                "url": paper.get("url")
            }
        
        # Create paper-to-paper similarity links based on shared concepts/methods
        for i, paper1 in enumerate(papers):
            paper1_id = paper_nodes[i]
            paper1_concepts = set(paper1.get("concepts", []))
            paper1_methods = set(paper1.get("methods", []))
            
            for j, paper2 in enumerate(papers[i+1:], start=i+1):
                paper2_id = paper_nodes[j]
                paper2_concepts = set(paper2.get("concepts", []))
                paper2_methods = set(paper2.get("methods", []))
                
                # Calculate similarity based on shared concepts and methods
                shared_concepts = paper1_concepts & paper2_concepts
                shared_methods = paper1_methods & paper2_methods
                
                similarity_score = len(shared_concepts) + len(shared_methods) * 2  # Methods weighted more
                
                if similarity_score > 0:
                    self.edges.append({
                        "source": paper1_id,
                        "target": paper2_id,
                        "relationship": "RELATED_TO",
                        "properties": {
                            "similarity": similarity_score,
                            "shared_concepts": list(shared_concepts),
                            "shared_methods": list(shared_methods)
                        }
                    })
        
        # Create concept nodes and connect papers to concepts
        concept_nodes = {}
        for i, paper in enumerate(papers):
            paper_id = paper_nodes[i]
            concepts = paper.get("concepts", [])
            methods = paper.get("methods", [])
            
            # Add concept nodes
            for concept in concepts:
                if concept not in concept_nodes:
                    concept_id = f"concept_{concept.lower().replace(' ', '_')}"
                    concept_nodes[concept] = concept_id
                    self.nodes[concept_id] = {
                        "type": "Concept",
                        "name": concept
                    }
                
                # Connect paper to concept
                self.edges.append({
                    "source": paper_id,
                    "target": concept_nodes[concept],
                    "relationship": "DISCUSSES",
                    "properties": {}
                })
            
            # Add method nodes
            for method in methods:
                method_id = f"method_{method.lower().replace(' ', '_')}"
                if method_id not in self.nodes:
                    self.nodes[method_id] = {
                        "type": "Method",
                        "name": method
                    }
                
                # Connect paper to method
                self.edges.append({
                    "source": paper_id,
                    "target": method_id,
                    "relationship": "USES",
                    "properties": {}
                })
        
        # Find concept relationships (concepts used together)
        concept_cooccurrence = defaultdict(int)
        for paper in papers:
            concepts = paper.get("concepts", [])
            for i, concept1 in enumerate(concepts):
                for concept2 in concepts[i+1:]:
                    pair = tuple(sorted([concept1, concept2]))
                    concept_cooccurrence[pair] += 1
        
        # Create edges between frequently co-occurring concepts
        for (concept1, concept2), count in concept_cooccurrence.items():
            if count >= 2:  # Threshold for creating edge
                if concept1 in concept_nodes and concept2 in concept_nodes:
                    self.edges.append({
                        "source": concept_nodes[concept1],
                        "target": concept_nodes[concept2],
                        "relationship": "USED_WITH",
                        "properties": {"frequency": count}
                    })
        
        # Calculate statistics
        stats = {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "num_papers": len(papers),
            "num_concepts": len([n for n in self.nodes.values() if n["type"] == "Concept"]),
            "num_methods": len([n for n in self.nodes.values() if n["type"] == "Method"])
        }
        
        logger.info(f"Built graph with {stats['num_nodes']} nodes and {stats['num_edges']} edges")
        
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "stats": stats
        }
    
    async def find_gaps(self, target_paper_index: int, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Find knowledge graph gaps for the target paper.
        
        Args:
            target_paper_index: Index of target paper
            papers: List of all papers
        
        Returns:
            List of identified gaps
        """
        target_paper = papers[target_paper_index]
        target_concepts = set(target_paper.get("concepts", []))
        target_methods = set(target_paper.get("methods", []))
        
        gaps = []
        
        # Find unique concept combinations
        other_papers_concepts = set()
        other_papers_methods = set()
        
        for i, paper in enumerate(papers):
            if i != target_paper_index:
                other_papers_concepts.update(paper.get("concepts", []))
                other_papers_methods.update(paper.get("methods", []))
        
        # Find concepts used by target but not frequently by others
        unique_concepts = target_concepts - other_papers_concepts
        if unique_concepts:
            gaps.append({
                "type": "unique_concepts",
                "description": f"Target paper uses unique concepts: {', '.join(unique_concepts)}",
                "concepts": list(unique_concepts)
            })
        
        # Find novel method-concept combinations
        target_combinations = set()
        for method in target_methods:
            for concept in target_concepts:
                target_combinations.add((method, concept))
        
        other_combinations = set()
        for i, paper in enumerate(papers):
            if i != target_paper_index:
                for method in paper.get("methods", []):
                    for concept in paper.get("concepts", []):
                        other_combinations.add((method, concept))
        
        novel_combinations = target_combinations - other_combinations
        if novel_combinations:
            gaps.append({
                "type": "novel_combinations",
                "description": f"Target paper has {len(novel_combinations)} novel method-concept combinations",
                "combinations": list(novel_combinations)[:5]  # Limit to top 5
            })
        
        return gaps
