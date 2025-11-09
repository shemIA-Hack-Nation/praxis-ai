"""
Faiss-based Vector DB for efficient similarity search.
"""
import logging
from typing import List, Dict, Any, Optional
import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    faiss = None

from app.core.config import settings
from app.core.llm_providers import get_embeddings

logger = logging.getLogger(__name__)


class FaissVectorDB:
    """Faiss-based vector database for efficient similarity search."""
    
    def __init__(self, dimension: int = 384):
        """
        Initialize Faiss vector database.
        
        Args:
            dimension: Embedding dimension (default: 384 for all-MiniLM-L6-v2)
        """
        if not HAS_FAISS:
            logger.warning("Faiss not installed. Install with: pip install faiss-cpu")
            self.index = None
            self.dimension = dimension
            return
        
        self.dimension = dimension
        # Use IndexFlatIP for inner product (cosine similarity after normalization)
        # Or IndexFlatL2 for L2 distance
        self.index = faiss.IndexFlatL2(dimension)
        self.paper_metadata: List[Dict[str, Any]] = []
        self.embeddings = get_embeddings()
        logger.info(f"Initialized Faiss vector DB with dimension {dimension}")
    
    async def add_papers(self, papers: List[Dict[str, Any]]) -> bool:
        """
        Add papers to the vector database.
        
        Args:
            papers: List of paper dictionaries with 'abstract' or 'text' field
        
        Returns:
            True if successful
        """
        if not self.index or not self.embeddings:
            logger.error("Faiss index or embeddings not initialized")
            return False
        
        try:
            # Extract texts for embedding
            texts = []
            for paper in papers:
                text = paper.get("abstract", "") or paper.get("text", "") or paper.get("title", "")
                if text:
                    texts.append(text)
                else:
                    texts.append("")  # Empty text for papers without abstract
            
            if not texts:
                logger.warning("No texts to embed")
                return False
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} papers...")
            embeddings = await self.embeddings.aembed_documents(texts)
            
            # Convert to numpy array
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Normalize for cosine similarity (L2 normalization)
            faiss.normalize_L2(embeddings_array)
            
            # Add to index
            self.index.add(embeddings_array)
            
            # Store metadata
            self.paper_metadata.extend(papers)
            
            logger.info(f"Added {len(papers)} papers to Faiss index. Total: {self.index.ntotal}")
            return True
        
        except Exception as e:
            logger.error(f"Error adding papers to Faiss: {str(e)}")
            return False
    
    async def search_similar(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar papers using text query.
        
        Args:
            query_text: Query text to search for
            top_k: Number of results to return
        
        Returns:
            List of similar papers with similarity scores
        """
        if not self.index or not self.embeddings:
            logger.error("Faiss index or embeddings not initialized")
            return []
        
        if self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []
        
        try:
            # Ensure embeddings are initialized
            if hasattr(self.embeddings, '_ensure_initialized'):
                self.embeddings._ensure_initialized()
            
            # Generate query embedding
            query_embedding = await self.embeddings.aembed_query(query_text)
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_vector)
            
            # Search
            k = min(top_k, self.index.ntotal)
            distances, indices = self.index.search(query_vector, k)
            
            # Build results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if 0 <= idx < len(self.paper_metadata):
                    paper = self.paper_metadata[idx].copy()
                    # Convert L2 distance to cosine similarity
                    # After L2 normalization, distance = 2 * (1 - cosine_similarity)
                    # So: cosine_similarity = 1 - (distance / 2)
                    similarity = 1 - (distance / 2.0)
                    similarity = max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
                    
                    paper["similarity_score"] = float(similarity)
                    paper["distance"] = float(distance)
                    paper["rank"] = i + 1
                    results.append(paper)
            
            logger.info(f"Found {len(results)} similar papers for query")
            return results
        
        except Exception as e:
            logger.error(f"Error searching Faiss index: {str(e)}")
            return []
    
    async def search_similar_by_embedding(self, query_embedding: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar papers using embedding vector directly.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
        
        Returns:
            List of similar papers with similarity scores
        """
        if not self.index:
            logger.error("Faiss index not initialized")
            return []
        
        if self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []
        
        try:
            # Convert to numpy array and normalize
            query_vector = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_vector)
            
            # Search
            k = min(top_k, self.index.ntotal)
            distances, indices = self.index.search(query_vector, k)
            
            # Build results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if 0 <= idx < len(self.paper_metadata):
                    paper = self.paper_metadata[idx].copy()
                    similarity = 1 - (distance / 2.0)
                    similarity = max(0.0, min(1.0, similarity))
                    
                    results.append({
                        "paper_id": str(idx),
                        "score": float(similarity),
                        "paper": paper
                    })
            
            return results
        
        except Exception as e:
            logger.error(f"Error searching Faiss by embedding: {str(e)}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return {
            "total_papers": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "has_faiss": HAS_FAISS
        }
    
    def clear(self):
        """Clear the index."""
        if self.index:
            # Reset index
            self.index.reset()
            self.paper_metadata.clear()
            logger.info("Cleared Faiss index")

