"""
Vector Database Manager for storing and querying paper embeddings.
Uses Faiss for efficient similarity search.
"""
import logging
from typing import List, Dict, Any, Optional
import numpy as np

from app.core.config import settings
from app.agents.industrynav.faiss_vector_db import FaissVectorDB

logger = logging.getLogger(__name__)


class VectorDBManager:
    """Manages vector database operations. Uses Faiss by default for efficiency."""
    
    def __init__(self):
        """Initialize vector DB manager."""
        # Use Faiss by default for efficient local search
        self.faiss_db = FaissVectorDB(dimension=384)  # all-MiniLM-L6-v2 dimension
        self.client = None
        self.index = None
        self._initialize_external_clients()
    
    def _initialize_external_clients(self):
        """Initialize external vector database clients if configured."""
        # Faiss is always available as primary
        # External DBs are optional for production scaling
        try:
            if settings.VECTOR_DB_PROVIDER == "pinecone":
                self._init_pinecone()
            elif settings.VECTOR_DB_PROVIDER == "weaviate":
                self._init_weaviate()
            elif settings.VECTOR_DB_PROVIDER == "qdrant":
                self._init_qdrant()
        except Exception as e:
            logger.warning(f"Could not initialize external vector DB: {str(e)}, using Faiss only")
    
    def _init_pinecone(self):
        """Initialize Pinecone client."""
        try:
            import pinecone
            pinecone.init(
                api_key=settings.PINECONE_API_KEY,
                environment=settings.PINECONE_ENVIRONMENT
            )
            self.index = pinecone.Index(settings.PINECONE_INDEX_NAME)
            logger.info("Pinecone client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {str(e)}")
            self._init_memory()
    
    def _init_weaviate(self):
        """Initialize Weaviate client."""
        try:
            import weaviate
            self.client = weaviate.Client("http://localhost:8080")
            logger.info("Weaviate client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate: {str(e)}")
            self._init_memory()
    
    def _init_qdrant(self):
        """Initialize Qdrant client."""
        try:
            from qdrant_client import QdrantClient
            self.client = QdrantClient(host="localhost", port=6333)
            logger.info("Qdrant client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {str(e)}")
            self._init_memory()
    
    def _init_memory(self):
        """Initialize in-memory storage (fallback)."""
        self.storage = {}
        logger.info("Using in-memory vector storage")
    
    async def store_papers(self, papers: List[Dict[str, Any]]):
        """
        Store paper embeddings in vector database.
        Uses Faiss for efficient local storage.
        
        Args:
            papers: List of papers with embeddings
        """
        if not papers:
            return
        
        try:
            # Always use Faiss for efficient local search
            await self.faiss_db.add_papers(papers)
            logger.info(f"Stored {len(papers)} papers in Faiss vector DB")
            
            # Optionally sync to external DB if configured
            if settings.VECTOR_DB_PROVIDER == "pinecone" and self.index:
                await self._store_pinecone(papers)
            elif settings.VECTOR_DB_PROVIDER == "weaviate" and self.client:
                await self._store_weaviate(papers)
            elif settings.VECTOR_DB_PROVIDER == "qdrant" and self.client:
                await self._store_qdrant(papers)
        except Exception as e:
            logger.error(f"Error storing papers: {str(e)}")
    
    async def _store_pinecone(self, papers: List[Dict[str, Any]]):
        """Store papers in Pinecone."""
        vectors = []
        for i, paper in enumerate(papers):
            if "embedding" in paper:
                vectors.append({
                    "id": str(i),
                    "values": paper["embedding"],
                    "metadata": {
                        "title": paper.get("title", ""),
                        "abstract": paper.get("abstract", "")[:1000]  # Limit metadata size
                    }
                })
        
        if vectors:
            self.index.upsert(vectors=vectors)
            logger.info(f"Stored {len(vectors)} papers in Pinecone")
    
    async def _store_weaviate(self, papers: List[Dict[str, Any]]):
        """Store papers in Weaviate."""
        # Weaviate implementation would go here
        logger.info("Weaviate storage not fully implemented, using memory")
        await self._store_memory(papers)
    
    async def _store_qdrant(self, papers: List[Dict[str, Any]]):
        """Store papers in Qdrant."""
        # Qdrant implementation would go here
        logger.info("Qdrant storage not fully implemented, using memory")
        await self._store_memory(papers)
    
    async def _store_memory(self, papers: List[Dict[str, Any]]):
        """Store papers in memory."""
        for i, paper in enumerate(papers):
            if "embedding" in paper:
                self.storage[i] = {
                    "paper": paper,
                    "embedding": paper["embedding"]
                }
        logger.info(f"Stored {len(papers)} papers in memory")
    
    async def query_similar(
        self,
        query_embedding: List[float],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Query similar papers using vector similarity.
        Uses Faiss for efficient search.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
        
        Returns:
            List of similar papers with similarity scores
        """
        try:
            # Use Faiss for efficient search
            # Convert embedding to text for Faiss query (it needs text to generate embedding)
            # Actually, we need to pass the text, not the embedding
            # For now, use direct embedding search in memory as fallback
            return await self._query_faiss_direct(query_embedding, top_k)
        except Exception as e:
            logger.error(f"Error querying vector DB: {str(e)}")
            return await self._query_memory(query_embedding, top_k)
    
    async def _query_faiss_direct(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Query Faiss directly with embedding vector."""
        try:
            # Use FaissVectorDB's search_by_embedding method
            return await self.faiss_db.search_similar_by_embedding(query_embedding, top_k)
        except Exception as e:
            logger.error(f"Error in Faiss direct query: {str(e)}")
            return []
    
    async def _query_pinecone(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Query Pinecone for similar papers."""
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        similar_papers = []
        for match in results.matches:
            similar_papers.append({
                "paper_id": match.id,
                "score": match.score,
                "metadata": match.metadata
            })
        
        return similar_papers
    
    async def _query_memory(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Query in-memory storage for similar papers."""
        if not self.storage:
            return []
        
        # Calculate cosine similarity
        similarities = []
        query_vec = np.array(query_embedding)
        
        for paper_id, data in self.storage.items():
            paper_vec = np.array(data["embedding"])
            # Cosine similarity
            similarity = np.dot(query_vec, paper_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(paper_vec))
            similarities.append({
                "paper_id": paper_id,
                "score": float(similarity),
                "paper": data["paper"]
            })
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x["score"], reverse=True)
        return similarities[:top_k]
