# Optimization Guide: Efficient Paper Scraping and Novelty Assessment

## Overview

This guide explains the optimized pipeline for efficient research paper scraping and novelty assessment with real-time visualization.

## Key Optimizations

### 1. LLM-Based Search Term Extraction

**Location:** `backend/app/agents/industrynav/search_term_extractor.py`

**Features:**
- Uses Hugging Face LLM (Llama-3-70B) to extract structured search terms
- Extracts:
  - **Keywords**: 5-10 core keywords/phrases
  - **Methods**: Specific methods/models/architectures
  - **Categories**: arXiv categories (cs.AI, cs.CV, etc.)
  - **Authors**: Author list for targeted searching

**Benefits:**
- Highly focused search queries
- Reduces irrelevant results
- Limits scraping to 20-50 most relevant papers

### 2. Optimized arXiv Search

**Location:** `backend/app/agents/industrynav/agent_research_expander.py`

**Optimizations:**
- **Max Results**: Hard limit of 50-100 papers (configurable)
- **Categories**: Filters by arXiv categories (e.g., `cat:cs.AI`)
- **Date Range**: Limits to ±3 years around target paper publication date
- **Focused Query**: Uses AND/OR logic for methods and keywords
  - Example: `("transformer" AND "object detection") OR "gcn-based"`

**Query Construction:**
```python
# Methods: AND logic (both must be present)
methods_query = "transformer" AND "attention mechanism"

# Keywords: OR logic (any can be present)
keywords_query = "machine learning" OR "deep learning" OR "neural networks"

# Combined: (methods) OR (keywords)
final_query = (methods_query) OR (keywords_query)
```

### 3. Efficient Vector-Based Pipeline

**Components:**
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2) - 384 dimensions
- **Vector DB**: Faiss (IndexFlatL2) for fast similarity search
- **Similarity**: Cosine similarity via L2-normalized vectors

**Performance:**
- Embedding generation: ~100ms per paper (CPU)
- Vector search: O(log n) with Faiss, instant for 50 papers
- No GPU required for embeddings

### 4. Real-Time Visualization

**WebSocket Endpoint:** `/api/v1/ws/scraping/{client_id}`

**Events:**
- `extraction_started`: LLM extraction begins
- `extraction_complete`: Search terms extracted
- `search_started`: arXiv search begins
- `paper_found`: Each paper found (real-time)
- `search_complete`: Search finished
- `ranking_complete`: Papers ranked
- `processing_complete`: All processing done

**Frontend Component:** `frontend/src/components/ScrapingVisualization.tsx`

**Features:**
- Real-time paper list updates
- Progress bar
- Search terms display
- Paper metadata (title, year, authors, categories)

## Pipeline Flow

### Step 1: Content Extraction
1. Upload PDF
2. Extract structured content:
   - Title, Abstract, Methods, Results
   - Publication date, Authors, Keywords

### Step 2: LLM-Based Term Extraction
1. Use LLM to extract:
   - Core keywords (5-10)
   - Specific methods/models
   - arXiv categories
2. Build optimized query

### Step 3: Optimized Paper Search
1. Search arXiv with:
   - Focused query (methods AND keywords)
   - Category filters
   - Date range (±3 years)
   - Max results: 50
2. Real-time visualization of papers found

### Step 4: Embedding & Indexing
1. Generate embeddings for all papers (Sentence Transformers)
2. Store in Faiss vector DB
3. Build knowledge graph

### Step 5: Novelty Assessment
1. Calculate semantic distance (K-NN search in Faiss)
2. Analyze date context (new vs old paper)
3. Calculate novelty score (1-10)
4. Generate justification

### Step 6: Report Generation
1. Generate research paper summary
2. Generate novelty analysis report
3. Include temporal context and recommendations

## Configuration

### Search Parameters

**File:** `backend/app/core/config.py`

```python
MAX_PAPERS_IN_CORPUS: int = 50  # Maximum papers to scrape
MIN_PAPERS_IN_CORPUS: int = 20  # Minimum papers required
K_NEAREST_NEIGHBORS: int = 10   # K for similarity search
```

### LLM Configuration

```python
LLM_PROVIDER: str = "huggingface"
LLM_MODEL: str = "meta-llama/Meta-Llama-3-70B"
HF_TOKEN: str = "your_token_here"
```

### Embedding Configuration

```python
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
```

## Usage

### Backend

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables:**
   ```bash
   export HF_TOKEN="your_huggingface_token"
   ```

3. **Run backend:**
   ```bash
   cd backend
   python -m uvicorn app.main:app --reload --port 8000
   ```

### Frontend

1. **Install dependencies:**
   ```bash
   cd frontend
   npm install
   ```

2. **Run frontend:**
   ```bash
   npm run dev
   ```

3. **Upload paper:**
   - Connect to WebSocket automatically
   - Upload PDF file
   - Watch real-time progress
   - View papers as they're found

## Real-Time Visualization

The visualization component shows:
1. **Progress Bar**: Overall progress (0-100%)
2. **Current Phase**: Current processing step
3. **Search Terms**: Extracted keywords, methods, categories
4. **Papers Found**: Real-time list of papers with:
   - Title
   - Year
   - Authors
   - Categories
   - Status (found/processing)

## Performance Metrics

### Expected Performance

- **Search Term Extraction**: 2-5 seconds (LLM call)
- **arXiv Search**: 5-10 seconds (for 50 papers)
- **Embedding Generation**: 5-10 seconds (50 papers, CPU)
- **Vector Search**: <100ms (instant)
- **Novelty Assessment**: 1-2 seconds
- **Total Time**: 15-30 seconds for full pipeline

### Hardware Requirements

- **CPU**: Any modern CPU (no GPU required)
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: ~500MB for models and dependencies

## Troubleshooting

### Issue: WebSocket not connecting

**Solution:**
- Check that backend is running on port 8000
- Verify CORS origins in `config.py`
- Check browser console for WebSocket errors

### Issue: No papers found

**Solution:**
- Check search terms extraction
- Verify arXiv API is accessible
- Check date range (may be too restrictive)
- Increase `MAX_PAPERS_IN_CORPUS` if needed

### Issue: Slow embedding generation

**Solution:**
- First run downloads model (~90MB)
- Subsequent runs are faster
- Consider using GPU if available (automatic with CUDA)

### Issue: Faiss not working

**Solution:**
- Install: `pip install faiss-cpu`
- Falls back to in-memory search if Faiss unavailable
- Check logs for Faiss initialization errors

## Best Practices

1. **Limit Paper Count**: Keep `MAX_PAPERS_IN_CORPUS` ≤ 50 for fast processing
2. **Use Date Ranges**: Set appropriate date ranges based on paper publication date
3. **Monitor Progress**: Use WebSocket visualization to track progress
4. **Error Handling**: Check logs for extraction/search errors
5. **Cache Embeddings**: Consider caching embeddings for repeated papers

## Future Improvements

1. **Caching**: Cache embeddings for papers to avoid recomputation
2. **Batch Processing**: Process multiple papers in parallel
3. **GPU Support**: Use GPU for faster embedding generation
4. **Advanced Filtering**: Add more sophisticated paper filtering
5. **Graph Visualization**: Visualize knowledge graph structure

