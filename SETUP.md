# Setup Guide for Praxis AI

This guide will help you set up and run the Praxis AI system.

## Prerequisites

- Python 3.9 or higher
- Node.js 18.0 or higher
- npm 8.0 or higher
- (Optional) OpenAI API key for LLM features
- (Optional) Pinecone/Neo4j accounts for vector and graph databases

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd praxis-ai
```

### 2. Install Backend Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
# Required for LLM features
OPENAI_API_KEY=your_openai_api_key_here

# Optional: For vector database
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment

# Optional: For graph database
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

# Optional: For Semantic Scholar API
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_api_key
```

### 5. Create Required Directories

```bash
mkdir -p data/uploads
mkdir -p data/reports
```

## Running the Application

### Development Mode

Run both frontend and backend simultaneously:

```bash
npm run dev
```

This will start:
- Backend API server on http://localhost:8000
- Frontend development server on http://localhost:8080

### Running Separately

#### Backend Only

```bash
cd backend
python -m uvicorn app.main:app --reload --port 8000
```

#### Frontend Only

```bash
cd frontend
npm run dev
```

## Usage

1. Open your browser and navigate to http://localhost:8080
2. Upload a Jupyter notebook (`.ipynb` file)
3. Wait for the analysis to complete (this may take several minutes)
4. View the generated research paper and industry analysis in the dual-tab interface

## Configuration

### Vector Database

The system supports multiple vector database providers:
- **Pinecone** (recommended for production)
- **Weaviate**
- **Qdrant**
- **In-memory** (default, for testing)

Set `VECTOR_DB_PROVIDER` in `.env` to change the provider.

### Graph Database

The system supports:
- **Neo4j** (recommended for production)
- **In-memory** (default, for testing)

Set `GRAPH_DB_PROVIDER` in `.env` to change the provider.

### LLM Provider

The system supports:
- **OpenAI** (default)
- **Anthropic**

Set `LLM_PROVIDER` and corresponding API keys in `.env`.

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed correctly
   ```bash
   pip install -r requirements.txt
   ```

2. **API Key Errors**: Ensure your API keys are set correctly in `.env`

3. **Port Already in Use**: Change the port in `vite.config.ts` (frontend) or `main.py` (backend)

4. **CORS Errors**: Check that `CORS_ORIGINS` in `.env` includes your frontend URL

5. **Timeout Errors**: Increase the timeout in `frontend/src/pages/Index.tsx` if analysis takes longer

### Logging

Check the console output for detailed error messages. Set `LOG_LEVEL=DEBUG` in `.env` for more verbose logging.

## Testing

Run tests with:

```bash
cd backend
pytest tests/
```

## Production Deployment

For production deployment:

1. Set `LOG_LEVEL=INFO` or `WARNING`
2. Use production-grade vector and graph databases (Pinecone, Neo4j)
3. Configure proper CORS origins
4. Set up proper error monitoring and logging
5. Use a production-grade ASGI server (e.g., Gunicorn with Uvicorn workers)

## Support

For issues or questions, please open an issue on the GitHub repository.
