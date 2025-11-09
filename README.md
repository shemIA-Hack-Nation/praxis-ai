# Praxis AI

The Multi-Agent System that Turns Raw Code into Industry-Ready Research.

## 🚀 Overview

Praxis AI is an R&D Acceleration Platform that transforms Jupyter notebooks into comprehensive research papers and industry analysis. The system uses a multi-agent architecture to assess scientific novelty, map research landscapes, and generate actionable insights.

## ✨ Key Features

### Scientific Novelty and Research Landscape Engine
- **Novelty Scoring (1-10)**: Quantifiable assessment of research novelty using semantic analysis and knowledge graph techniques
- **Knowledge Graph Analysis**: Dynamic mapping of research concepts, methods, and relationships
- **Vector Database Integration**: Semantic similarity search across literature corpus
- **Industry Mapping**: Identification of market opportunities and interdisciplinary connections
- **Automated Report Generation**: Dual-output system producing both research papers and comprehensive analysis

### Multi-Agent Architecture
The system employs specialized agents working in concert:
1. **Research Expander**: Builds literature corpus from arXiv/Semantic Scholar
2. **Paper Parser & Embedder**: Extracts concepts and generates embeddings
3. **Novelty Assessor**: Calculates novelty scores and identifies gaps
4. **Industry Mapper**: Maps research to industry applications
5. **Report Generator**: Synthesizes final deliverables

## 🏗️ Architecture

For detailed architecture documentation, see [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md).

## 🛠️ Tech Stack

### Frontend
- React 18 + TypeScript
- Tailwind CSS + shadcn/ui
- TanStack Query
- React Markdown

### Backend
- FastAPI (Python)
- Multi-agent system with LangChain
- Vector Database (Pinecone/Weaviate/Qdrant)
- Graph Database (Neo4j/ArangoDB)
- OpenAI GPT-4 / Anthropic Claude

## 🚀 Getting Started

### Prerequisites
- Node.js >= 18.0.0
- Python 3.9+
- npm >= 8.0.0

### Installation

```bash
# Install all dependencies
npm run install:all

# Start development servers (frontend + backend)
npm run dev
```

The application will be available at:
- Frontend: http://localhost:8080
- Backend API: http://localhost:8000

## 📖 Usage

1. Upload a Jupyter notebook (`.ipynb`) file via the web interface
2. Wait for the multi-agent system to process your notebook
3. View the generated research paper and industry analysis in the dual-tab interface

## 📚 Documentation

- [Architecture Documentation](./docs/ARCHITECTURE.md) - Complete system architecture and design
- [API Documentation](./backend/app/api/v1/) - API endpoint specifications

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## 📄 License

MIT License - see [LICENSE](./LICENSE) for details.
