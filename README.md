# PraxisAI 🚀

**The Multi-Agent System that Turns Raw Code into Industry-Ready Research**

PraxisAI is an intelligent research assistant that transforms Jupyter notebooks into comprehensive research papers and identifies real-world industrial applications. Built with a sophisticated multi-agent architecture, it bridges the gap between academic research and commercial innovation.

## 🌟 Overview

PraxisAI operates in two main phases:

### 📊 **Sprint 1: Notebook-to-Paper Generation** *(Current)*
Transform Jupyter notebooks into professional research papers with:
- **Automated Code Analysis**: Extract insights, methodology, and results
- **Multi-Agent Coordination**: Specialized AI agents for different aspects
- **LaTeX & PDF Generation**: Professional academic formatting
- **Web Dashboard**: Real-time progress tracking and downloads

### 🔍 **Sprint 2: Paper-to-Insights Analysis** *(Coming Soon)*
Analyze generated papers for industrial potential with:
- **Novelty Assessment**: Evaluate research originality and contribution
- **Industry Mapping**: Identify commercial applications across sectors
- **Research Context**: Find related work and build research landscape
- **Market Intelligence**: Assess real-world implementation potential

## 🏗️ Architecture

### Current Multi-Agent System

```
📱 Frontend (React + TypeScript)
    ↓
🌐 FastAPI Backend
    ↓
🎯 Orchestrator Agent
    ↓
┌─────────────────────────────────────┐
│  🔬 NotebookAnalyzer Agent          │
│  📝 ContentGenerator Agent          │
│  🖼️  IllustrationGenerator Agent    │
│  📋 FormatterAgent                  │
└─────────────────────────────────────┘
    ↓
📄 LaTeX/PDF Research Paper
```

### Upcoming Industry Navigator System

```
📄 Generated Research Paper
    ↓
🎯 Navigator Orchestrator
    ↓
┌─────────────────────────────────────┐
│  📖 PaperParser Agent               │
│  🔍 ResearchExpander Agent          │
│  ⚖️  NoveltyAssessor Agent          │
│  🏭 IndustryMapper Agent            │
└─────────────────────────────────────┘
    ↓
📊 Industry Insights & Applications
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** with pip
- **Node.js 18+** with npm/bun
- **LaTeX Distribution** (for PDF generation)
  - Windows: MikTeX or TeX Live
  - macOS: MacTeX
  - Linux: TeX Live

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/shemIA-Hack-Nation/praxis-ai.git
   cd praxis-ai
   ```

2. **Backend Setup**
   ```bash
   # Create virtual environment
   python -m venv .venv
   
   # Activate virtual environment
   # Windows (PowerShell)
   .\.venv\Scripts\Activate.ps1
   # Windows (Command Prompt)
   .venv\Scripts\activate.bat
   # macOS/Linux
   source .venv/bin/activate
   
   # Install Python dependencies
   pip install -r requirements.txt
   ```

3. **Frontend Setup**
   ```bash
   cd frontend
   
   # Install dependencies (using bun - faster)
   bun install
   # Or using npm
   npm install
   ```

4. **Environment Configuration**
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Add your API keys to .env
   # GROQ_API_KEY=your_groq_api_key_here
   # OPENAI_API_KEY=your_openai_api_key_here (optional)
   ```

5. **LaTeX Installation** *(for PDF generation)*
   
   **Windows:**
   ```powershell
   # Using winget
   winget install MiKTeX.MiKTeX
   # Or download from https://miktex.org/download
   ```
   
   **macOS:**
   ```bash
   # Using Homebrew
   brew install --cask mactex
   ```
   
   **Ubuntu/Debian:**
   ```bash
   sudo apt-get install texlive-full
   ```

### Running the Application

1. **Start Backend Server**
   ```bash
   # From project root with activated virtual environment
   python -m backend.app.main
   ```
   Backend will run on: `http://localhost:8000`

2. **Start Frontend Development Server**
   ```bash
   cd frontend
   
   # Using bun
   bun run dev
   # Or using npm
   npm run dev
   ```
   Frontend will run on: `http://localhost:5173`

3. **Access the Application**
   - Open your browser to `http://localhost:5173`
   - Upload a Jupyter notebook (.ipynb file)
   - Watch the multi-agent system generate your research paper
   - Download LaTeX and PDF files when complete

## 📋 API Documentation

### Core Endpoints

- `POST /api/v1/generate-paper-stream` - Generate research paper from notebook
- `GET /reports/{filename}` - Download generated LaTeX/PDF files
- `POST /api/v1/upload` - Upload notebook files

### WebSocket Streams

Real-time updates during paper generation:
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onmessage = (event) => {
    const update = JSON.parse(event.data);
    // Handle agent progress updates
};
```

## 🤖 Agent System Details

### Current Agents (Sprint 1)

#### 🔬 **NotebookAnalyzer Agent**
- **Purpose**: Extract code structure, methodology, and technical insights
- **Input**: Jupyter notebook (.ipynb)
- **Output**: Structured analysis with code blocks, data flow, and technical approach
- **Technologies**: AST parsing, code analysis, pattern recognition

#### 📝 **ContentGenerator Agent** 
- **Purpose**: Generate academic sections (Abstract, Introduction, Methodology, etc.)
- **Input**: Analysis results and extracted insights
- **Output**: Well-structured academic content in markdown
- **Technologies**: Groq LLM, academic writing patterns

#### 🖼️ **IllustrationGenerator Agent**
- **Purpose**: Create technical diagrams and visualizations
- **Input**: Code structure and methodology descriptions
- **Output**: SVG diagrams, flowcharts, and technical illustrations
- **Technologies**: Programmatic SVG generation, technical diagramming

#### 📋 **FormatterAgent**
- **Purpose**: Compile all content into professional research paper format
- **Input**: Generated content from all agents
- **Output**: LaTeX file with academic formatting, compiled PDF
- **Technologies**: LaTeX generation, PDF compilation, academic templates

### Upcoming Agents (Sprint 2: Industry Navigator)

#### 📖 **PaperParser Agent**
- **Purpose**: Extract core concepts from generated research papers
- **Input**: `Generated_Paper.md` or LaTeX file
- **Output**: Structured paper analysis with keywords, methods, and references
- **Example Output**:
  ```json
  {
    "title": "A CNN for Handwritten Digit Recognition...",
    "problem": "Classifying handwritten digits (MNIST)",
    "method": "Standard Sequential CNN",
    "keywords": ["CNN", "MNIST", "Image Classification", "Deep Learning"],
    "references": ["LeCun-1998", "Krizhevsky-2012"]
  }
  ```

#### 🔍 **ResearchExpander Agent**
- **Purpose**: Find related work and build research context
- **Input**: Paper keywords and references
- **Output**: Survey papers, related work, and citation network
- **Example Output**:
  ```json
  {
    "surveys": [
      {
        "title": "A Comprehensive Survey of Deep Learning for Image Classification",
        "url": "arxiv.org/...",
        "summary": "..."
      }
    ],
    "cited_by": [
      {
        "title": "LeNet-5: The original CNN paper...",
        "url": "...",
        "summary": "..."
      }
    ]
  }
  ```

#### ⚖️ **NoveltyAssessor Agent**
- **Purpose**: Evaluate research originality and contribution
- **Input**: User paper + research context
- **Output**: Novelty score and detailed assessment
- **Example Output**:
  ```markdown
  **Novelty Assessment (Score: 3/10)**
  
  The application of a standard CNN to the MNIST dataset is a well-established 
  benchmark problem (LeCun, 1998). The architecture itself does not present 
  a novel contribution.
  
  **Originality**: This appears to be a reimplementation of a classic solution. 
  The value is educational rather than a new research contribution.
  ```

#### 🏭 **IndustryMapper Agent**
- **Purpose**: Identify real-world commercial applications
- **Input**: Paper methodology and problem domain
- **Output**: Sector-specific application opportunities
- **Example Output**:
  ```json
  [
    {
      "sector": "Logistics & Postal",
      "application": "Automated sorting of mail by reading handwritten ZIP codes on envelopes, a direct extension of MNIST."
    },
    {
      "sector": "Fintech & Banking", 
      "application": "Digitizing handwritten numbers from checks or forms (Optical Character Recognition - OCR)."
    },
    {
      "sector": "Quality Assurance (Manufacturing)",
      "application": "Basic visual inspection to classify simple parts on an assembly line as 'Pass' or 'Fail' based on shape."
    }
  ]
  ```

## 🛠️ Development

### Project Structure

```
praxis-ai/
├── backend/                    # FastAPI backend application
│   ├── app/
│   │   ├── main.py            # FastAPI app entry point
│   │   ├── agents/            # Multi-agent system
│   │   │   ├── agent_tools.py # Shared agent utilities
│   │   │   ├── papergen/      # Paper generation agents
│   │   │   └── industrynav/   # Industry navigation agents (Sprint 2)
│   │   ├── api/v1/            # REST API endpoints
│   │   ├── core/              # Configuration and utilities
│   │   └── services/          # Business logic services
│   └── tests/                 # Backend test suite
├── frontend/                  # React + TypeScript frontend
│   ├── src/
│   │   ├── components/        # React components
│   │   ├── pages/            # Application pages
│   │   ├── hooks/            # Custom React hooks
│   │   └── lib/              # Utilities and helpers
│   └── public/               # Static assets
├── data/                     # Generated files and uploads
│   ├── reports/              # Generated LaTeX/PDF papers
│   ├── uploads/              # Uploaded notebooks
│   └── notebook_images/      # Extracted notebook data
├── templates/                # Template files
└── requirements.txt          # Python dependencies
```

### Adding New Agents

1. **Create Agent Class**
   ```python
   # backend/app/agents/your_agent.py
   from langchain.agents import BaseSingleActionAgent
   
   class YourAgent(BaseSingleActionAgent):
       def plan(self, intermediate_steps, **kwargs):
           # Agent logic here
           pass
   ```

2. **Register in Orchestrator**
   ```python
   # Add to orchestrator workflow
   self.your_agent = YourAgent()
   ```

3. **Update State Management**
   ```python
   # Update state TypedDict if needed
   class YourState(TypedDict):
       new_field: str
   ```

### Running Tests

```bash
# Backend tests
python -m pytest backend/tests/

# Frontend tests  
cd frontend
npm test
```

### Building for Production

```bash
# Backend (using Docker recommended)
docker build -t praxis-ai-backend .

# Frontend
cd frontend
npm run build
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# API Keys
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Application Settings
DEBUG=true
LOG_LEVEL=INFO

# File Upload Settings
MAX_UPLOAD_SIZE=50MB
ALLOWED_EXTENSIONS=.ipynb,.py

# LaTeX Settings
LATEX_COMPILER=pdflatex
LATEX_TIMEOUT=300
```

### API Key Setup

1. **Groq API Key** (Required)
   - Sign up at [console.groq.com](https://console.groq.com)
   - Create an API key
   - Add to `.env` file

2. **OpenAI API Key** (Optional)
   - Sign up at [platform.openai.com](https://platform.openai.com)
   - Create an API key  
   - Add to `.env` file for enhanced capabilities

## 📊 Usage Examples

### Basic Notebook Processing

1. **Upload Notebook**: Select a `.ipynb` file from your machine
2. **Monitor Progress**: Watch real-time agent coordination
3. **Download Results**: Get LaTeX source and compiled PDF

### Advanced Configuration

```python
# Custom agent configuration
from backend.app.agents.papergen.orchestrator_papergen import PapergenOrchestrator

orchestrator = PapergenOrchestrator(
    model_name="llama-3.1-70b-versatile",
    temperature=0.1,
    max_sections=6
)
```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the Repository**
2. **Create Feature Branch**: `git checkout -b feature/amazing-feature`
3. **Make Changes**: Follow our coding standards
4. **Add Tests**: Ensure your changes are tested
5. **Submit Pull Request**: Describe your changes clearly

### Coding Standards

- **Python**: Follow PEP 8, use type hints
- **TypeScript**: Follow Prettier formatting
- **Commits**: Use conventional commit messages
- **Documentation**: Update README for new features

## 🐛 Troubleshooting

### Common Issues

**PDF Generation Fails**
- Ensure LaTeX is installed and in PATH
- Check `pdflatex --version` works
- Install missing LaTeX packages as needed

**Agent Timeout Errors**
- Check your API keys are valid
- Verify internet connection for LLM calls
- Increase timeout in configuration

**Upload Issues**
- Ensure notebook is valid JSON
- Check file size limits (50MB default)
- Verify file extension is `.ipynb`

**Port Already in Use**
- Change ports in configuration
- Kill existing processes: `netstat -ano | findstr :8000`

### Getting Help

- **Issues**: [GitHub Issues](https://github.com/shemIA-Hack-Nation/praxis-ai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/shemIA-Hack-Nation/praxis-ai/discussions)
- **Documentation**: Check this README and code comments

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hack Nation 3.0** - Competition that inspired this project
- **ESPRIT** - Academic institution support
- **Groq** - High-performance LLM inference
- **LangChain** - Agent framework foundation
- **FastAPI** - Modern Python web framework
- **React** - Frontend framework excellence

## 🚧 Roadmap

### Sprint 2: Industry Navigator (Q1 2025)
- [ ] PaperParser Agent implementation
- [ ] ResearchExpander Agent with arXiv integration  
- [ ] NoveltyAssessor Agent with scoring system
- [ ] IndustryMapper Agent with sector database
- [ ] Navigator web interface and visualizations

### Sprint 3: Advanced Features (Q2 2025)
- [ ] Multi-language notebook support
- [ ] Collaborative research features
- [ ] Enterprise deployment options
- [ ] Advanced LaTeX templates
- [ ] Integration with research databases

### Sprint 4: AI Enhancements (Q3 2025)
- [ ] Custom model fine-tuning
- [ ] Improved code understanding
- [ ] Better technical diagrams
- [ ] Automated peer review
- [ ] Research trend analysis

---

**Built with ❤️ by the PraxisAI Team**

*Transforming Code into Knowledge, Research into Innovation*
