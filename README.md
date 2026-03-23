# 🔬 AI Research Scientist

An intelligent research paper discovery, analysis, and management platform powered by AI. Search across 8+ academic sources, manage papers in a Neo4j knowledge graph, chat with your documents using RAG (Retrieval-Augmented Generation), and process PDFs with OCR support.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)
![Neo4j](https://img.shields.io/badge/Neo4j-5.x-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## ✨ Features

### 📚 Multi-Source Paper Search
- **ArXiv** — Free, open access preprints
- **Semantic Scholar** — AI-powered academic search
- **OpenAlex** — Comprehensive open academic data
- **CrossRef** — DOI metadata registry
- **PubMed** — Biomedical literature
- **IEEE Xplore** — Engineering & CS papers
- **CORE** — Open access aggregator
- **Google Scholar** — Via SerpAPI

### 🧠 Knowledge Graph (Neo4j)
- Store papers, authors, and research fields as graph nodes
- Explore relationships between papers and authors
- Interactive graph visualization in the web UI

### 🤖 RAG Chatbot
- Chat with your uploaded research papers
- Uses **Ollama** (local, free) or **Anthropic Claude** for LLM
- Sentence-transformer embeddings with MMR reranking
- Persistent document storage

### 📄 PDF Processing
- Native text extraction via PyPDF2
- OCR fallback using Tesseract for scanned documents
- Smart chunking with sentence boundaries

### 🌐 Web Interface
- Modern Flask-based UI with dark mode & glassmorphism
- Paper search, filtering, and export (JSON, CSV, BibTeX)
- Knowledge graph visualization
- Document upload and RAG chat

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.9+**
- **Neo4j** (optional, for knowledge graph)
- **Ollama** (optional, for local LLM chat) — [Install Ollama](https://ollama.ai/download)
- **Tesseract OCR** (optional, for scanned PDFs)
- **Poppler** (optional, for PDF-to-image conversion)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/ai-research-scientist.git
   cd ai-research-scientist
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Run the web application**
   ```bash
   cd web
   python app.py
   ```
   Open [http://localhost:5000](http://localhost:5000) in your browser.

### Optional Setup

- **Neo4j**: Install and start Neo4j, then update `NEO4J_*` variables in `.env`
- **Ollama**: Install Ollama, pull a model (`ollama pull llama3`), and run `ollama serve`
- **Tesseract**: Install Tesseract OCR for scanned PDF support

---

## 📁 Project Structure

```
research_scientist/
├── web/                    # Flask web application
│   ├── app.py              # Main Flask app with all API routes
│   └── templates/          # HTML templates
├── scrapers/               # Paper scraper modules
│   ├── arxiv_scraper.py
│   ├── semantic_scholar_scraper.py
│   ├── ieee_scraper.py
│   ├── additional_sources.py   # PubMed, OpenAlex, CrossRef, CORE, Google Scholar
│   ├── enhanced_paper_scraper.py   # Unified scraper with deduplication
│   └── base_scraper.py
├── knowledge_graph/        # Neo4j knowledge graph
│   └── kg_manager.py
├── rag_pipeline/           # RAG chatbot pipeline
│   ├── ollama_chatbot.py   # Ollama-based RAG (local, free)
│   └── standalone_chatbot.py   # Anthropic-based RAG
├── pdf_processing/         # PDF & OCR processing
│   ├── pdf_processor.py
│   └── ocr_processor.py
├── chatbot/                # Claude-based chatbot
│   └── research_chatbot.py
├── config.py               # Basic configuration
├── config_enhanced.py      # Full configuration with all sources
├── main.py                 # CLI entry point
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
└── .gitignore
```

---

## 🔑 API Keys

| Source | Key Required | Free Tier | Get Key |
|--------|-------------|-----------|---------|
| ArXiv | ❌ No | Unlimited | — |
| Semantic Scholar | ⚡ Optional | Yes (rate limited) | [Link](https://www.semanticscholar.org/product/api) |
| OpenAlex | ❌ No | Unlimited | — |
| CrossRef | ❌ No | Unlimited | — |
| PubMed | ⚡ Optional | Yes (rate limited) | [Link](https://www.ncbi.nlm.nih.gov/account/) |
| IEEE Xplore | ✅ Yes | 200 calls/day | [Link](https://developer.ieee.org/) |
| CORE | ✅ Yes | Free | [Link](https://core.ac.uk/services/api) |
| Google Scholar | ✅ Yes | 100/month | [Link](https://serpapi.com/) |
| Anthropic | ✅ Yes | Paid | [Link](https://console.anthropic.com/) |

---

## 🖥️ Usage

### Web Interface
```bash
cd web && python app.py
```

### CLI
```bash
# Search papers
python main.py search "machine learning" --max-results 20 --start-year 2020

# Process a PDF
python main.py process-pdf path/to/paper.pdf

# Interactive chat with papers
python main.py chat --pdf paper1.pdf --pdf paper2.pdf

# Search knowledge graph
python main.py kg-search "neural networks"
```

---

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Database**: Neo4j (Graph DB)
- **AI/ML**: Sentence Transformers, Ollama, Anthropic Claude
- **PDF**: PyPDF2, Tesseract OCR, pdf2image
- **Frontend**: HTML, CSS, JavaScript, Chart.js

---

## 📄 License

This project is licensed under the MIT License.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.
