"""
Setup script to create complete project structure
Run this to create all necessary files and directories
"""

import os
import sys

def create_directory_structure():
    """Create all necessary directories"""
    directories = [
        'scrapers',
        'knowledge_graph',
        'pdf_processing',
        'chatbot',
        'web',
        'web/templates',
        'utils',
        'data',
        'data/pdfs',
        'data/cache',
        'uploads'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def create_init_files():
    """Create __init__.py files for all packages"""
    
    init_files = {
        'scrapers/__init__.py': '''"""
Scrapers package for fetching research papers from multiple sources
"""

from .arxiv_scraper import ArxivScraper
from .semantic_scholar_scraper import SemanticScholarScraper
from .ieee_scraper import IEEEScraper
from .paper_scraper import PaperScraper

__all__ = [
    'ArxivScraper',
    'SemanticScholarScraper', 
    'IEEEScraper',
    'PaperScraper'
]
''',
        
        'knowledge_graph/__init__.py': '''"""
Knowledge graph package for managing research paper relationships
"""

from .kg_manager import KnowledgeGraphManager

__all__ = ['KnowledgeGraphManager']
''',
        
        'pdf_processing/__init__.py': '''"""
PDF processing package for text extraction and OCR
"""

from .pdf_processor import PDFProcessor, OCRProcessor

__all__ = ['PDFProcessor', 'OCRProcessor']
''',
        
        'chatbot/__init__.py': '''"""
Chatbot package for AI-powered paper analysis
"""

from .research_chatbot import ResearchChatbot, EmbeddingManager

__all__ = ['ResearchChatbot', 'EmbeddingManager']
''',
        
        'web/__init__.py': '''"""
Web interface package for Flask application
"""

from .app import create_app

__all__ = ['create_app']
''',
        
        'utils/__init__.py': '''"""
Utility functions package
"""

from .helpers import *

__all__ = [
    'clean_text',
    'extract_keywords',
    'calculate_similarity',
    'format_citations',
    'download_pdf',
    'validate_doi',
    'parse_date'
]
'''
    }
    
    for filepath, content in init_files.items():
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ Created: {filepath}")

def create_config_file():
    """Create config.py file"""
    config_content = '''"""
Configuration file for AI Research Scientist
"""

import os

# API Keys - Set these in environment variables or directly here
IEEE_API_KEY = os.getenv('IEEE_API_KEY', '')
SEMANTIC_SCHOLAR_API_KEY = os.getenv('SEMANTIC_SCHOLAR_API_KEY', '')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')

# Neo4j Configuration
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'password')

# PDF Processing Settings
USE_OCR = True
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Directories
DATA_DIR = 'data'
PDF_DIR = os.path.join(DATA_DIR, 'pdfs')
CACHE_DIR = os.path.join(DATA_DIR, 'cache')
UPLOAD_DIR = 'uploads'

# Create directories
for directory in [DATA_DIR, PDF_DIR, CACHE_DIR, UPLOAD_DIR]:
    os.makedirs(directory, exist_ok=True)

# Search Settings
MAX_RESULTS_PER_SOURCE = 20
DEFAULT_START_YEAR = 2020
DEFAULT_END_YEAR = 2024

# Embedding Model
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# Flask Settings
FLASK_SECRET_KEY = os.urandom(24)
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = True
'''
    
    with open('config.py', 'w', encoding='utf-8') as f:
        f.write(config_content)
    print("✓ Created: config.py")

def create_requirements_file():
    """Create requirements.txt"""
    requirements = '''requests==2.31.0
PyPDF2==3.0.1
pytesseract==0.3.10
pdf2image==1.16.3
Pillow==10.1.0
neo4j==5.14.0
sentence-transformers==2.2.2
scikit-learn==1.3.2
numpy==1.24.3
anthropic==0.8.1
flask==3.0.0
werkzeug==3.0.1
python-docx==1.1.0
'''
    
    with open('requirements.txt', 'w', encoding='utf-8') as f:
        f.write(requirements)
    print("✓ Created: requirements.txt")

def create_readme():
    """Create README.md"""
    readme = '''# AI Research Scientist

Complete AI-powered research paper management system.

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install system dependencies:
- Tesseract OCR
- Poppler (for PDF processing)
- Neo4j (for knowledge graph)

3. Configure API keys in config.py

4. Run:
```bash
# Search papers
python main.py search "machine learning"

# Start web interface
python main.py web
```

## Features

- Multi-source paper scraping (ArXiv, Semantic Scholar, IEEE)
- Knowledge graph with Neo4j
- PDF processing with OCR
- AI chatbot for paper analysis
- Web interface

See full documentation in the artifacts.
'''
    
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(readme)
    print("✓ Created: README.md")

def create_gitignore():
    """Create .gitignore file"""
    gitignore = '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Project specific
uploads/
data/cache/
*.pdf

# Environment
.env
config_local.py

# Neo4j
*.db

# Logs
*.log
'''
    
    with open('.gitignore', 'w', encoding='utf-8') as f:
        f.write(gitignore)
    print("✓ Created: .gitignore")

def check_existing_files():
    """Check which important files already exist"""
    important_files = [
        'main.py',
        'knowledge_graph/kg_manager.py',
        'pdf_processing/pdf_processor.py',
        'chatbot/research_chatbot.py',
        'web/app.py',
        'utils/helpers.py'
    ]
    
    print("\n📋 Checking existing files:")
    for filepath in important_files:
        exists = os.path.exists(filepath)
        status = "✓ EXISTS" if exists else "✗ MISSING"
        print(f"  {status}: {filepath}")
    
    print()

def main():
    """Main setup function"""
    print("=" * 60)
    print("AI Research Scientist - Project Setup")
    print("=" * 60)
    print()
    
    # Check current directory
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    print()
    
    # Ask for confirmation
    response = input("Create project structure in current directory? (y/n): ")
    if response.lower() != 'y':
        print("Setup cancelled.")
        return
    
    print("\n🚀 Creating project structure...\n")
    
    # Create everything
    create_directory_structure()
    print()
    
    create_init_files()
    print()
    
    create_config_file()
    create_requirements_file()
    create_readme()
    create_gitignore()
    
    print("\n" + "=" * 60)
    print("✅ Project structure created successfully!")
    print("=" * 60)
    print()
    
    # Check what's missing
    check_existing_files()
    
    print("📝 Next steps:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Install system dependencies (Tesseract, Poppler, Neo4j)")
    print("  3. Configure API keys in config.py")
    print("  4. Copy the remaining Python files from the artifacts")
    print("  5. Run: python main.py search 'machine learning'")
    print()

if __name__ == '__main__':
    main()