import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
IEEE_API_KEY = os.getenv('IEEE_API_KEY', '')
SEMANTIC_SCHOLAR_API_KEY = os.getenv('SEMANTIC_SCHOLAR_API_KEY', '')  # Optional, increases rate limit
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')

# Neo4j Configuration
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', '')

# PDF Processing
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