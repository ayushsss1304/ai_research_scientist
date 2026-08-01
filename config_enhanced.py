"""
Enhanced Configuration for AI Research Scientist
Supports all 8 research paper sources
"""

import os
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# API KEYS
# ============================================================================

# ArXiv (No key needed - FREE)

# Semantic Scholar (FREE + Optional key for higher limits)
SEMANTIC_SCHOLAR_API_KEY = os.getenv('SEMANTIC_SCHOLAR_API_KEY', '')

# IEEE Xplore (FREE - 200 calls/day)
IEEE_API_KEY = os.getenv('IEEE_API_KEY', '')

# Elsevier Scopus API
ELSEVIER_API_KEY = os.getenv('ELSEVIER_API_KEY', os.getenv('SCOPUS_API_KEY', ''))

# PubMed (FREE + Optional key for higher limits)
PUBMED_API_KEY = os.getenv('PUBMED_API_KEY', '')

# CORE (FREE key required)
# Get at: https://core.ac.uk/services/api
CORE_API_KEY = os.getenv('CORE_API_KEY', '')

# Google Scholar via SerpAPI (FREE tier: 100 searches/month)
# Get at: https://serpapi.com/
SERPAPI_KEY = os.getenv('SERPAPI_KEY', '')

# Anthropic API (for chatbot)
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')

# Email for OpenAlex and CrossRef (polite API access - FREE)
USER_EMAIL = os.getenv('USER_EMAIL', '')

# ============================================================================
# NEO4J CONFIGURATION
# ============================================================================

NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', '')

# ============================================================================
# PDF PROCESSING
# ============================================================================

USE_OCR = False  # Keep default processing fast; enable manually for scanned PDFs.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# OCR Settings
TESSERACT_LANGUAGE = 'eng'
PDF_DPI = 300

# ============================================================================
# DIRECTORIES
# ============================================================================

IS_VERCEL = bool(os.getenv('VERCEL'))
RUNTIME_DIR = os.getenv(
    'RUNTIME_DIR',
    os.path.join(tempfile.gettempdir(), 'research_scientist') if IS_VERCEL else '.'
)
DATA_DIR = os.path.join(RUNTIME_DIR, 'data')
PDF_DIR = os.path.join(DATA_DIR, 'pdfs')
CACHE_DIR = os.path.join(DATA_DIR, 'cache')
UPLOAD_DIR = os.path.join(RUNTIME_DIR, 'uploads')
RAG_STORAGE_DIR = os.path.join(RUNTIME_DIR, 'rag_storage')

# Create directories
for directory in [DATA_DIR, PDF_DIR, CACHE_DIR, UPLOAD_DIR, RAG_STORAGE_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================================
# SEARCH SETTINGS
# ============================================================================

# Default search settings
MAX_RESULTS_PER_SOURCE = 20
DEFAULT_START_YEAR = 2020
DEFAULT_END_YEAR = 2024
MIN_SIMILARITY_THRESHOLD = 0.3

# Deduplication settings
FUZZY_MATCH_THRESHOLD = 0.90  # 90% similarity for fuzzy matching
ENABLE_ADVANCED_DEDUPLICATION = True

# Default sources (customize as needed)
DEFAULT_SOURCES = [
    'scopus',
    'ieee',
    'semantic_scholar',
    'openalex',
    'arxiv'
]

# Optional sources (require API keys)
OPTIONAL_SOURCES = [
    'crossref',
    'pubmed',
    'core',         # Requires CORE_API_KEY
    'google_scholar'  # Requires SERPAPI_KEY
]

# ============================================================================
# EMBEDDING SETTINGS
# ============================================================================

# Embedding model for RAG chatbot
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# Alternative models (uncomment to use):
# EMBEDDING_MODEL = 'all-mpnet-base-v2'  # Better quality, slower
# EMBEDDING_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'  # Multilingual

# RAG settings
RAG_TOP_K = 3  # Number of chunks to retrieve
RAG_USE_RERANKING = True  # Use MMR reranking for diversity
RAG_MIN_SIMILARITY = 0.3  # Minimum similarity threshold

# ============================================================================
# FLASK WEB SERVER
# ============================================================================

FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = True
FLASK_SECRET_KEY = os.getenv('FLASK_SECRET_KEY') or os.urandom(24)

# Max file upload size (50 MB)
MAX_CONTENT_LENGTH = 50 * 1024 * 1024

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = 'research_scientist.log'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# ============================================================================
# FEATURE FLAGS
# ============================================================================

# Enable/disable features
ENABLE_KNOWLEDGE_GRAPH = os.getenv('ENABLE_KNOWLEDGE_GRAPH', 'true').lower() == 'true'
ENABLE_RAG_CHATBOT = os.getenv(
    'ENABLE_RAG_CHATBOT',
    'false' if IS_VERCEL else 'true'
).lower() == 'true'
ENABLE_WEB_INTERFACE = True
ENABLE_PDF_PROCESSING = True
ENABLE_OCR = True

# Experimental features
ENABLE_CITATION_NETWORK = False
ENABLE_AUTO_SUMMARIZATION = False
ENABLE_RECOMMENDATIONS = False

# ============================================================================
# RATE LIMITING
# ============================================================================

# Rate limit settings for different sources
RATE_LIMITS = {
    'arxiv': 3,  # seconds between requests
    'scopus': 0.25,
    'semantic_scholar': 0.5,
    'pubmed': 0.4,
    'ieee': 1,
    'openalex': 0.1,
    'crossref': 0.1,
    'core': 0.5,
    'google_scholar': 2
}

# ============================================================================
# CACHE SETTINGS
# ============================================================================

ENABLE_CACHE = True
CACHE_TTL = 3600  # Cache time-to-live in seconds (1 hour)
CACHE_MAX_SIZE = 1000  # Maximum cached items

# ============================================================================
# DATABASE SETTINGS (for future features)
# ============================================================================

# SQLite for metadata
SQLITE_DB = os.path.join(DATA_DIR, 'research_scientist.db')

# ============================================================================
# EXPORT SETTINGS
# ============================================================================

# Supported export formats
EXPORT_FORMATS = ['json', 'csv', 'bibtex', 'ris', 'markdown']

# Default export format
DEFAULT_EXPORT_FORMAT = 'json'

# ============================================================================
# ADVANCED SETTINGS
# ============================================================================

# Parallel processing
MAX_WORKERS = 4  # Number of parallel workers for scraping

# Timeout settings
REQUEST_TIMEOUT = 30  # seconds
PDF_PROCESSING_TIMEOUT = 300  # seconds

# Retry settings
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Validate configuration settings"""
    issues = []
    
    # Check Anthropic API key for chatbot
    if ENABLE_RAG_CHATBOT and not ANTHROPIC_API_KEY:
        issues.append("RAG Chatbot enabled but ANTHROPIC_API_KEY not set")
    
    # Check Neo4j settings
    if ENABLE_KNOWLEDGE_GRAPH:
        if not NEO4J_PASSWORD or NEO4J_PASSWORD == 'password':
            issues.append("Knowledge Graph enabled but Neo4j password not set")
    
    # Check email for polite API access
    if not USER_EMAIL or USER_EMAIL == 'your.email@example.com':
        issues.append("USER_EMAIL not set - some APIs may have lower rate limits")
    
    # Check optional API keys
    if IEEE_API_KEY and IEEE_API_KEY not in ['', 'your_key']:
        DEFAULT_SOURCES.append('ieee')
    
    if CORE_API_KEY and CORE_API_KEY not in ['', 'your_key']:
        DEFAULT_SOURCES.append('core')
    
    if SERPAPI_KEY and SERPAPI_KEY not in ['', 'your_key']:
        DEFAULT_SOURCES.append('google_scholar')
    
    if issues:
        print("\n⚠️  Configuration Issues:")
        for issue in issues:
            print(f"  - {issue}")
        print()
    
    return len(issues) == 0

# Run validation
if __name__ == '__main__':
    validate_config()

# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================

def display_config():
    """Display current configuration"""
    print("=" * 80)
    print("AI RESEARCH SCIENTIST - CONFIGURATION")
    print("=" * 80)
    
    print("\n📚 AVAILABLE SOURCES:")
    sources = []
    sources.append("✓ ArXiv (FREE)")
    sources.append("✓ Semantic Scholar (FREE" + ("+" if SEMANTIC_SCHOLAR_API_KEY else "") + ")")
    sources.append("✓ OpenAlex (FREE)")
    sources.append("✓ CrossRef (FREE)")
    sources.append("✓ PubMed (FREE" + ("+" if PUBMED_API_KEY else "") + ")")
    
    if IEEE_API_KEY:
        sources.append("✓ IEEE Xplore")
    else:
        sources.append("✗ IEEE Xplore (no key)")
    
    if CORE_API_KEY:
        sources.append("✓ CORE")
    else:
        sources.append("✗ CORE (no key)")
    
    if SERPAPI_KEY:
        sources.append("✓ Google Scholar")
    else:
        sources.append("✗ Google Scholar (no key)")
    
    for source in sources:
        print(f"  {source}")
    
    print("\n⚙️  FEATURES:")
    print(f"  Knowledge Graph: {'✓ Enabled' if ENABLE_KNOWLEDGE_GRAPH else '✗ Disabled'}")
    print(f"  RAG Chatbot: {'✓ Enabled' if ENABLE_RAG_CHATBOT and ANTHROPIC_API_KEY else '✗ Disabled'}")
    print(f"  Web Interface: {'✓ Enabled' if ENABLE_WEB_INTERFACE else '✗ Disabled'}")
    print(f"  PDF Processing: {'✓ Enabled' if ENABLE_PDF_PROCESSING else '✗ Disabled'}")
    print(f"  OCR Support: {'✓ Enabled' if ENABLE_OCR else '✗ Disabled'}")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    display_config()
