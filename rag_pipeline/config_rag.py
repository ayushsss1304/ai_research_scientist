"""
RAG Pipeline Configuration
Easy switching between Anthropic and Ollama
"""

import os

# ============================================================================
# CHOOSE YOUR LLM PROVIDER
# ============================================================================

# Options: 'anthropic' or 'ollama'
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'ollama')  # Change to 'anthropic' if you want

# ============================================================================
# ANTHROPIC SETTINGS
# ============================================================================

ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
ANTHROPIC_MODEL = 'claude-sonnet-4-20250514'
ANTHROPIC_MAX_TOKENS = 4000

# ============================================================================
# OLLAMA SETTINGS
# ============================================================================

OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3')  # or mistral, phi3, etc.

# Available Ollama models with descriptions
OLLAMA_MODELS = {
    'llama3': 'Best balance (8GB RAM)',
    'llama3:70b': 'Best quality (40GB RAM)', 
    'mistral': 'Fast responses (6GB RAM)',
    'phi3': 'Smallest model (4GB RAM)',
    'gemma2': 'Google model (9GB RAM)',
    'codellama': 'Best for code (7GB RAM)'
}

# ============================================================================
# EMBEDDING SETTINGS (Same for both)
# ============================================================================

EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# Alternative embedding models:
# EMBEDDING_MODEL = 'all-mpnet-base-v2'  # Better quality, slower
# EMBEDDING_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'  # Multilingual

# ============================================================================
# RAG SETTINGS (Same for both)
# ============================================================================

# Storage
RAG_STORAGE_DIR = 'rag_storage'
ANTHROPIC_STORAGE_DIR = 'rag_storage_anthropic'
OLLAMA_STORAGE_DIR = 'rag_storage_ollama'

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval
TOP_K = 5
MIN_SIMILARITY = 0.3
USE_RERANKING = True
DIVERSITY_WEIGHT = 0.3

# Generation
TEMPERATURE = 0.7
USE_HISTORY = True
MAX_HISTORY_LENGTH = 20

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_chatbot():
    """
    Get the configured chatbot instance
    Returns either Anthropic or Ollama chatbot based on LLM_PROVIDER
    """
    if LLM_PROVIDER == 'anthropic':
        from .standalone_chatbot import StandaloneRAGChatbot
        
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not set!")
        
        return StandaloneRAGChatbot(
            anthropic_api_key=ANTHROPIC_API_KEY,
            embedding_model=EMBEDDING_MODEL,
            storage_dir=ANTHROPIC_STORAGE_DIR
        )
    
    elif LLM_PROVIDER == 'ollama':
        from .ollama_chatbot import OllamaRAGChatbot
        
        return OllamaRAGChatbot(
            model=OLLAMA_MODEL,
            ollama_url=OLLAMA_BASE_URL,
            embedding_model=EMBEDDING_MODEL,
            storage_dir=OLLAMA_STORAGE_DIR
        )
    
    else:
        raise ValueError(f"Invalid LLM_PROVIDER: {LLM_PROVIDER}")


def display_config():
    """Display current configuration"""
    print("=" * 80)
    print("RAG PIPELINE CONFIGURATION")
    print("=" * 80)
    
    print(f"\n🤖 LLM Provider: {LLM_PROVIDER.upper()}")
    
    if LLM_PROVIDER == 'anthropic':
        print(f"   Model: {ANTHROPIC_MODEL}")
        print(f"   API Key: {'✅ Set' if ANTHROPIC_API_KEY else '❌ Not Set'}")
        print(f"   Storage: {ANTHROPIC_STORAGE_DIR}")
        print(f"   Cost: 💰 Paid API")
    
    elif LLM_PROVIDER == 'ollama':
        print(f"   Model: {OLLAMA_MODEL}")
        print(f"   URL: {OLLAMA_BASE_URL}")
        print(f"   Storage: {OLLAMA_STORAGE_DIR}")
        print(f"   Cost: ✅ FREE")
    
    print(f"\n📊 Embedding Model: {EMBEDDING_MODEL}")
    print(f"📁 Chunk Size: {CHUNK_SIZE} (overlap: {CHUNK_OVERLAP})")
    print(f"🔍 Retrieval: Top-{TOP_K} chunks (min similarity: {MIN_SIMILARITY})")
    print(f"🎯 Reranking: {'✅ Enabled' if USE_RERANKING else '❌ Disabled'}")
    print(f"🌡️  Temperature: {TEMPERATURE}")
    
    print("\n" + "=" * 80)


def switch_provider(provider: str):
    """Switch between Anthropic and Ollama"""
    global LLM_PROVIDER
    
    if provider not in ['anthropic', 'ollama']:
        raise ValueError(f"Invalid provider: {provider}")
    
    LLM_PROVIDER = provider
    print(f"✅ Switched to {provider}")


if __name__ == '__main__':
    display_config()