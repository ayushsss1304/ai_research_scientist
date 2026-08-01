"""RAG providers, with local/heavy integrations kept optional."""

try:
    from .standalone_chatbot import StandaloneRAGChatbot as AnthropicRAGChatbot
except ImportError:
    AnthropicRAGChatbot = None

try:
    from .ollama_chatbot import OllamaRAGChatbot
except ImportError:
    OllamaRAGChatbot = None

from .openrouter_chatbot import OpenRouterRAGChatbot

__all__ = ["AnthropicRAGChatbot", "OllamaRAGChatbot", "OpenRouterRAGChatbot"]
