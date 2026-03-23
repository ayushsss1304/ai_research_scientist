"""
Standalone RAG Chatbot with Ollama
Uses local Ollama models instead of Anthropic API
Completely free and runs locally!
"""

import requests
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
import os
import pickle
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Client for Ollama API
    Ollama runs locally - no API key needed!
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3"):
        """
        Initialize Ollama client
        
        Args:
            base_url: Ollama server URL (default: localhost)
            model: Model name (llama3, mistral, phi3, etc.)
        """
        self.base_url = base_url
        self.model = model
        self._verify_ollama()
    
    def _verify_ollama(self):
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [m['name'] for m in models]
                
                logger.info(f"✅ Ollama is running")
                logger.info(f"📦 Available models: {', '.join(available_models)}")
                
                # Check if requested model is available
                if self.model not in available_models:
                    logger.warning(f"⚠️  Model '{self.model}' not found")
                    logger.warning(f"💡 Run: ollama pull {self.model}")
            else:
                logger.error("❌ Ollama server not responding")
        except Exception as e:
            logger.error(f"❌ Cannot connect to Ollama: {e}")
            logger.error("💡 Install: https://ollama.ai/download")
            logger.error("💡 Then run: ollama serve")
    
    def generate(self, prompt: str, system: str = "", 
                 temperature: float = 0.7, 
                 max_tokens: int = 2000) -> str:
        """
        Generate text using Ollama
        
        Args:
            prompt: User prompt
            system: System prompt
            temperature: Temperature (0-1)
            max_tokens: Max tokens to generate
            
        Returns:
            Generated text
        """
        try:
            url = f"{self.base_url}/api/generate"
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "system": system,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', '')
        
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return f"Error: {str(e)}"
    
    def chat(self, messages: List[Dict], temperature: float = 0.7) -> str:
        """
        Chat with Ollama using chat format
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Temperature
            
        Returns:
            Generated response
        """
        try:
            url = f"{self.base_url}/api/chat"
            
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature
                }
            }
            
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            return result.get('message', {}).get('content', '')
        
        except Exception as e:
            logger.error(f"Ollama chat error: {e}")
            return f"Error: {str(e)}"


class DocumentStore:
    """
    Persistent document storage for RAG pipeline
    Same as before - no changes needed
    """
    
    def __init__(self, storage_dir: str = 'rag_storage'):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        self.documents = {}
        self.metadata_file = os.path.join(storage_dir, 'documents.pkl')
        
        self._load_documents()
    
    def add_document(self, doc_id: str, text: str, chunks: List[str], 
                     embeddings: np.ndarray, metadata: Dict):
        """Add document to store"""
        self.documents[doc_id] = {
            'text': text,
            'chunks': chunks,
            'embeddings': embeddings,
            'metadata': metadata,
            'added_at': datetime.now().isoformat()
        }
        
        self._save_documents()
        logger.info(f"Added document to store: {doc_id}")
    
    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Get document from store"""
        return self.documents.get(doc_id)
    
    def list_documents(self) -> List[str]:
        """List all document IDs"""
        return list(self.documents.keys())
    
    def remove_document(self, doc_id: str):
        """Remove document from store"""
        if doc_id in self.documents:
            del self.documents[doc_id]
            self._save_documents()
            logger.info(f"Removed document: {doc_id}")
    
    def clear_all(self):
        """Clear all documents"""
        self.documents = {}
        self._save_documents()
        logger.info("Cleared all documents")
    
    def _save_documents(self):
        """Save documents to disk"""
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.documents, f)
    
    def _load_documents(self):
        """Load documents from disk"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'rb') as f:
                    self.documents = pickle.load(f)
                logger.info(f"Loaded {len(self.documents)} documents from storage")
            except Exception as e:
                logger.warning(f"Could not load documents: {e}")
                self.documents = {}


class AdvancedEmbeddingManager:
    """
    Same as before - embeddings don't change
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded")
    
    def create_embeddings(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Create embeddings for texts"""
        return self.model.encode(texts, show_progress_bar=show_progress)
    
    def create_query_embedding(self, query: str) -> np.ndarray:
        """Create embedding for query"""
        return self.model.encode([query])[0]
    
    def find_similar_chunks(self, query_embedding: np.ndarray, 
                           chunk_embeddings: np.ndarray,
                           chunks: List[str],
                           top_k: int = 5,
                           min_similarity: float = 0.0) -> List[Tuple[str, float, int]]:
        """Find similar chunks"""
        similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] >= min_similarity:
                results.append((chunks[idx], similarities[idx], idx))
        
        return results
    
    def rerank_chunks(self, query: str, chunks: List[Tuple[str, float, int]], 
                      diversity_weight: float = 0.3) -> List[Tuple[str, float, int]]:
        """Rerank chunks for diversity using MMR"""
        if len(chunks) <= 1:
            return chunks
        
        selected = []
        remaining = chunks.copy()
        selected.append(remaining.pop(0))
        
        while remaining and len(selected) < len(chunks):
            mmr_scores = []
            
            for chunk, similarity, idx in remaining:
                relevance = similarity
                
                diversities = []
                for sel_chunk, _, _ in selected:
                    chunk_words = set(chunk.lower().split())
                    sel_words = set(sel_chunk.lower().split())
                    overlap = len(chunk_words & sel_words) / max(len(chunk_words), len(sel_words))
                    diversities.append(1 - overlap)
                
                diversity = min(diversities) if diversities else 1.0
                mmr = (1 - diversity_weight) * relevance + diversity_weight * diversity
                mmr_scores.append((mmr, (chunk, similarity, idx)))
            
            mmr_scores.sort(reverse=True)
            best = mmr_scores[0][1]
            selected.append(best)
            remaining.remove(best)
        
        return selected


class OllamaRAGChatbot:
    """
    RAG Chatbot using Ollama (Local, Free!)
    Same features as Anthropic version but runs locally
    """
    
    def __init__(self, 
                 model: str = "llama3",
                 ollama_url: str = "http://localhost:11434",
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 storage_dir: str = 'rag_storage'):
        """
        Initialize Ollama RAG chatbot
        
        Args:
            model: Ollama model name (llama3, mistral, phi3, etc.)
            ollama_url: Ollama server URL
            embedding_model: Sentence transformer model
            storage_dir: Storage directory
        """
        self.client = OllamaClient(ollama_url, model)
        self.embedding_manager = AdvancedEmbeddingManager(embedding_model)
        self.document_store = DocumentStore(storage_dir)
        self.conversation_history = []
        
        logger.info("✅ Ollama RAG Chatbot initialized")
        logger.info(f"🤖 Model: {model}")
        logger.info(f"📚 Documents: {len(self.document_store.list_documents())}")
    
    def add_document_from_file(self, file_path: str, doc_id: Optional[str] = None,
                               chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Add document from file - SAME AS BEFORE
        """
        if doc_id is None:
            doc_id = os.path.basename(file_path).replace('.pdf', '').replace('.txt', '')
        
        if doc_id in self.document_store.list_documents():
            logger.warning(f"Document {doc_id} already exists")
            return False
        
        logger.info(f"Processing document: {file_path}")
        
        # Extract text
        from pdf_processing.pdf_processor import PDFProcessor
        processor = PDFProcessor()
        result = processor.process_document(file_path)
        
        if result['error']:
            logger.error(f"Error processing document: {result['error']}")
            return False
        
        text = result['text']
        chunks = self._create_smart_chunks(text, chunk_size, chunk_overlap)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Create embeddings
        logger.info("Creating embeddings...")
        embeddings = self.embedding_manager.create_embeddings(chunks)
        
        # Store document
        metadata = {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'file_type': result['file_type'],
            'num_chunks': len(chunks),
            'text_length': len(text),
            **result.get('metadata', {})
        }
        
        self.document_store.add_document(doc_id, text, chunks, embeddings, metadata)
        logger.info(f"✅ Document added successfully: {doc_id}")
        return True
    
    def add_document_from_text(self, text: str, doc_id: str, metadata: Optional[Dict] = None,
                               chunk_size: int = 1000, chunk_overlap: int = 200):
        """Add document from text - SAME AS BEFORE"""
        logger.info(f"Adding text document: {doc_id}")
        
        chunks = self._create_smart_chunks(text, chunk_size, chunk_overlap)
        embeddings = self.embedding_manager.create_embeddings(chunks)
        
        meta = metadata or {}
        meta.update({'num_chunks': len(chunks), 'text_length': len(text)})
        
        self.document_store.add_document(doc_id, text, chunks, embeddings, meta)
        logger.info(f"✅ Text document added: {doc_id}")
        return True
    
    def _create_smart_chunks(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Create smart chunks - SAME AS BEFORE"""
        if not text:
            return []
        
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def chat(self, user_message: str, 
             top_k: int = 5,
             min_similarity: float = 0.3,
             use_reranking: bool = True,
             use_history: bool = True,
             temperature: float = 0.7) -> str:
        """
        Chat with documents using Ollama
        
        Args:
            user_message: User's query
            top_k: Number of chunks to retrieve
            min_similarity: Minimum similarity threshold
            use_reranking: Use MMR reranking
            use_history: Use conversation history
            temperature: Model temperature (0-1)
            
        Returns:
            Assistant's response
        """
        if not self.document_store.list_documents():
            return "No documents loaded. Please add documents first."
        
        logger.info(f"Processing query: {user_message[:50]}...")
        
        # Create query embedding
        query_embedding = self.embedding_manager.create_query_embedding(user_message)
        
        # Find relevant chunks
        all_relevant_chunks = []
        
        for doc_id in self.document_store.list_documents():
            doc = self.document_store.get_document(doc_id)
            
            similar_chunks = self.embedding_manager.find_similar_chunks(
                query_embedding,
                doc['embeddings'],
                doc['chunks'],
                top_k=top_k,
                min_similarity=min_similarity
            )
            
            for chunk, similarity, idx in similar_chunks:
                all_relevant_chunks.append({
                    'doc_id': doc_id,
                    'chunk': chunk,
                    'similarity': similarity,
                    'index': idx
                })
        
        # Sort and limit
        all_relevant_chunks.sort(key=lambda x: x['similarity'], reverse=True)
        all_relevant_chunks = all_relevant_chunks[:top_k]
        
        # Rerank if enabled
        if use_reranking and len(all_relevant_chunks) > 1:
            chunk_tuples = [(c['chunk'], c['similarity'], c['index']) for c in all_relevant_chunks]
            reranked = self.embedding_manager.rerank_chunks(user_message, chunk_tuples)
            
            reranked_chunks = []
            for chunk, similarity, idx in reranked:
                for original in all_relevant_chunks:
                    if original['chunk'] == chunk:
                        reranked_chunks.append(original)
                        break
            
            all_relevant_chunks = reranked_chunks
        
        # Build context
        context = self._build_context(all_relevant_chunks)
        
        # Build system message
        system_message = self._build_system_message(context)
        
        # Build prompt (Ollama format)
        if use_history and self.conversation_history:
            # Use chat API with history
            messages = [{"role": "system", "content": system_message}]
            messages.extend(self.conversation_history)
            messages.append({"role": "user", "content": user_message})
            
            try:
                assistant_message = self.client.chat(messages, temperature)
            except Exception as e:
                logger.error(f"Error calling Ollama: {e}")
                return f"Error: {str(e)}"
        else:
            # Use generate API without history
            full_prompt = f"{system_message}\n\nUser: {user_message}\n\nAssistant:"
            
            try:
                assistant_message = self.client.generate(
                    full_prompt, 
                    temperature=temperature
                )
            except Exception as e:
                logger.error(f"Error calling Ollama: {e}")
                return f"Error: {str(e)}"
        
        # Update history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        # Keep history manageable
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        logger.info("✅ Response generated")
        return assistant_message
    
    def _build_context(self, relevant_chunks: List[Dict]) -> str:
        """Build context from relevant chunks"""
        context = "# Relevant information from documents:\n\n"
        
        for i, chunk_data in enumerate(relevant_chunks, 1):
            context += f"## Excerpt {i} (from {chunk_data['doc_id']}, "
            context += f"relevance: {chunk_data['similarity']:.2%})\n"
            context += f"{chunk_data['chunk']}\n\n"
        
        return context
    
    def _build_system_message(self, context: str) -> str:
        """Build system message"""
        doc_list = ", ".join(self.document_store.list_documents())
        
        system = f"""You are a research assistant helping users understand research documents.

Available documents: {doc_list}

Guidelines:
1. Base answers on the provided context
2. Cite specific documents
3. Be precise and academic
4. If information isn't in context, say so
5. Provide structured comparisons when asked

{context}

Always reference which document your information comes from."""
        
        return system
    
    def get_statistics(self) -> Dict:
        """Get chatbot statistics"""
        total_chunks = 0
        total_chars = 0
        
        for doc_id in self.document_store.list_documents():
            doc = self.document_store.get_document(doc_id)
            total_chunks += len(doc['chunks'])
            total_chars += len(doc['text'])
        
        return {
            'num_documents': len(self.document_store.list_documents()),
            'total_chunks': total_chunks,
            'total_characters': total_chars,
            'document_ids': self.document_store.list_documents(),
            'conversation_length': len(self.conversation_history)
        }
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def remove_document(self, doc_id: str):
        """Remove document"""
        self.document_store.remove_document(doc_id)
    
    def list_documents(self) -> List[Dict]:
        """List all documents with metadata"""
        docs = []
        for doc_id in self.document_store.list_documents():
            doc = self.document_store.get_document(doc_id)
            docs.append({
                'doc_id': doc_id,
                'metadata': doc['metadata'],
                'added_at': doc['added_at'],
                'num_chunks': len(doc['chunks'])
            })
        return docs