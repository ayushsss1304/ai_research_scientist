"""
Standalone RAG Chatbot Pipeline
Completely independent from scraper and knowledge graph
Focuses solely on PDF chat with advanced RAG techniques
"""

import anthropic
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


class DocumentStore:
    """
    Persistent document storage for RAG pipeline
    Stores documents, embeddings, and metadata separately from other systems
    """
    
    def __init__(self, storage_dir: str = 'rag_storage'):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        self.documents = {}  # doc_id -> document data
        self.metadata_file = os.path.join(storage_dir, 'documents.pkl')
        
        # Load existing documents
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
    Advanced embedding manager with multiple strategies
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
        """
        Find similar chunks with advanced filtering
        
        Returns:
            List of (chunk, similarity, index) tuples
        """
        similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] >= min_similarity:
                results.append((chunks[idx], similarities[idx], idx))
        
        return results
    
    def rerank_chunks(self, query: str, chunks: List[Tuple[str, float, int]], 
                      diversity_weight: float = 0.3) -> List[Tuple[str, float, int]]:
        """
        Rerank chunks for diversity and relevance
        Implements Maximal Marginal Relevance (MMR)
        """
        if len(chunks) <= 1:
            return chunks
        
        selected = []
        remaining = chunks.copy()
        
        # Select first (most relevant)
        selected.append(remaining.pop(0))
        
        while remaining and len(selected) < len(chunks):
            mmr_scores = []
            
            for chunk, similarity, idx in remaining:
                # Relevance score
                relevance = similarity
                
                # Diversity score (dissimilarity to already selected)
                diversities = []
                for sel_chunk, _, _ in selected:
                    # Simple word overlap as diversity measure
                    chunk_words = set(chunk.lower().split())
                    sel_words = set(sel_chunk.lower().split())
                    overlap = len(chunk_words & sel_words) / max(len(chunk_words), len(sel_words))
                    diversities.append(1 - overlap)
                
                diversity = min(diversities) if diversities else 1.0
                
                # MMR score
                mmr = (1 - diversity_weight) * relevance + diversity_weight * diversity
                mmr_scores.append((mmr, (chunk, similarity, idx)))
            
            # Select best MMR score
            mmr_scores.sort(reverse=True)
            best = mmr_scores[0][1]
            selected.append(best)
            remaining.remove(best)
        
        return selected


class StandaloneRAGChatbot:
    """
    Standalone RAG chatbot completely independent from other systems
    """
    
    def __init__(self, anthropic_api_key: str, 
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 storage_dir: str = 'rag_storage'):
        """
        Initialize standalone RAG chatbot
        
        Args:
            anthropic_api_key: Anthropic API key
            embedding_model: Embedding model name
            storage_dir: Directory for document storage
        """
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.embedding_manager = AdvancedEmbeddingManager(embedding_model)
        self.document_store = DocumentStore(storage_dir)
        self.conversation_history = []
        
        logger.info("✓ Standalone RAG Chatbot initialized")
        logger.info(f"✓ Documents in store: {len(self.document_store.list_documents())}")
    
    def add_document_from_file(self, file_path: str, doc_id: Optional[str] = None,
                               chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Add document from file (PDF, TXT, etc.)
        
        Args:
            file_path: Path to document file
            doc_id: Document ID (default: filename)
            chunk_size: Size of chunks
            chunk_overlap: Overlap between chunks
        """
        if doc_id is None:
            doc_id = os.path.basename(file_path).replace('.pdf', '').replace('.txt', '')
        
        # Check if already exists
        if doc_id in self.document_store.list_documents():
            logger.warning(f"Document {doc_id} already exists. Use different ID or remove first.")
            return False
        
        logger.info(f"Processing document: {file_path}")
        
        # Extract text based on file type
        from pdf_processing.pdf_processor import PDFProcessor
        processor = PDFProcessor(use_ocr=False)
        result = processor.process_document(file_path)
        
        if result['error']:
            logger.error(f"Error processing document: {result['error']}")
            return False
        
        text = result['text']
        
        # Create chunks
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
        
        logger.info(f"✓ Document added successfully: {doc_id}")
        return True
    
    def add_document_from_text(self, text: str, doc_id: str, metadata: Optional[Dict] = None,
                               chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Add document from text string
        
        Args:
            text: Document text
            doc_id: Document ID
            metadata: Optional metadata
            chunk_size: Size of chunks
            chunk_overlap: Overlap between chunks
        """
        logger.info(f"Adding text document: {doc_id}")
        
        # Create chunks
        chunks = self._create_smart_chunks(text, chunk_size, chunk_overlap)
        
        # Create embeddings
        embeddings = self.embedding_manager.create_embeddings(chunks)
        
        # Store document
        meta = metadata or {}
        meta.update({
            'num_chunks': len(chunks),
            'text_length': len(text)
        })
        
        self.document_store.add_document(doc_id, text, chunks, embeddings, meta)
        logger.info(f"✓ Text document added: {doc_id}")
        return True
    
    def _create_smart_chunks(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Create smart chunks with sentence boundary awareness
        """
        if not text:
            return []
        
        # Split into sentences first
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
        
        # Add overlapping chunks for continuity
        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            overlapped_chunks.append(chunk)
            
            # Add overlap with next chunk
            if i < len(chunks) - 1:
                overlap_text = chunk[-overlap:] + " " + chunks[i+1][:overlap]
                overlapped_chunks.append(overlap_text)
        
        return overlapped_chunks
    
    def chat(self, user_message: str, 
             top_k: int = 5,
             min_similarity: float = 0.3,
             use_reranking: bool = True,
             use_history: bool = True) -> str:
        """
        Chat with documents using advanced RAG
        
        Args:
            user_message: User's query
            top_k: Number of chunks to retrieve
            min_similarity: Minimum similarity threshold
            use_reranking: Use MMR reranking for diversity
            use_history: Use conversation history
            
        Returns:
            Assistant's response
        """
        if not self.document_store.list_documents():
            return "No documents loaded. Please add documents first using add_document_from_file() or add_document_from_text()."
        
        logger.info(f"Processing query: {user_message[:50]}...")
        
        # Create query embedding
        query_embedding = self.embedding_manager.create_query_embedding(user_message)
        
        # Find relevant chunks across all documents
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
            
            # Add document context
            for chunk, similarity, idx in similar_chunks:
                all_relevant_chunks.append({
                    'doc_id': doc_id,
                    'chunk': chunk,
                    'similarity': similarity,
                    'index': idx
                })
        
        # Sort by similarity
        all_relevant_chunks.sort(key=lambda x: x['similarity'], reverse=True)
        all_relevant_chunks = all_relevant_chunks[:top_k]
        
        # Rerank for diversity if enabled
        if use_reranking and len(all_relevant_chunks) > 1:
            chunk_tuples = [(c['chunk'], c['similarity'], c['index']) for c in all_relevant_chunks]
            reranked = self.embedding_manager.rerank_chunks(user_message, chunk_tuples)
            
            # Reconstruct with doc_id
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
        
        # Build messages
        messages = []
        if use_history and self.conversation_history:
            messages = self.conversation_history.copy()
        
        messages.append({"role": "user", "content": user_message})
        
        # Call Claude
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                system=system_message,
                messages=messages
            )
            
            assistant_message = response.content[0].text
            
            # Update conversation history
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
            
            logger.info("✓ Response generated")
            return assistant_message
        
        except Exception as e:
            logger.error(f"Error calling Claude API: {e}")
            return f"Error: {str(e)}"
    
    def _build_context(self, relevant_chunks: List[Dict]) -> str:
        """Build context from relevant chunks"""
        context = "# Relevant information from your documents:\n\n"
        
        for i, chunk_data in enumerate(relevant_chunks, 1):
            context += f"## Excerpt {i} (from {chunk_data['doc_id']}, "
            context += f"relevance: {chunk_data['similarity']:.2%})\n"
            context += f"{chunk_data['chunk']}\n\n"
        
        return context
    
    def _build_system_message(self, context: str) -> str:
        """Build system message for Claude"""
        
        doc_list = ", ".join(self.document_store.list_documents())
        
        system = f"""You are a research assistant helping users understand and analyze research documents.

You have access to the following documents: {doc_list}

Guidelines for your responses:
1. Base your answers on the provided context from the documents
2. When comparing documents, clearly identify which document you're referring to
3. If information isn't in the provided context, say so honestly
4. Be precise and academic in your responses
5. Cite specific documents when making claims
6. Extract key findings, methodologies, and conclusions
7. Provide structured comparisons when asked
8. Synthesize information across multiple documents when relevant

{context}

Remember: Always reference which document your information comes from, and provide accurate, context-based responses."""
        
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
