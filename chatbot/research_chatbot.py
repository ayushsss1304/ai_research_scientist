"""
Research Chatbot Module
Chat with multiple research papers using Claude and embeddings
"""

import anthropic
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages embeddings for semantic search"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize embedding model
        
        Args:
            model_name: Name of sentence transformer model
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded successfully")
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for list of texts
        
        Args:
            texts: List of text strings
        
        Returns:
            Numpy array of embeddings
        """
        logger.info(f"Creating embeddings for {len(texts)} texts")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def find_similar(self, query: str, embeddings: np.ndarray, 
                     texts: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find most similar texts to query
        
        Args:
            query: Query string
            embeddings: Pre-computed embeddings
            texts: Corresponding texts
            top_k: Number of results to return
        
        Returns:
            List of (text, similarity_score) tuples
        """
        query_embedding = self.model.encode([query])[0]
        
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = [(texts[i], similarities[i]) for i in top_indices]
        
        return results


class ResearchChatbot:
    """Chatbot for interacting with research papers"""
    
    def __init__(self, anthropic_api_key: str, 
                 embedding_model: str = 'all-MiniLM-L6-v2'):
        """
        Initialize research chatbot
        
        Args:
            anthropic_api_key: Anthropic API key
            embedding_model: Name of embedding model
        """
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.embedding_manager = EmbeddingManager(embedding_model)
        self.documents = {}  # doc_id -> document data
        self.conversation_history = []
    
    def add_document(self, file_path: str, doc_id: str, 
                     text: str, chunks: List[str]):
        """
        Add a document to the chatbot's knowledge base
        
        Args:
            file_path: Path to document
            doc_id: Unique identifier for document
            text: Full text of document
            chunks: List of text chunks
        """
        logger.info(f"Adding document: {doc_id}")
        
        # Create embeddings for chunks
        embeddings = self.embedding_manager.create_embeddings(chunks)
        
        self.documents[doc_id] = {
            'path': file_path,
            'text': text,
            'chunks': chunks,
            'embeddings': embeddings
        }
        
        logger.info(f"Document added: {doc_id} ({len(chunks)} chunks)")
    
    def remove_document(self, doc_id: str):
        """Remove a document from knowledge base"""
        if doc_id in self.documents:
            del self.documents[doc_id]
            logger.info(f"Removed document: {doc_id}")
    
    def list_documents(self) -> List[str]:
        """Get list of all document IDs"""
        return list(self.documents.keys())
    
    def find_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Find most relevant chunks across all documents
        
        Args:
            query: User query
            top_k: Number of chunks to return
        
        Returns:
            List of dictionaries with chunk info
        """
        all_results = []
        
        for doc_id, doc_data in self.documents.items():
            # Find similar chunks in this document
            similar = self.embedding_manager.find_similar(
                query,
                doc_data['embeddings'],
                doc_data['chunks'],
                top_k=top_k
            )
            
            for chunk, similarity in similar:
                all_results.append({
                    'doc_id': doc_id,
                    'chunk': chunk,
                    'similarity': similarity
                })
        
        # Sort by similarity across all documents
        all_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return all_results[:top_k]
    
    def chat(self, user_message: str, top_k: int = 5, 
             use_history: bool = True) -> str:
        """
        Chat with the research papers
        
        Args:
            user_message: User's message
            top_k: Number of relevant chunks to use
            use_history: Whether to use conversation history
        
        Returns:
            Assistant's response
        """
        if not self.documents:
            return "No documents have been uploaded yet. Please upload PDFs first."
        
        # Find relevant chunks
        relevant_chunks = self.find_relevant_chunks(user_message, top_k)
        
        # Build context
        context = self._build_context(relevant_chunks)
        
        # Build system message
        system_message = self._build_system_message(context)
        
        # Build messages
        messages = []
        
        if use_history:
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
            
            return assistant_message
        
        except Exception as e:
            logger.error(f"Error calling Claude API: {e}")
            return f"Error: {str(e)}"
    
    def _build_context(self, relevant_chunks: List[Dict]) -> str:
        """Build context string from relevant chunks"""
        context = "# Relevant excerpts from research papers:\n\n"
        
        for i, chunk_info in enumerate(relevant_chunks, 1):
            context += f"## Excerpt {i} (from {chunk_info['doc_id']}, "
            context += f"relevance: {chunk_info['similarity']:.2f})\n"
            context += f"{chunk_info['chunk']}\n\n"
        
        return context
    
    def _build_system_message(self, context: str) -> str:
        """Build system message for Claude"""
        
        doc_list = ", ".join(self.documents.keys())
        
        system = f"""You are a research assistant helping users understand and analyze research papers.

You have access to the following documents: {doc_list}

Guidelines for your responses:
1. Base your answers on the provided context from the papers
2. When comparing papers, clearly identify which paper you're referring to
3. If information isn't in the provided context, say so
4. Be precise and academic in your responses
5. Cite specific papers when making claims
6. If asked to compare, provide structured comparisons
7. Extract key findings, methodologies, and conclusions

{context}

Remember to always reference which document your information comes from."""
        
        return system
    
    def compare_papers(self, aspect: str = "methodology") -> str:
        """
        Compare all papers on a specific aspect
        
        Args:
            aspect: Aspect to compare (methodology, results, etc.)
        
        Returns:
            Comparison summary
        """
        if len(self.documents) < 2:
            return "Need at least 2 documents to compare."
        
        query = f"Compare the {aspect} across these papers"
        return self.chat(query)
    
    def summarize_document(self, doc_id: str) -> str:
        """
        Summarize a specific document
        
        Args:
            doc_id: Document identifier
        
        Returns:
            Summary
        """
        if doc_id not in self.documents:
            return f"Document '{doc_id}' not found."
        
        # Use first few chunks for summary
        doc_data = self.documents[doc_id]
        context = "\n\n".join(doc_data['chunks'][:5])
        
        system = f"""Summarize the following research paper ({doc_id}):

{context}

Provide a concise summary covering:
1. Main research question
2. Methodology
3. Key findings
4. Conclusions"""
        
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                system=system,
                messages=[{"role": "user", "content": "Please provide the summary."}]
            )
            
            return response.content[0].text
        
        except Exception as e:
            return f"Error: {str(e)}"
    
    def extract_key_points(self, doc_id: str, num_points: int = 5) -> List[str]:
        """
        Extract key points from a document
        
        Args:
            doc_id: Document identifier
            num_points: Number of key points to extract
        
        Returns:
            List of key points
        """
        if doc_id not in self.documents:
            return [f"Document '{doc_id}' not found."]
        
        doc_data = self.documents[doc_id]
        context = "\n\n".join(doc_data['chunks'][:5])
        
        system = f"""Extract {num_points} key points from this research paper:

{context}"""
        
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                system=system,
                messages=[{
                    "role": "user",
                    "content": f"List the {num_points} most important key points."
                }]
            )
            
            # Parse response into list
            text = response.content[0].text
            points = [line.strip() for line in text.split('\n') if line.strip()]
            
            return points
        
        except Exception as e:
            return [f"Error: {str(e)}"]
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_statistics(self) -> Dict:
        """Get statistics about loaded documents"""
        total_chunks = sum(len(doc['chunks']) for doc in self.documents.values())
        total_chars = sum(len(doc['text']) for doc in self.documents.values())
        
        return {
            'num_documents': len(self.documents),
            'total_chunks': total_chunks,
            'total_characters': total_chars,
            'document_ids': list(self.documents.keys())
        }