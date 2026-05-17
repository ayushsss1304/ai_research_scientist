"""
Plagiarism Checker
==================
Checks a piece of text for similarity against:
  1. Documents stored in the RAG document store (sentence-transformer embeddings + Jaccard n-grams)
  2. Paper abstracts/titles stored in the Neo4j Knowledge Graph

No external API needed — fully local using existing project infrastructure.
"""

import re
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


class PlagiarismChecker:
    """
    Local plagiarism detection using:
      - Cosine similarity  (semantic — catches paraphrasing)
      - Jaccard similarity (n-gram — catches copy-paste)
    """

    # Similarity thresholds
    HIGH_SIMILARITY   = 0.80   # very likely plagiarised
    MEDIUM_SIMILARITY = 0.50   # suspicious / paraphrased
    LOW_SIMILARITY    = 0.30   # some overlap

    def __init__(self, rag_document_store=None, embedding_manager=None):
        """
        Args:
            rag_document_store: Optional DocumentStore instance from OllamaRAGChatbot.
                                Pass this to enable checking against uploaded PDFs.
            embedding_manager:  Optional AdvancedEmbeddingManager instance.
                                Required when rag_document_store is provided.
        """
        self.document_store  = rag_document_store
        self.embedding_manager = embedding_manager

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def check(
        self,
        text: str,
        check_docs: bool = True,
        check_kg: bool = True,
        kg_manager=None,
        n_gram_size: int = 5,
        top_k_chunks: int = 10,
    ) -> Dict:
        """
        Run plagiarism check and return a structured report.

        Returns:
            {
              "overall_score": float,          # 0-100 %
              "verdict": str,                  # "Low" / "Medium" / "High"
              "total_matches": int,
              "matches": [
                  {
                    "source":  str,            # doc id or paper title
                    "type":    str,            # "Document" | "KG Paper"
                    "score":   float,          # 0-100 %
                    "method":  str,            # "Semantic" | "Exact Phrase"
                    "snippet": str,            # matched excerpt
                  }
              ]
            }
        """
        if not text or not text.strip():
            return self._empty_report("No text provided.")

        text = text.strip()
        all_matches: List[Dict] = []

        # 1. Check against RAG documents
        if check_docs and self.document_store and self.embedding_manager:
            doc_matches = self._check_rag_documents(text, n_gram_size, top_k_chunks)
            all_matches.extend(doc_matches)

        # 2. Check against Neo4j KG papers
        if check_kg and kg_manager:
            kg_matches = self._check_kg_papers(text, kg_manager, n_gram_size)
            all_matches.extend(kg_matches)

        # Deduplicate (same source may appear from both methods — keep highest)
        all_matches = self._deduplicate(all_matches)

        # Sort by score descending
        all_matches.sort(key=lambda m: m["score"], reverse=True)

        # Compute overall score — weighted average of top matches, capped at 100
        overall = self._compute_overall(all_matches)

        verdict = (
            "High"   if overall >= self.HIGH_SIMILARITY   * 100 else
            "Medium" if overall >= self.MEDIUM_SIMILARITY * 100 else
            "Low"
        )

        return {
            "overall_score":  round(overall, 1),
            "verdict":        verdict,
            "total_matches":  len(all_matches),
            "matches":        all_matches,
        }

    # ------------------------------------------------------------------ #
    #  Internal: RAG document store checking                               #
    # ------------------------------------------------------------------ #

    def _check_rag_documents(
        self,
        text: str,
        n_gram_size: int,
        top_k_chunks: int,
    ) -> List[Dict]:
        matches = []

        # Build n-gram fingerprint of input text
        input_shingles = self._shingle(text, n_gram_size)
        # Embedding of input text
        input_embedding = self.embedding_manager.create_query_embedding(text)

        doc_ids = self.document_store.list_documents()
        if not doc_ids:
            return []

        for doc_id in doc_ids:
            doc = self.document_store.get_document(doc_id)
            if not doc:
                continue

            # --- Semantic similarity (cosine) ---
            from sklearn.metrics.pairwise import cosine_similarity as cos_sim
            chunk_embeddings = doc.get("embeddings")
            chunks           = doc.get("chunks", [])

            if chunk_embeddings is not None and len(chunks) > 0:
                sims = cos_sim([input_embedding], chunk_embeddings)[0]
                top_idx = np.argsort(sims)[-top_k_chunks:][::-1]
                best_sem_score = float(sims[top_idx[0]]) if len(top_idx) > 0 else 0.0
                best_sem_chunk = chunks[top_idx[0]] if len(top_idx) > 0 else ""

                if best_sem_score >= self.LOW_SIMILARITY:
                    matches.append({
                        "source":  doc_id,
                        "type":    "Document",
                        "score":   round(best_sem_score * 100, 1),
                        "method":  "Semantic",
                        "snippet": best_sem_chunk[:300] + ("..." if len(best_sem_chunk) > 300 else ""),
                    })

            # --- Exact phrase (Jaccard n-gram) ---
            full_text = doc.get("text", "")
            if full_text:
                doc_shingles   = self._shingle(full_text, n_gram_size)
                jaccard_score  = self._jaccard_similarity(input_shingles, doc_shingles)

                if jaccard_score >= (self.LOW_SIMILARITY * 0.5):  # lower threshold for exact
                    # Find a representative matching snippet
                    snippet = self._find_matching_snippet(text, full_text)
                    matches.append({
                        "source":  doc_id,
                        "type":    "Document",
                        "score":   round(jaccard_score * 100, 1),
                        "method":  "Exact Phrase",
                        "snippet": snippet,
                    })

        return matches

    # ------------------------------------------------------------------ #
    #  Internal: Neo4j KG papers checking                                 #
    # ------------------------------------------------------------------ #

    def _check_kg_papers(
        self,
        text: str,
        kg_manager,
        n_gram_size: int,
    ) -> List[Dict]:
        matches = []

        input_shingles = self._shingle(text, n_gram_size)

        # Pull all papers from KG (up to 200)
        try:
            papers = kg_manager.get_all_papers(limit=200)
        except Exception as e:
            logger.warning(f"Could not fetch KG papers: {e}")
            return []

        for paper in papers:
            title    = paper.get("title", "") or ""
            abstract = paper.get("abstract", "") or ""

            # Combine title + abstract for comparison
            corpus = f"{title}. {abstract}".strip()
            if not corpus:
                continue

            # Jaccard on n-grams
            doc_shingles = self._shingle(corpus, n_gram_size)
            j_score      = self._jaccard_similarity(input_shingles, doc_shingles)

            # Also try the embedding path if embedding manager available
            sem_score = 0.0
            if self.embedding_manager:
                try:
                    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
                    q_emb    = self.embedding_manager.create_query_embedding(text)
                    d_emb    = self.embedding_manager.create_query_embedding(corpus[:512])
                    sem_score = float(cos_sim([q_emb], [d_emb])[0][0])
                except Exception:
                    pass

            best_score = max(j_score, sem_score)
            method     = "Exact Phrase" if j_score >= sem_score else "Semantic"

            if best_score >= self.LOW_SIMILARITY * 0.5:
                source_label = title[:80] + ("..." if len(title) > 80 else "")
                snippet      = abstract[:300] + ("..." if len(abstract) > 300 else "")
                matches.append({
                    "source":  source_label,
                    "type":    "KG Paper",
                    "score":   round(best_score * 100, 1),
                    "method":  method,
                    "snippet": snippet,
                })

        return matches

    # ------------------------------------------------------------------ #
    #  Utilities                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _shingle(text: str, n: int) -> set:
        """Build a set of character-level n-grams (shingles) from text."""
        text  = re.sub(r"\s+", " ", text.lower().strip())
        words = text.split()
        if len(words) < n:
            return set([" ".join(words)])
        return {" ".join(words[i:i+n]) for i in range(len(words) - n + 1)}

    @staticmethod
    def _jaccard_similarity(set_a: set, set_b: set) -> float:
        """Compute Jaccard index between two sets."""
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union        = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _find_matching_snippet(query: str, corpus: str, window: int = 300) -> str:
        """Find the most overlapping window in corpus relative to query words."""
        query_words  = set(re.findall(r"\w+", query.lower()))
        corpus_words = re.findall(r"\w+", corpus.lower())
        best_start, best_count = 0, 0
        step = max(1, window // 5)
        for start in range(0, max(1, len(corpus_words) - window), step):
            window_words = set(corpus_words[start:start + window])
            count        = len(query_words & window_words)
            if count > best_count:
                best_count = count
                best_start = start

        # Convert word index back to character slice (approximation)
        char_pos = len(" ".join(corpus_words[:best_start]))
        snippet  = corpus[char_pos:char_pos + window * 6]
        if len(snippet) > 300:
            snippet = snippet[:300] + "..."
        return snippet.strip()

    @staticmethod
    def _deduplicate(matches: List[Dict]) -> List[Dict]:
        """Keep the highest-score match per (source, method) pair."""
        best: Dict[str, Dict] = {}
        for m in matches:
            key = m["source"]
            if key not in best or m["score"] > best[key]["score"]:
                best[key] = m
        return list(best.values())

    @staticmethod
    def _compute_overall(matches: List[Dict]) -> float:
        """Weighted mean of top-5 matches (higher weight for top match)."""
        if not matches:
            return 0.0
        top = sorted([m["score"] for m in matches], reverse=True)[:5]
        weights = [5, 4, 3, 2, 1][:len(top)]
        return sum(s * w for s, w in zip(top, weights)) / sum(weights)

    @staticmethod
    def _empty_report(reason: str = "") -> Dict:
        return {
            "overall_score": 0.0,
            "verdict":       "Low",
            "total_matches": 0,
            "matches":       [],
            "note":          reason,
        }
