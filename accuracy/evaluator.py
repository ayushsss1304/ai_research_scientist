"""
Chatbot Accuracy Evaluator
===========================
Evaluates how accurately the RAG chatbot answers questions by comparing
actual responses against human-written reference (expected) answers using:

  1. BLEU-4  — precision of n-gram overlaps (0-1)
  2. ROUGE-L  — longest common subsequence recall (0-1)
  3. Semantic Similarity — cosine similarity of sentence embeddings (0-1)
  4. Composite Score — weighted combination of all three (0-100 %)

No external API needed — all computation is local.
"""

import re
import math
import logging
from typing import List, Dict, Optional, Tuple
from collections import Counter

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  BLEU Score (up to 4-gram)
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def bleu_score(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """
    Compute corpus-level BLEU-N score between reference and hypothesis.

    Formula:
        BLEU = BP · exp( Σ wₙ · log pₙ )
        where:
            BP   = brevity penalty  = min(1, exp(1 - |ref|/|hyp|))
            pₙ   = modified n-gram precision for order n
            wₙ   = 1/max_n  (uniform weights)
    """
    ref_tokens  = _tokenize(reference)
    hyp_tokens  = _tokenize(hypothesis)

    if not hyp_tokens:
        return 0.0

    # Brevity Penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1)))

    log_sum = 0.0
    for n in range(1, max_n + 1):
        ref_ngrams = _ngrams(ref_tokens, n)
        hyp_ngrams = _ngrams(hyp_tokens, n)

        if not hyp_ngrams:
            return 0.0

        # Clipped count
        clipped = sum(min(count, ref_ngrams[gram]) for gram, count in hyp_ngrams.items())
        total   = sum(hyp_ngrams.values())

        pn = clipped / total if total > 0 else 0.0

        if pn == 0.0:
            return 0.0
        log_sum += (1.0 / max_n) * math.log(pn)

    return round(bp * math.exp(log_sum), 4)


# ─────────────────────────────────────────────────────────────────────────────
#  ROUGE-L (Longest Common Subsequence)
# ─────────────────────────────────────────────────────────────────────────────

def _lcs_length(a: List[str], b: List[str]) -> int:
    """Dynamic programming LCS length."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def rouge_l(reference: str, hypothesis: str, beta: float = 1.2) -> float:
    """
    Compute ROUGE-L F1 score.

    Formula:
        Rₗₒₗ = LCS(ref, hyp) / len(ref)
        Pₗₒₗ = LCS(ref, hyp) / len(hyp)
        F1   = (1 + β²) · Rₗₒₗ · Pₗₒₗ / (β² · Pₗₒₗ + Rₗₒₗ)
    """
    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)

    if not ref_tokens or not hyp_tokens:
        return 0.0

    lcs = _lcs_length(ref_tokens, hyp_tokens)
    recall    = lcs / len(ref_tokens)
    precision = lcs / len(hyp_tokens)

    if recall == 0 or precision == 0:
        return 0.0

    f1 = ((1 + beta**2) * recall * precision) / (beta**2 * precision + recall)
    return round(f1, 4)


# ─────────────────────────────────────────────────────────────────────────────
#  Semantic Similarity (cosine of sentence embeddings)
# ─────────────────────────────────────────────────────────────────────────────

def cosine_sim(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Cosine similarity:
        cos(θ) = (A · B) / (‖A‖ · ‖B‖)
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


# ─────────────────────────────────────────────────────────────────────────────
#  Main Evaluator Class
# ─────────────────────────────────────────────────────────────────────────────

class ChatbotAccuracyEvaluator:
    """
    Evaluates chatbot accuracy against reference answers.

    Weights for composite score:
        semantic  : 50 %   (most important — captures meaning)
        rouge_l   : 30 %   (structural coverage)
        bleu      : 20 %   (n-gram precision)
    """

    WEIGHTS = {"semantic": 0.50, "rouge_l": 0.30, "bleu": 0.20}

    # Accuracy tiers
    EXCELLENT = 80
    GOOD      = 60
    FAIR      = 40

    def __init__(self, embedding_manager=None):
        """
        Args:
            embedding_manager: Optional AdvancedEmbeddingManager instance
                               from the RAG pipeline. If None, semantic
                               similarity will be set to 0.
        """
        self.embedding_manager = embedding_manager

    # ── Public API ──────────────────────────────────────────────────────────

    def evaluate(
        self,
        test_cases: List[Dict],
        chatbot=None,
    ) -> Dict:
        """
        Run evaluation over a list of test cases.

        Args:
            test_cases: [
                {"question": str, "expected": str, "actual": str (optional)},
                ...
            ]
            chatbot: Optional OllamaRAGChatbot instance. If provided and
                     "actual" is missing, the chatbot will answer the question.

        Returns:
            {
              "summary": {
                "total": int,
                "avg_bleu": float,
                "avg_rouge_l": float,
                "avg_semantic": float,
                "avg_composite": float,
                "accuracy_tier": str
              },
              "results": [
                {
                  "question": str,
                  "expected": str,
                  "actual": str,
                  "bleu": float (0-100),
                  "rouge_l": float (0-100),
                  "semantic": float (0-100),
                  "composite": float (0-100),
                  "tier": str,
                  "error": str or None
                }
              ]
            }
        """
        results = []

        for tc in test_cases:
            question = tc.get("question", "").strip()
            expected = tc.get("expected", "").strip()
            actual   = tc.get("actual", "").strip()
            error    = None

            if not question or not expected:
                continue

            # Get chatbot answer if not supplied
            if not actual and chatbot:
                try:
                    actual = chatbot.chat(
                        question,
                        top_k=5,
                        use_reranking=True,
                        temperature=0.3,   # low temp → deterministic
                    )
                except Exception as e:
                    actual = ""
                    error  = str(e)
                    logger.warning(f"Chatbot error for '{question[:40]}': {e}")

            if not actual:
                results.append({
                    "question": question,
                    "expected": expected,
                    "actual":   actual or "",
                    "bleu":     0.0, "rouge_l": 0.0,
                    "semantic": 0.0, "composite": 0.0,
                    "tier":  "N/A",
                    "error": error or "No actual answer provided",
                })
                continue

            # ── Compute metrics ────────────────────────────────────────────
            bleu   = bleu_score(expected, actual)
            r_l    = rouge_l(expected, actual)
            sem    = self._semantic_similarity(expected, actual)

            composite = (
                self.WEIGHTS["bleu"]     * bleu +
                self.WEIGHTS["rouge_l"]  * r_l  +
                self.WEIGHTS["semantic"] * sem
            )

            # Scale to 0-100 %
            bleu_pct = round(bleu * 100, 1)
            r_l_pct  = round(r_l  * 100, 1)
            sem_pct  = round(sem  * 100, 1)
            comp_pct = round(composite * 100, 1)

            tier = (
                "Excellent" if comp_pct >= self.EXCELLENT else
                "Good"      if comp_pct >= self.GOOD      else
                "Fair"      if comp_pct >= self.FAIR       else
                "Poor"
            )

            results.append({
                "question":  question,
                "expected":  expected,
                "actual":    actual,
                "bleu":      bleu_pct,
                "rouge_l":   r_l_pct,
                "semantic":  sem_pct,
                "composite": comp_pct,
                "tier":      tier,
                "error":     error,
            })

        # ── Summary ────────────────────────────────────────────────────────
        if results:
            valid = [r for r in results if r["tier"] != "N/A"]
            n = len(valid) or 1
            avg_bleu      = round(sum(r["bleu"]      for r in valid) / n, 1)
            avg_rouge_l   = round(sum(r["rouge_l"]   for r in valid) / n, 1)
            avg_semantic  = round(sum(r["semantic"]  for r in valid) / n, 1)
            avg_composite = round(sum(r["composite"] for r in valid) / n, 1)
        else:
            avg_bleu = avg_rouge_l = avg_semantic = avg_composite = 0.0

        overall_tier = (
            "Excellent" if avg_composite >= self.EXCELLENT else
            "Good"      if avg_composite >= self.GOOD      else
            "Fair"      if avg_composite >= self.FAIR       else
            "Poor"
        )

        return {
            "summary": {
                "total":         len(results),
                "avg_bleu":      avg_bleu,
                "avg_rouge_l":   avg_rouge_l,
                "avg_semantic":  avg_semantic,
                "avg_composite": avg_composite,
                "accuracy_tier": overall_tier,
            },
            "results": results,
        }

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _semantic_similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity of sentence embeddings (0-1)."""
        if not self.embedding_manager:
            return 0.0
        try:
            emb_a = self.embedding_manager.create_query_embedding(text_a[:512])
            emb_b = self.embedding_manager.create_query_embedding(text_b[:512])
            sim   = cosine_sim(emb_a, emb_b)
            # Sentence embeddings range [-1, 1]; normalise to [0, 1]
            return max(0.0, (sim + 1.0) / 2.0)
        except Exception as e:
            logger.warning(f"Semantic similarity error: {e}")
            return 0.0
