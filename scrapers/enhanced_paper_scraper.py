"""
Enhanced Paper Scraper with ALL sources and advanced deduplication
Supports: ArXiv, Semantic Scholar, IEEE, PubMed, CORE, OpenAlex, CrossRef, Google Scholar
"""

from typing import List, Dict, Optional, Set
import logging
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import time

from .arxiv_scraper import ArxivScraper
from .semantic_scholar_scraper import SemanticScholarScraper
from .ieee_scraper import IEEEScraper
from .scopus_scraper import ScopusScraper
from .additional_sources import (
    PubMedScraper, COREScrap, OpenAlexScraper, 
    CrossRefScraper, GoogleScholarScraper
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedPaperScraper:
    """
    Enhanced unified scraper for multiple research paper sources
    with advanced deduplication and abstract inclusion
    """
    
    def __init__(self, 
                 ieee_api_key: Optional[str] = None,
                 semantic_scholar_api_key: Optional[str] = None,
                 elsevier_api_key: Optional[str] = None,
                 pubmed_api_key: Optional[str] = None,
                 core_api_key: Optional[str] = None,
                 serpapi_key: Optional[str] = None,
                 email: Optional[str] = None,
                 cache_dir: str = 'data/cache/search',
                 cache_ttl: int = 3600,
                 max_workers: int = 5):
        """
        Initialize all scrapers
        
        Args:
            ieee_api_key: IEEE Xplore API key
            semantic_scholar_api_key: Semantic Scholar API key
            pubmed_api_key: PubMed API key (optional)
            core_api_key: CORE API key
            serpapi_key: SerpAPI key for Google Scholar
            email: Your email for polite API access (OpenAlex, CrossRef)
        """
        
        # Initialize all scrapers
        self.scrapers = {}
        self.cache_dir = cache_dir
        self.cache_ttl = cache_ttl
        self.max_workers = max_workers
        os.makedirs(self.cache_dir, exist_ok=True)

        # Verified academic source
        if elsevier_api_key and elsevier_api_key not in ['', 'your_key', 'your_elsevier_api_key_here']:
            try:
                self.scrapers['scopus'] = ScopusScraper(elsevier_api_key)
                logger.info("✓ Scopus initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Scopus: {e}")
        
        # Free sources (no API key required)
        self.scrapers['arxiv'] = ArxivScraper()
        logger.info("✓ ArXiv initialized")
        
        self.scrapers['semantic_scholar'] = SemanticScholarScraper(semantic_scholar_api_key)
        logger.info("✓ Semantic Scholar initialized")
        
        self.scrapers['openalex'] = OpenAlexScraper(email)
        logger.info("✓ OpenAlex initialized")
        
        self.scrapers['crossref'] = CrossRefScraper(email)
        logger.info("✓ CrossRef initialized")
        
        self.scrapers['pubmed'] = PubMedScraper(pubmed_api_key)
        logger.info("✓ PubMed initialized")
        
        # Sources requiring API keys
        if ieee_api_key and ieee_api_key not in ['', 'your_key']:
            try:
                self.scrapers['ieee'] = IEEEScraper(ieee_api_key)
                logger.info("✓ IEEE Xplore initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize IEEE: {e}")
        
        if core_api_key and core_api_key not in ['', 'your_key']:
            self.scrapers['core'] = COREScrap(core_api_key)
            logger.info("✓ CORE initialized")
        
        if serpapi_key and serpapi_key not in ['', 'your_key']:
            self.scrapers['google_scholar'] = GoogleScholarScraper(serpapi_key)
            logger.info("✓ Google Scholar initialized")
        
        logger.info(f"\n📚 Total sources available: {len(self.scrapers)}")

    # Common English stopwords to ignore in keyword matching
    _STOPWORDS = {
        'a', 'an', 'the', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for',
        'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'it', 'its',
        'this', 'that', 'these', 'those', 'be', 'been', 'being', 'have', 'has',
        'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
        'might', 'can', 'about', 'into', 'through', 'during', 'using', 'via',
        'than', 'more', 'most', 'also', 'such', 'their', 'our', 'we', 'i',
    }

    def _normalize_str(self, s: str) -> str:
        """Lowercase, strip punctuation."""
        import re
        return re.sub(r'[^\w\s]', '', (s or '').lower()).strip()

    def _keywords(self, text: str) -> set:
        """Return meaningful keywords after removing stopwords."""
        words = self._normalize_str(text).split()
        return {w for w in words if w and w not in self._STOPWORDS}

    def compute_relevance_score(self, paper: Dict, query: str) -> float:
        """
        Compute a 0-100 relevance score for a paper against a query.
        Factors:
          - Exact title match          → 100 immediately
          - Near-exact / phrase match  → up to 40 pts (title) + bonus
          - Abstract keyword match     → up to 20 pts
          - Source prestige            → up to 10 pts
          - Recency                    → up to 10 pts
          - Citation count (log)       → up to 10 pts
        """
        import math
        from datetime import date

        title = paper.get('title') or ''
        abstract_text = (paper.get('abstract') or '').lower()

        norm_query = self._normalize_str(query)
        norm_title = self._normalize_str(title)

        # ── Exact title match → instant 100 ──────────────────────────────
        if norm_query == norm_title:
            return 100.0

        score = 0.0

        # ── 1. Title match (up to 40 pts) ─────────────────────────────────
        # a) Phrase containment bonus (one contains the other as a substring)
        if norm_query in norm_title or norm_title in norm_query:
            score += 40.0
        else:
            # b) Stopword-filtered keyword overlap
            query_kw = self._keywords(query)
            title_kw = self._keywords(title)
            if query_kw and title_kw:
                overlap = len(query_kw & title_kw) / len(query_kw)
                score += overlap * 40

        # ── 2. Abstract keyword match (up to 20 pts) ──────────────────────
        query_kw = self._keywords(query)
        if query_kw and abstract_text:
            matched = sum(1 for t in query_kw if t in abstract_text)
            score += (matched / len(query_kw)) * 20

        # ── 3. Source / publication prestige (up to 10 pts) ───────────────
        source_weights = {
            'scopus': 10,
            'ieee': 10, 'ieee xplore': 10,
            'nature': 10, 'science': 10,
            'acm': 8,
            'semantic scholar': 6, 'pubmed': 6,
            'arxiv': 4, 'openalex': 4,
            'crossref': 3, 'core': 3,
            'google scholar': 2,
        }
        src = (paper.get('source') or '').lower()
        journal = (paper.get('journal') or '').lower()
        prestige = 0
        for key, val in source_weights.items():
            if key in src or key in journal:
                prestige = max(prestige, val)
        score += min(prestige, 10)

        # ── 4. Recency (up to 10 pts) ─────────────────────────────────────
        current_year = date.today().year
        paper_year = paper.get('year') or 0
        if paper_year > 0:
            age = max(0, current_year - paper_year)
            score += max(0, 10 - age)

        # ── 5. Citation count — log-scaled up to 10 pts ───────────────────
        citations = paper.get('citations') or 0
        if citations > 0:
            score += min(10, math.log10(citations + 1) * 3)

        return round(min(score, 100), 1)

    def search_all(self, 
                   query: str, 
                   max_results_per_source: int = 50,
                   start_year: Optional[int] = None,
                   end_year: Optional[int] = None,
                   min_citations: Optional[int] = None,
                   journal_filter: Optional[str] = None,
                   sources: Optional[List[str]] = None,
                   include_abstract: bool = True) -> List[Dict]:
        """
        Search all available sources with advanced deduplication
        
        Args:
            query: Search query
            max_results_per_source: Max results per source
            start_year: Filter start year
            end_year: Filter end year
            min_citations: Minimum citation count
            journal_filter: Journal name filter
            sources: List of sources to search (default: all)
            include_abstract: Include abstract in results
            
        Returns:
            List of unique papers sorted by relevance score
        """
        
        if sources is None:
            sources = list(self.scrapers.keys())
        sources = [source for source in sources if source in self.scrapers]
        cache_key = self._cache_key(
            query=query,
            max_results_per_source=max_results_per_source,
            start_year=start_year,
            end_year=end_year,
            min_citations=min_citations,
            journal_filter=journal_filter,
            sources=sources,
            include_abstract=include_abstract,
        )
        cached = self._get_cached_search(cache_key)
        if cached is not None:
            logger.info(f"Returning cached search results for: {query}")
            return cached
        
        logger.info(f"\n{'='*80}")
        logger.info(f"SEARCHING: {query}")
        logger.info(f"Sources: {', '.join(sources)}")
        logger.info(f"{'='*80}\n")
        
        all_papers = []
        source_counts = {}
        
        # Search each source in parallel. Most time here is network I/O.
        workers = max(1, min(self.max_workers, len(sources)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    self._search_single_source,
                    source_name,
                    query,
                    max_results_per_source,
                    start_year,
                    end_year,
                    min_citations,
                ): source_name
                for source_name in sources
            }

            for future in as_completed(futures):
                source_name = futures[future]
                try:
                    papers = future.result()
                    all_papers.extend(papers)
                    source_counts[source_name] = len(papers)
                    logger.info(f"  Found {len(papers)} papers from {source_name}")
                except Exception as e:
                    logger.error(f"  Error searching {source_name}: {e}")
                    source_counts[source_name] = 0

        if False:
            try:
                logger.info(f"🔍 Searching {source_name}...")
                scraper = self.scrapers[source_name]
                
                papers = scraper.search(
                    query=query,
                    max_results=max_results_per_source,
                    start_year=start_year,
                    end_year=end_year,
                    min_citations=min_citations
                )
                
                all_papers.extend(papers)
                source_counts[source_name] = len(papers)
                logger.info(f"  ✓ Found {len(papers)} papers from {source_name}")
            
            except Exception as e:
                logger.error(f"  ✗ Error searching {source_name}: {e}")
                source_counts[source_name] = 0
        
        logger.info(f"\n{'='*80}")
        logger.info(f"SEARCH RESULTS SUMMARY")
        logger.info(f"{'='*80}")
        for source, count in source_counts.items():
            logger.info(f"  {source:20s}: {count:4d} papers")
        logger.info(f"  {'Total (before dedup)':20s}: {len(all_papers):4d} papers")
        
        # Apply journal filter
        if journal_filter:
            original_count = len(all_papers)
            all_papers = [
                p for p in all_papers
                if journal_filter.lower() in p.get('journal', '').lower()
            ]
            logger.info(f"  Journal filter: {original_count} -> {len(all_papers)} papers")
        
        # Remove duplicates with advanced algorithm
        unique_papers = self.advanced_deduplication(all_papers)
        logger.info(f"  {'After deduplication':20s}: {len(unique_papers):4d} papers")

        # If Scopus is available, verify DOI-bearing non-Scopus results for free.
        if 'scopus' in self.scrapers:
            self._verify_with_scopus(unique_papers, limit=10)
        
        # Compute relevance score and sort by it (descending)
        for paper in unique_papers:
            paper['relevance_score'] = self.compute_relevance_score(paper, query)
        unique_papers.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Optionally remove abstracts to save space
        if not include_abstract:
            for paper in unique_papers:
                paper.pop('abstract', None)
        
        logger.info(f"{'='*80}\n")
        self._set_cached_search(cache_key, unique_papers)
        return unique_papers

    def _search_single_source(
        self,
        source_name: str,
        query: str,
        max_results: int,
        start_year: Optional[int],
        end_year: Optional[int],
        min_citations: Optional[int],
    ) -> List[Dict]:
        logger.info(f"Searching {source_name}...")
        scraper = self.scrapers[source_name]
        return scraper.search(
            query=query,
            max_results=max_results,
            start_year=start_year,
            end_year=end_year,
            min_citations=min_citations
        )

    def _cache_key(self, **kwargs) -> str:
        payload = json.dumps(kwargs, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()

    def _cache_path(self, cache_key: str) -> str:
        return os.path.join(self.cache_dir, f'{cache_key}.json')

    def _get_cached_search(self, cache_key: str):
        path = self._cache_path(cache_key)
        if not os.path.exists(path):
            return None
        if time.time() - os.path.getmtime(path) > self.cache_ttl:
            return None
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                return json.load(handle)
        except Exception as exc:
            logger.debug(f"Could not read search cache {path}: {exc}")
            return None

    def _set_cached_search(self, cache_key: str, papers: List[Dict]) -> None:
        path = self._cache_path(cache_key)
        try:
            with open(path, 'w', encoding='utf-8') as handle:
                json.dump(papers, handle, ensure_ascii=False)
        except Exception as exc:
            logger.debug(f"Could not write search cache {path}: {exc}")
    
    def advanced_deduplication(self, papers: List[Dict]) -> List[Dict]:
        """
        Advanced deduplication using multiple strategies
        
        Strategies:
        1. Exact title match
        2. Fuzzy title matching (>90% similarity)
        3. DOI matching
        4. ArXiv ID matching
        5. Choose best source for duplicates
        """
        
        if not papers:
            return []
        
        unique_papers = []
        seen_titles = set()
        seen_dois = set()
        seen_arxiv_ids = set()
        doi_map = {}
        arxiv_map = {}
        title_map = {}  # For fuzzy matching
        
        for paper in papers:
            is_duplicate = False
            
            # Strategy 1: DOI matching (most reliable)
            doi = paper.get('doi', '').strip().lower()
            if doi and doi in seen_dois:
                is_duplicate = True
                existing_paper = doi_map.get(doi)
                if existing_paper:
                    self._merge_paper_data(existing_paper, paper)
                logger.debug(f"Duplicate (DOI): {paper['title'][:50]}")
            elif doi:
                seen_dois.add(doi)
                doi_map[doi] = paper
            
            # Strategy 2: ArXiv ID matching
            arxiv_id = paper.get('arxiv_id', '').strip()
            if not is_duplicate and arxiv_id and arxiv_id in seen_arxiv_ids:
                is_duplicate = True
                existing_paper = arxiv_map.get(arxiv_id)
                if existing_paper:
                    self._merge_paper_data(existing_paper, paper)
                logger.debug(f"Duplicate (ArXiv ID): {paper['title'][:50]}")
            elif arxiv_id:
                seen_arxiv_ids.add(arxiv_id)
                arxiv_map[arxiv_id] = paper
            
            # Strategy 3: Exact title matching
            title = paper.get('title', '').strip().lower()
            title_normalized = self._normalize_title(title)
            
            if not is_duplicate and title_normalized in seen_titles:
                is_duplicate = True
                existing_paper = title_map.get(title_normalized)
                if existing_paper:
                    self._merge_paper_data(existing_paper, paper)
                logger.debug(f"Duplicate (Exact title): {paper['title'][:50]}")
            
            # Strategy 4: Fuzzy title matching
            if not is_duplicate and not doi and not arxiv_id:
                for existing_title, existing_paper in title_map.items():
                    similarity = self._title_similarity(title_normalized, existing_title)
                    if similarity > 0.90:  # 90% similarity threshold
                        is_duplicate = True
                        logger.debug(f"Duplicate (Fuzzy {similarity:.2%}): {paper['title'][:50]}")
                        
                        # Keep paper with more information
                        if self._paper_quality_score(paper) > self._paper_quality_score(existing_paper):
                            self._merge_paper_data(paper, existing_paper)
                            # Replace with better version
                            unique_papers.remove(existing_paper)
                            unique_papers.append(paper)
                            title_map[title_normalized] = paper
                        else:
                            self._merge_paper_data(existing_paper, paper)
                        
                        break
            
            # Add if not duplicate
            if not is_duplicate:
                unique_papers.append(paper)
                seen_titles.add(title_normalized)
                title_map[title_normalized] = paper
        
        removed = len(papers) - len(unique_papers)
        if removed > 0:
            logger.info(f"  Removed {removed} duplicates")
        
        return unique_papers

    def _verify_with_scopus(self, papers: List[Dict], limit: int = 10) -> None:
        """Mark DOI-bearing papers as Scopus verified when Scopus finds them."""
        scopus = self.scrapers.get('scopus')
        checked = 0

        for paper in papers:
            if checked >= limit:
                break
            if paper.get('scopus_verified') or not paper.get('doi'):
                continue

            checked += 1
            try:
                scopus_match = scopus.search_by_doi(paper.get('doi', ''))
                if scopus_match:
                    self._merge_paper_data(paper, scopus_match)
                    paper['scopus_verified'] = True
            except Exception as exc:
                logger.debug(f"Scopus verification skipped for {paper.get('doi')}: {exc}")

    def _merge_paper_data(self, target: Dict, incoming: Dict) -> None:
        """Fill missing metadata on target from a duplicate incoming record."""
        scalar_fields = [
            'abstract', 'doi', 'pdf_url', 'url', 'journal', 'published_date',
            'paper_id', 'eid', 'scopus_id', 'document_type', 'document_type_code',
            'issn', 'isbn'
        ]
        for field in scalar_fields:
            if not target.get(field) and incoming.get(field):
                target[field] = incoming[field]

        target['citations'] = max(target.get('citations') or 0, incoming.get('citations') or 0)
        target['scopus_verified'] = bool(target.get('scopus_verified') or incoming.get('scopus_verified'))
        target['open_access'] = bool(target.get('open_access') or incoming.get('open_access'))

        list_fields = [
            'authors', 'fields', 'keywords', 'categories', 'subject_areas',
            'affiliations', 'funding_sponsors', 'publication_types'
        ]
        for field in list_fields:
            merged = self._merge_list_values(target.get(field), incoming.get(field))
            if merged:
                target[field] = merged

        if incoming.get('source_metrics') and not target.get('source_metrics'):
            target['source_metrics'] = incoming['source_metrics']

        sources = self._merge_list_values(target.get('merged_sources'), [target.get('source'), incoming.get('source')])
        if sources:
            target['merged_sources'] = sources

    def _merge_list_values(self, current, incoming):
        """Merge lists while preserving order and supporting dict items."""
        if current is None:
            current = []
        if incoming is None:
            incoming = []
        if not isinstance(current, list):
            current = [current]
        if not isinstance(incoming, list):
            incoming = [incoming]

        merged = []
        seen = set()
        for item in current + incoming:
            if not item:
                continue
            marker = tuple(sorted(item.items())) if isinstance(item, dict) else str(item).lower()
            if marker in seen:
                continue
            seen.add(marker)
            merged.append(item)
        return merged
    
    def _normalize_title(self, title: str) -> str:
        """Normalize title for comparison"""
        import re
        
        title = title.lower().strip()
        # Remove special characters
        title = re.sub(r'[^\w\s]', '', title)
        # Remove extra whitespace
        title = re.sub(r'\s+', ' ', title)
        
        return title
    
    def _title_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity between two titles"""
        return SequenceMatcher(None, title1, title2).ratio()
    
    def _paper_quality_score(self, paper: Dict) -> int:
        """
        Calculate quality score for a paper
        Used to choose best version of duplicate papers
        """
        score = 0
        
        # Has abstract
        if paper.get('abstract'):
            score += 10
        
        # Has DOI
        if paper.get('doi'):
            score += 5
        
        # Has PDF URL
        if paper.get('pdf_url'):
            score += 3
        
        # Has citation count
        if paper.get('citations', 0) > 0:
            score += 2
        
        # Has authors
        if paper.get('authors'):
            score += 1
        
        # Prefer certain sources
        source_priority = {
            'Scopus': 7,
            'Semantic Scholar': 5,
            'IEEE Xplore': 4,
            'OpenAlex': 3,
            'ArXiv': 3,
            'PubMed': 3,
            'CrossRef': 2,
            'CORE': 2,
            'Google Scholar': 1
        }
        score += source_priority.get(paper.get('source', ''), 0)
        
        return score
    
    def get_available_sources(self) -> List[str]:
        """Get list of available sources"""
        return list(self.scrapers.keys())
    
    def get_source_stats(self) -> Dict[str, Dict]:
        """Get statistics about available sources"""
        stats = {}
        
        for source_name, scraper in self.scrapers.items():
            stats[source_name] = {
                'name': source_name,
                'type': scraper.__class__.__name__,
                'available': True
            }
        
        return stats
    
    def search_by_doi(self, doi: str) -> Optional[Dict]:
        """Search for a paper by DOI across all sources"""
        logger.info(f"Searching for DOI: {doi}")
        
        # Try Semantic Scholar first (fastest)
        if 'semantic_scholar' in self.scrapers:
            try:
                response = self.scrapers['semantic_scholar'].search(doi, max_results=1)
                if response:
                    return response[0]
            except:
                pass
        
        # Try other sources
        for scraper in self.scrapers.values():
            try:
                results = scraper.search(doi, max_results=1)
                if results:
                    return results[0]
            except:
                continue
        
        return None
