"""
Enhanced Paper Scraper with ALL sources and advanced deduplication
Supports: ArXiv, Semantic Scholar, IEEE, PubMed, CORE, OpenAlex, CrossRef, Google Scholar
"""

from typing import List, Dict, Optional, Set
import logging
from difflib import SequenceMatcher

from .arxiv_scraper import ArxivScraper
from .semantic_scholar_scraper import SemanticScholarScraper
from .ieee_scraper import IEEEScraper
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
                 pubmed_api_key: Optional[str] = None,
                 core_api_key: Optional[str] = None,
                 serpapi_key: Optional[str] = None,
                 email: Optional[str] = None):
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
            List of unique papers with all metadata
        """
        
        if sources is None:
            sources = list(self.scrapers.keys())
        
        logger.info(f"\n{'='*80}")
        logger.info(f"SEARCHING: {query}")
        logger.info(f"Sources: {', '.join(sources)}")
        logger.info(f"{'='*80}\n")
        
        all_papers = []
        source_counts = {}
        
        # Search each source
        for source_name in sources:
            if source_name not in self.scrapers:
                logger.warning(f"Source '{source_name}' not available")
                continue
            
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
        
        # Sort by citations
        unique_papers.sort(key=lambda x: x.get('citations', 0), reverse=True)
        
        # Optionally remove abstracts to save space
        if not include_abstract:
            for paper in unique_papers:
                paper.pop('abstract', None)
        
        logger.info(f"{'='*80}\n")
        
        return unique_papers
    
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
        title_map = {}  # For fuzzy matching
        
        for paper in papers:
            is_duplicate = False
            
            # Strategy 1: DOI matching (most reliable)
            doi = paper.get('doi', '').strip().lower()
            if doi and doi in seen_dois:
                is_duplicate = True
                logger.debug(f"Duplicate (DOI): {paper['title'][:50]}")
            elif doi:
                seen_dois.add(doi)
            
            # Strategy 2: ArXiv ID matching
            arxiv_id = paper.get('arxiv_id', '').strip()
            if not is_duplicate and arxiv_id and arxiv_id in seen_arxiv_ids:
                is_duplicate = True
                logger.debug(f"Duplicate (ArXiv ID): {paper['title'][:50]}")
            elif arxiv_id:
                seen_arxiv_ids.add(arxiv_id)
            
            # Strategy 3: Exact title matching
            title = paper.get('title', '').strip().lower()
            title_normalized = self._normalize_title(title)
            
            if not is_duplicate and title_normalized in seen_titles:
                is_duplicate = True
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
                            # Replace with better version
                            unique_papers.remove(existing_paper)
                            unique_papers.append(paper)
                            title_map[title_normalized] = paper
                        
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