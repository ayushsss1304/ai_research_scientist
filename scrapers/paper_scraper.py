"""
Unified paper scraper - FIXED VERSION
Handles all sources with proper error handling
"""

from typing import List, Dict, Optional
import logging

from .arxiv_scraper import ArxivScraper
from .semantic_scholar_scraper import SemanticScholarScraper
from .ieee_scraper import IEEEScraper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PaperScraper:
    """Unified scraper for multiple sources"""
    
    def __init__(self, ieee_api_key: Optional[str] = None,
                 semantic_scholar_api_key: Optional[str] = None):
        """
        Initialize paper scraper
        
        Args:
            ieee_api_key: IEEE API key (optional)
            semantic_scholar_api_key: Semantic Scholar API key (optional)
        """
        self.arxiv = ArxivScraper()
        self.semantic_scholar = SemanticScholarScraper(semantic_scholar_api_key)
        
        # Only initialize IEEE if we have a valid key
        self.ieee = None
        if ieee_api_key and ieee_api_key not in ['your_ieee_api_key', '', 'YOUR_KEY_HERE']:
            try:
                self.ieee = IEEEScraper(ieee_api_key)
                # Test the key
                if self.ieee.test_api_key():
                    logger.info("IEEE Xplore initialized successfully")
                else:
                    logger.warning("IEEE API key test failed - IEEE search disabled")
                    self.ieee = None
            except Exception as e:
                logger.warning(f"Failed to initialize IEEE scraper: {e}")
                self.ieee = None
        else:
            logger.info("IEEE API key not provided - IEEE search disabled")
    
    def search_all(self, query: str, max_results_per_source: int = 10,
                   start_year: Optional[int] = None,
                   end_year: Optional[int] = None,
                   min_citations: Optional[int] = None,
                   journal_filter: Optional[str] = None,
                   sources: List[str] = None) -> List[Dict]:
        """
        Search all available sources
        
        Args:
            query: Search query
            max_results_per_source: Max results per source
            start_year: Start year filter
            end_year: End year filter
            min_citations: Minimum citations filter
            journal_filter: Journal name filter
            sources: List of sources to search (default: all available)
            
        Returns:
            List of paper dictionaries
        """
        if sources is None:
            sources = ['arxiv', 'semantic_scholar']
            if self.ieee:
                sources.append('ieee')
        
        all_papers = []
        
        # Search ArXiv
        if 'arxiv' in sources:
            try:
                logger.info("Searching ArXiv...")
                arxiv_papers = self.arxiv.search(
                    query, max_results_per_source, start_year, end_year
                )
                all_papers.extend(arxiv_papers)
                logger.info(f"✓ Found {len(arxiv_papers)} papers from ArXiv")
            except Exception as e:
                logger.error(f"ArXiv search failed: {e}")
        
        # Search Semantic Scholar
        if 'semantic_scholar' in sources:
            try:
                logger.info("Searching Semantic Scholar...")
                s2_papers = self.semantic_scholar.search(
                    query, max_results_per_source, start_year, end_year, min_citations
                )
                all_papers.extend(s2_papers)
                logger.info(f"✓ Found {len(s2_papers)} papers from Semantic Scholar")
            except Exception as e:
                logger.error(f"Semantic Scholar search failed: {e}")
        
        # Search IEEE
        if 'ieee' in sources and self.ieee:
            try:
                logger.info("Searching IEEE Xplore...")
                ieee_papers = self.ieee.search(
                    query, max_results_per_source, start_year, end_year, min_citations
                )
                all_papers.extend(ieee_papers)
                logger.info(f"✓ Found {len(ieee_papers)} papers from IEEE Xplore")
            except Exception as e:
                logger.error(f"IEEE search failed: {e}")
        elif 'ieee' in sources and not self.ieee:
            logger.info("IEEE search skipped - not initialized")
        
        # Apply journal filter if specified
        if journal_filter:
            original_count = len(all_papers)
            all_papers = [
                p for p in all_papers
                if journal_filter.lower() in p.get('journal', '').lower()
            ]
            logger.info(f"Journal filter applied: {original_count} -> {len(all_papers)} papers")
        
        # Remove duplicates
        all_papers = self._remove_duplicates(all_papers)
        
        # Sort by citations
        all_papers.sort(key=lambda x: x.get('citations', 0), reverse=True)
        
        logger.info(f"Total unique papers found: {len(all_papers)}")
        
        return all_papers
    
    def _remove_duplicates(self, papers: List[Dict]) -> List[Dict]:
        """
        Remove duplicate papers based on title similarity
        
        Args:
            papers: List of paper dictionaries
            
        Returns:
            List with duplicates removed
        """
        unique_papers = []
        seen_titles = set()
        
        for paper in papers:
            # Normalize title for comparison
            title_normalized = paper['title'].lower().strip()
            title_normalized = ''.join(c for c in title_normalized if c.isalnum() or c.isspace())
            
            if title_normalized not in seen_titles:
                seen_titles.add(title_normalized)
                unique_papers.append(paper)
        
        if len(papers) != len(unique_papers):
            logger.info(f"Removed {len(papers) - len(unique_papers)} duplicate papers")
        
        return unique_papers
    
    def get_available_sources(self) -> List[str]:
        """
        Get list of available sources
        
        Returns:
            List of source names
        """
        sources = ['arxiv', 'semantic_scholar']
        if self.ieee:
            sources.append('ieee')
        return sources
    
    def test_all_sources(self) -> Dict[str, bool]:
        """
        Test all configured sources
        
        Returns:
            Dictionary with source name and status
        """
        status = {}
        
        # Test ArXiv
        try:
            test_papers = self.arxiv.search('test', max_results=1)
            status['arxiv'] = len(test_papers) > 0
        except:
            status['arxiv'] = False
        
        # Test Semantic Scholar
        try:
            test_papers = self.semantic_scholar.search('test', max_results=1)
            status['semantic_scholar'] = len(test_papers) > 0
        except:
            status['semantic_scholar'] = False
        
        # Test IEEE
        if self.ieee:
            status['ieee'] = self.ieee.test_api_key()
        else:
            status['ieee'] = False
        
        return status