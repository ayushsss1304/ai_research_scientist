"""
IEEE Xplore scraper - FIXED VERSION
Handles IEEE API properly with error handling
"""

import requests
from typing import List, Dict, Optional
import logging

from .base_scraper import BaseScraper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IEEEScraper(BaseScraper):
    """Scraper for IEEE Xplore papers"""
    
    def __init__(self, api_key: str):
        """
        Initialize IEEE scraper
        
        Args:
            api_key: IEEE API key
        """
        self.api_key = api_key
        self.base_url = 'http://ieeexploreapi.ieee.org/api/v1/search/articles'
        
        # Check if API key is valid (not placeholder)
        if not api_key or api_key in ['your_ieee_api_key', '', 'YOUR_KEY_HERE']:
            logger.warning("IEEE API key not set properly")
            self.api_key = None
    
    def search(self, query: str, max_results: int = 10,
               start_year: Optional[int] = None,
               end_year: Optional[int] = None,
               min_citations: Optional[int] = None,
               **kwargs) -> List[Dict]:
        """
        Search IEEE Xplore for papers
        
        Args:
            query: Search query
            max_results: Maximum results
            start_year: Filter by start year
            end_year: Filter by end year
            min_citations: Minimum citations
            **kwargs: Additional parameters
            
        Returns:
            List of paper dictionaries
        """
        # Skip if no valid API key
        if not self.api_key:
            logger.warning("IEEE search skipped - no valid API key")
            return []
        
        logger.info(f"Searching IEEE Xplore for: {query}")
        
        params = {
            'apikey': self.api_key,
            'querytext': query,
            'max_records': max_results,
            'sort_field': 'article_number',
            'sort_order': 'desc'
        }
        
        if start_year:
            params['start_year'] = start_year
        if end_year:
            params['end_year'] = end_year
        
        try:
            response = requests.get(
                self.base_url, 
                params=params, 
                timeout=30,
                headers={'Accept': 'application/json'}
            )
            
            # Check for specific error codes
            if response.status_code == 403:
                logger.error("IEEE API key is invalid or expired")
                logger.error("Get a new key at: https://developer.ieee.org/")
                return []
            
            if response.status_code == 401:
                logger.error("IEEE API authentication failed")
                return []
            
            if response.status_code == 429:
                logger.error("IEEE API rate limit exceeded")
                return []
            
            response.raise_for_status()
            
            data = response.json()
            
            # Check if we got results
            if 'articles' not in data:
                logger.warning("No articles found in IEEE response")
                return []
            
            papers = []
            
            for article in data.get('articles', []):
                paper = self._parse_article(article)
                
                # Apply citation filter
                if min_citations and paper['citations'] < min_citations:
                    continue
                
                papers.append(paper)
            
            logger.info(f"IEEE returned {len(papers)} papers")
            return papers
        
        except requests.exceptions.RequestException as e:
            logger.error(f"IEEE search error: {e}")
            return []
        
        except Exception as e:
            logger.error(f"Unexpected error in IEEE search: {e}")
            return []
    
    def _parse_article(self, article: Dict) -> Dict:
        """
        Parse IEEE article data
        
        Args:
            article: Article dictionary from IEEE API
            
        Returns:
            Normalized paper dictionary
        """
        authors = []
        if article.get('authors') and article['authors'].get('authors'):
            authors = [
                author.get('full_name', '')
                for author in article['authors']['authors']
            ]
        
        article_number = article.get('article_number', '')
        
        return {
            'title': article.get('title', ''),
            'authors': authors,
            'abstract': article.get('abstract', ''),
            'year': article.get('publication_year', 0),
            'published_date': article.get('publication_date', ''),
            'pdf_url': article.get('pdf_url', ''),
            'source': 'IEEE Xplore',
            'citations': article.get('citing_paper_count', 0),
            'journal': article.get('publication_title', ''),
            'doi': article.get('doi', ''),
            'article_number': article_number,
            'isbn': article.get('isbn', ''),
            'issn': article.get('issn', ''),
            'url': f"https://ieeexplore.ieee.org/document/{article_number}" if article_number else ''
        }
    
    def test_api_key(self) -> bool:
        """
        Test if IEEE API key is valid
        
        Returns:
            True if valid, False otherwise
        """
        if not self.api_key:
            return False
        
        try:
            params = {
                'apikey': self.api_key,
                'querytext': 'test',
                'max_records': 1
            }
            
            response = requests.get(
                self.base_url,
                params=params,
                timeout=10,
                headers={'Accept': 'application/json'}
            )
            
            if response.status_code == 200:
                logger.info("IEEE API key is valid")
                return True
            else:
                logger.error(f"IEEE API key test failed: {response.status_code}")
                return False
        
        except Exception as e:
            logger.error(f"IEEE API key test error: {e}")
            return False