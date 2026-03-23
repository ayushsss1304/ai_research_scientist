"""
Additional Research Paper Scrapers
Includes: PubMed, CORE, OpenAlex, CrossRef, Google Scholar
"""

import requests
import time
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import logging
from .base_scraper import BaseScraper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PubMedScraper(BaseScraper):
    """Scraper for PubMed (biomedical literature)"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.base_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils'
        self.api_key = api_key  # Optional, increases rate limit
    
    def search(self, query: str, max_results: int = 10,
               start_year: Optional[int] = None,
               end_year: Optional[int] = None,
               **kwargs) -> List[Dict]:
        """Search PubMed for papers"""
        
        logger.info(f"Searching PubMed for: {query}")
        
        # Build date filter
        date_filter = ""
        if start_year and end_year:
            date_filter = f"+AND+{start_year}:{end_year}[pdat]"
        elif start_year:
            date_filter = f"+AND+{start_year}:3000[pdat]"
        
        # Search for IDs
        search_url = f"{self.base_url}/esearch.fcgi"
        params = {
            'db': 'pubmed',
            'term': query + date_filter,
            'retmax': max_results,
            'retmode': 'json'
        }
        
        if self.api_key:
            params['api_key'] = self.api_key
        
        try:
            response = requests.get(search_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            ids = data.get('esearchresult', {}).get('idlist', [])
            
            if not ids:
                logger.info("No PubMed results found")
                return []
            
            # Fetch details for each ID
            papers = self._fetch_details(ids)
            logger.info(f"PubMed returned {len(papers)} papers")
            
            return papers
        
        except Exception as e:
            logger.error(f"PubMed search error: {e}")
            return []
    
    def _fetch_details(self, ids: List[str]) -> List[Dict]:
        """Fetch paper details from PubMed IDs"""
        
        fetch_url = f"{self.base_url}/efetch.fcgi"
        params = {
            'db': 'pubmed',
            'id': ','.join(ids),
            'retmode': 'xml'
        }
        
        if self.api_key:
            params['api_key'] = self.api_key
        
        try:
            response = requests.get(fetch_url, params=params, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'xml')
            papers = []
            
            for article in soup.find_all('PubmedArticle'):
                paper = self._parse_pubmed_article(article)
                if paper:
                    papers.append(paper)
            
            time.sleep(0.4)  # Rate limit: 3 requests per second without key
            return papers
        
        except Exception as e:
            logger.error(f"Error fetching PubMed details: {e}")
            return []
    
    def _parse_pubmed_article(self, article) -> Optional[Dict]:
        """Parse PubMed XML article"""
        
        try:
            title_elem = article.find('ArticleTitle')
            title = title_elem.text if title_elem else ''
            
            abstract_elem = article.find('AbstractText')
            abstract = abstract_elem.text if abstract_elem else ''
            
            year_elem = article.find('PubDate').find('Year') if article.find('PubDate') else None
            year = int(year_elem.text) if year_elem else 0
            
            authors = []
            for author in article.find_all('Author'):
                lastname = author.find('LastName')
                forename = author.find('ForeName')
                if lastname and forename:
                    authors.append(f"{forename.text} {lastname.text}")
            
            journal_elem = article.find('Journal').find('Title') if article.find('Journal') else None
            journal = journal_elem.text if journal_elem else ''
            
            pmid_elem = article.find('PMID')
            pmid = pmid_elem.text if pmid_elem else ''
            
            doi_elem = article.find('ArticleId', {'IdType': 'doi'})
            doi = doi_elem.text if doi_elem else ''
            
            return {
                'title': title,
                'authors': authors,
                'abstract': abstract,
                'year': year,
                'published_date': '',
                'pdf_url': '',
                'source': 'PubMed',
                'citations': 0,
                'journal': journal,
                'doi': doi,
                'pmid': pmid,
                'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ''
            }
        
        except Exception as e:
            logger.warning(f"Error parsing PubMed article: {e}")
            return None


class COREScrap(BaseScraper):
    """Scraper for CORE (Open access research papers)"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.base_url = 'https://api.core.ac.uk/v3'
        self.api_key = api_key  # Get free key at https://core.ac.uk/services/api
    
    def search(self, query: str, max_results: int = 10,
               start_year: Optional[int] = None,
               end_year: Optional[int] = None,
               **kwargs) -> List[Dict]:
        """Search CORE for papers"""
        
        if not self.api_key:
            logger.warning("CORE API key required - skipping")
            return []
        
        logger.info(f"Searching CORE for: {query}")
        
        headers = {'Authorization': f'Bearer {self.api_key}'}
        
        params = {
            'q': query,
            'limit': max_results
        }
        
        try:
            response = requests.post(
                f'{self.base_url}/search/works',
                json=params,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            papers = []
            
            for result in data.get('results', []):
                paper = self._parse_core_paper(result)
                if paper:
                    # Apply year filter
                    if start_year and paper['year'] < start_year:
                        continue
                    if end_year and paper['year'] > end_year:
                        continue
                    
                    papers.append(paper)
            
            logger.info(f"CORE returned {len(papers)} papers")
            return papers
        
        except Exception as e:
            logger.error(f"CORE search error: {e}")
            return []
    
    def _parse_core_paper(self, result: Dict) -> Optional[Dict]:
        """Parse CORE paper data"""
        
        try:
            return {
                'title': result.get('title', ''),
                'authors': result.get('authors', []),
                'abstract': result.get('abstract', ''),
                'year': result.get('yearPublished', 0),
                'published_date': result.get('publishedDate', ''),
                'pdf_url': result.get('downloadUrl', ''),
                'source': 'CORE',
                'citations': 0,
                'journal': result.get('publisher', ''),
                'doi': result.get('doi', ''),
                'url': result.get('links', [{}])[0].get('url', '')
            }
        
        except Exception as e:
            logger.warning(f"Error parsing CORE paper: {e}")
            return None


class OpenAlexScraper(BaseScraper):
    """Scraper for OpenAlex (comprehensive academic database)"""
    
    def __init__(self, email: Optional[str] = None):
        self.base_url = 'https://api.openalex.org'
        self.email = email  # Polite pool gets faster rate limit
    
    def search(self, query: str, max_results: int = 10,
               start_year: Optional[int] = None,
               end_year: Optional[int] = None,
               min_citations: Optional[int] = None,
               **kwargs) -> List[Dict]:
        """Search OpenAlex for papers"""
        
        logger.info(f"Searching OpenAlex for: {query}")
        
        params = {
            'search': query,
            'per-page': max_results
        }
        
        # Add filters
        filters = []
        if start_year:
            filters.append(f'publication_year:{start_year}-')
        if end_year:
            filters.append(f'publication_year:-{end_year}')
        if min_citations:
            filters.append(f'cited_by_count:>{min_citations}')
        
        if filters:
            params['filter'] = ','.join(filters)
        
        headers = {}
        if self.email:
            headers['User-Agent'] = f'mailto:{self.email}'
        
        try:
            response = requests.get(
                f'{self.base_url}/works',
                params=params,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            papers = []
            
            for work in data.get('results', []):
                paper = self._parse_openalex_work(work)
                if paper:
                    papers.append(paper)
            
            logger.info(f"OpenAlex returned {len(papers)} papers")
            return papers
        
        except Exception as e:
            logger.error(f"OpenAlex search error: {e}")
            return []
    
    def _parse_openalex_work(self, work: Dict) -> Optional[Dict]:
        """Parse OpenAlex work data"""
        
        try:
            authors = [
                authorship.get('author', {}).get('display_name', '')
                for authorship in work.get('authorships', [])
            ]
            
            pdf_url = ''
            if work.get('open_access') and work['open_access'].get('oa_url'):
                pdf_url = work['open_access']['oa_url']
            
            return {
                'title': work.get('title', ''),
                'authors': authors,
                'abstract': work.get('abstract', ''),
                'year': work.get('publication_year', 0),
                'published_date': work.get('publication_date', ''),
                'pdf_url': pdf_url,
                'source': 'OpenAlex',
                'citations': work.get('cited_by_count', 0),
                'journal': work.get('host_venue', {}).get('display_name', ''),
                'doi': work.get('doi', '').replace('https://doi.org/', ''),
                'url': work.get('id', ''),
                'fields': [concept.get('display_name', '') for concept in work.get('concepts', [])[:5]]
            }
        
        except Exception as e:
            logger.warning(f"Error parsing OpenAlex work: {e}")
            return None


class CrossRefScraper(BaseScraper):
    """Scraper for CrossRef (DOI metadata)"""
    
    def __init__(self, email: Optional[str] = None):
        self.base_url = 'https://api.crossref.org/works'
        self.email = email  # Polite pool
    
    def search(self, query: str, max_results: int = 10,
               start_year: Optional[int] = None,
               end_year: Optional[int] = None,
               **kwargs) -> List[Dict]:
        """Search CrossRef for papers"""
        
        logger.info(f"Searching CrossRef for: {query}")
        
        params = {
            'query': query,
            'rows': max_results,
            'sort': 'relevance'
        }
        
        if start_year:
            params['filter'] = f'from-pub-date:{start_year}'
        if end_year:
            if 'filter' in params:
                params['filter'] += f',until-pub-date:{end_year}'
            else:
                params['filter'] = f'until-pub-date:{end_year}'
        
        headers = {}
        if self.email:
            headers['User-Agent'] = f'mailto:{self.email}'
        
        try:
            response = requests.get(
                self.base_url,
                params=params,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            papers = []
            
            for item in data.get('message', {}).get('items', []):
                paper = self._parse_crossref_item(item)
                if paper:
                    papers.append(paper)
            
            logger.info(f"CrossRef returned {len(papers)} papers")
            return papers
        
        except Exception as e:
            logger.error(f"CrossRef search error: {e}")
            return []
    
    def _parse_crossref_item(self, item: Dict) -> Optional[Dict]:
        """Parse CrossRef item"""
        
        try:
            authors = []
            for author in item.get('author', []):
                given = author.get('given', '')
                family = author.get('family', '')
                authors.append(f"{given} {family}".strip())
            
            year = 0
            if item.get('published-print'):
                date_parts = item['published-print'].get('date-parts', [[]])[0]
                year = date_parts[0] if date_parts else 0
            
            return {
                'title': item.get('title', [''])[0],
                'authors': authors,
                'abstract': item.get('abstract', ''),
                'year': year,
                'published_date': '',
                'pdf_url': '',
                'source': 'CrossRef',
                'citations': item.get('is-referenced-by-count', 0),
                'journal': item.get('container-title', [''])[0],
                'doi': item.get('DOI', ''),
                'url': item.get('URL', '')
            }
        
        except Exception as e:
            logger.warning(f"Error parsing CrossRef item: {e}")
            return None


class GoogleScholarScraper(BaseScraper):
    """
    Scraper for Google Scholar (via SerpAPI or ScraperAPI)
    Note: Google Scholar blocks automated scraping, so we need a service
    """
    
    def __init__(self, serpapi_key: Optional[str] = None):
        self.serpapi_key = serpapi_key
        self.base_url = 'https://serpapi.com/search'
    
    def search(self, query: str, max_results: int = 10,
               start_year: Optional[int] = None,
               end_year: Optional[int] = None,
               **kwargs) -> List[Dict]:
        """
        Search Google Scholar via SerpAPI
        Get free key at: https://serpapi.com/
        """
        
        if not self.serpapi_key:
            logger.warning("Google Scholar requires SerpAPI key - skipping")
            return []
        
        logger.info(f"Searching Google Scholar for: {query}")
        
        # Build year range
        as_ylo = start_year if start_year else ''
        as_yhi = end_year if end_year else ''
        
        params = {
            'engine': 'google_scholar',
            'q': query,
            'api_key': self.serpapi_key,
            'num': max_results,
            'as_ylo': as_ylo,
            'as_yhi': as_yhi
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            papers = []
            
            for result in data.get('organic_results', []):
                paper = self._parse_scholar_result(result)
                if paper:
                    papers.append(paper)
            
            logger.info(f"Google Scholar returned {len(papers)} papers")
            return papers
        
        except Exception as e:
            logger.error(f"Google Scholar search error: {e}")
            return []
    
    def _parse_scholar_result(self, result: Dict) -> Optional[Dict]:
        """Parse Google Scholar result"""
        
        try:
            # Extract authors from snippet
            authors = []
            publication_info = result.get('publication_info', {})
            if publication_info.get('authors'):
                authors = [a.get('name', '') for a in publication_info['authors']]
            
            # Extract year
            year = 0
            if publication_info.get('summary'):
                import re
                year_match = re.search(r'\b(19|20)\d{2}\b', publication_info['summary'])
                if year_match:
                    year = int(year_match.group())
            
            return {
                'title': result.get('title', ''),
                'authors': authors,
                'abstract': result.get('snippet', ''),
                'year': year,
                'published_date': '',
                'pdf_url': result.get('resources', [{}])[0].get('link', ''),
                'source': 'Google Scholar',
                'citations': int(result.get('inline_links', {}).get('cited_by', {}).get('total', 0)),
                'journal': publication_info.get('summary', ''),
                'doi': '',
                'url': result.get('link', '')
            }
        
        except Exception as e:
            logger.warning(f"Error parsing Google Scholar result: {e}")
            return None