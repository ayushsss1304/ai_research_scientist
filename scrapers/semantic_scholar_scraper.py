import requests
import time
from typing import List, Dict, Optional
from .base_scraper import BaseScraper

class SemanticScholarScraper(BaseScraper):
    '''Scraper for Semantic Scholar papers'''
    
    def __init__(self, api_key: Optional[str] = None):
        self.base_url = 'https://api.semanticscholar.org/graph/v1'
        self.api_key = api_key
        self.headers = {}
        
        if api_key:
            self.headers['x-api-key'] = api_key
    
    def search(self, query: str, max_results: int = 10,
               start_year: Optional[int] = None,
               end_year: Optional[int] = None,
               min_citations: Optional[int] = None,
               **kwargs) -> List[Dict]:
        '''Search Semantic Scholar for papers'''
        
        papers = []
        offset = 0
        limit = 100  # Max per request
        
        # Build query with filters
        search_query = query
        if start_year:
            search_query += f' year>={start_year}'
        if end_year:
            search_query += f' year<={end_year}'
        if min_citations:
            search_query += f' citations>={min_citations}'
        
        while len(papers) < max_results and offset < 1000:  # S2 limit
            try:
                params = {
                    'query': search_query,
                    'offset': offset,
                    'limit': min(limit, max_results - len(papers)),
                    'fields': 'paperId,title,abstract,year,authors,citationCount,venue,publicationDate,url,openAccessPdf,journal,fieldsOfStudy,publicationTypes,externalIds'
                }
                
                response = requests.get(
                    f'{self.base_url}/paper/search',
                    params=params,
                    headers=self.headers,
                    timeout=30
                )
                
                if response.status_code == 429:  # Rate limit
                    time.sleep(2)
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                if 'data' not in data or not data['data']:
                    break
                
                for paper in data['data']:
                    papers.append(self._parse_paper(paper))
                
                offset += limit
                time.sleep(0.5)  # Be nice to the API
                
                if len(data['data']) < limit:
                    break
            
            except Exception as e:
                print(f"Semantic Scholar search error: {e}")
                break
        
        # Apply additional filters
        papers = self._apply_filters(
            papers, 
            start_year=start_year,
            end_year=end_year,
            min_citations=min_citations
        )
        
        # Sort by citations
        papers.sort(key=lambda x: x['citations'], reverse=True)
        
        return papers[:max_results]
    
    def get_paper_details(self, paper_id: str) -> Optional[Dict]:
        '''Get detailed information about a specific paper'''
        
        try:
            response = requests.get(
                f'{self.base_url}/paper/{paper_id}',
                params={'fields': 'paperId,title,abstract,year,authors,citationCount,venue,publicationDate,url,openAccessPdf,journal,fieldsOfStudy,publicationTypes,references,citations,externalIds'},
                headers=self.headers,
                timeout=30
            )
            
            response.raise_for_status()
            return self._parse_paper(response.json())
        
        except Exception as e:
            print(f"Error fetching paper details: {e}")
            return None
    
    def _parse_paper(self, paper: Dict) -> Dict:
        '''Parse Semantic Scholar paper data'''
        
        authors = []
        if paper.get('authors'):
            authors = [author.get('name', '') for author in paper['authors']]
        
        # Get PDF URL
        pdf_url = ''
        if paper.get('openAccessPdf') and paper['openAccessPdf'].get('url'):
            pdf_url = paper['openAccessPdf']['url']
        
        # Get DOI
        doi = ''
        if paper.get('externalIds'):
            doi = paper['externalIds'].get('DOI', '')
        
        return {
            'title': paper.get('title', ''),
            'authors': authors,
            'abstract': paper.get('abstract', ''),
            'year': paper.get('year', 0),
            'published_date': paper.get('publicationDate', ''),
            'pdf_url': pdf_url,
            'source': 'Semantic Scholar',
            'citations': paper.get('citationCount', 0),
            'journal': paper.get('venue', '') or paper.get('journal', {}).get('name', ''),
            'doi': doi,
            'paper_id': paper.get('paperId', ''),
            'url': paper.get('url', ''),
            'fields': paper.get('fieldsOfStudy', []),
            'publication_types': paper.get('publicationTypes', [])
        }