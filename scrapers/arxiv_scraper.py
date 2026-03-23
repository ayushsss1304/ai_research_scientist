import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
from .base_scraper import BaseScraper

class ArxivScraper(BaseScraper):
    '''Scraper for ArXiv papers'''
    
    def __init__(self):
        self.base_url = 'http://export.arxiv.org/api/query'
    
    def search(self, query: str, max_results: int = 10, 
               start_year: Optional[int] = None,
               end_year: Optional[int] = None,
               **kwargs) -> List[Dict]:
        '''Search ArXiv for papers'''
        
        params = {
            'search_query': f'all:{query}',
            'start': 0,
            'max_results': max_results * 2,  # Get more to account for filtering
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            papers = self._parse_response(response.content)
            papers = self._apply_filters(papers, start_year, end_year)
            
            return papers[:max_results]
        
        except Exception as e:
            print(f"ArXiv search error: {e}")
            return []
    
    def _parse_response(self, content: bytes) -> List[Dict]:
        '''Parse ArXiv XML response'''
        
        papers = []
        root = ET.fromstring(content)
        
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            try:
                title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip()
                summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip()
                published = entry.find('{http://www.w3.org/2005/Atom}published').text
                
                authors = [
                    author.find('{http://www.w3.org/2005/Atom}name').text
                    for author in entry.findall('{http://www.w3.org/2005/Atom}author')
                ]
                
                arxiv_id = entry.find('{http://www.w3.org/2005/Atom}id').text
                pdf_url = arxiv_id.replace('abs', 'pdf')
                
                year = int(published.split('-')[0])
                
                # Get categories
                categories = [
                    cat.get('term')
                    for cat in entry.findall('{http://www.w3.org/2005/Atom}category')
                ]
                
                papers.append({
                    'title': title,
                    'authors': authors,
                    'abstract': summary,
                    'year': year,
                    'published_date': published,
                    'pdf_url': pdf_url,
                    'source': 'ArXiv',
                    'citations': 0,  # ArXiv doesn't provide citations
                    'categories': categories,
                    'arxiv_id': arxiv_id.split('/')[-1],
                    'journal': 'ArXiv',
                    'doi': ''
                })
            
            except Exception as e:
                print(f"Error parsing ArXiv entry: {e}")
                continue
        
        return papers