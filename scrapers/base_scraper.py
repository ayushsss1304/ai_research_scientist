from abc import ABC, abstractmethod
from typing import List, Dict, Optional

class BaseScraper(ABC):
    '''Base class for all paper scrapers'''
    
    @abstractmethod
    def search(self, query: str, max_results: int = 10, **kwargs) -> List[Dict]:
        '''Search for papers'''
        pass
    
    def _apply_filters(self, papers: List[Dict], 
                       start_year: Optional[int] = None,
                       end_year: Optional[int] = None,
                       min_citations: Optional[int] = None,
                       journal_filter: Optional[str] = None) -> List[Dict]:
        '''Apply common filters to papers'''
        
        filtered = papers
        
        if start_year:
            filtered = [p for p in filtered if p.get('year', 0) >= start_year]
        
        if end_year:
            filtered = [p for p in filtered if p.get('year', 0) <= end_year]
        
        if min_citations:
            filtered = [p for p in filtered if p.get('citations', 0) >= min_citations]
        
        if journal_filter:
            filtered = [p for p in filtered 
                       if journal_filter.lower() in p.get('journal', '').lower()]
        
        return filtered