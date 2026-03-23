"""
Scrapers package for fetching research papers from multiple sources
"""

from .arxiv_scraper import ArxivScraper
from .semantic_scholar_scraper import SemanticScholarScraper
from .ieee_scraper import IEEEScraper
from .paper_scraper import PaperScraper

__all__ = [
    'ArxivScraper',
    'SemanticScholarScraper', 
    'IEEEScraper',
    'PaperScraper'
]