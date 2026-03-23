"""
Test Enhanced Scraper with All Sources
"""

from scrapers.enhanced_paper_scraper import EnhancedPaperScraper
from config import *

def main():
    print("=" * 80)
    print("TESTING ENHANCED SCRAPER (8 SOURCES)")
    print("=" * 80)
    
    # Initialize with all your API keys
    scraper = EnhancedPaperScraper(
        semantic_scholar_api_key=SEMANTIC_SCHOLAR_API_KEY,
        ieee_api_key=IEEE_API_KEY if hasattr(locals(), 'IEEE_API_KEY') else None,
        email="your.email@example.com"  # For OpenAlex/CrossRef
    )
    
    print(f"\n✅ Available sources: {len(scraper.get_available_sources())}")
    for source in scraper.get_available_sources():
        print(f"  - {source}")
    
    # Test search
    print("\n🔍 Searching all sources...")
    query = input("Enter search query (or press Enter for 'machine learning'): ").strip()
    if not query:
        query = "machine learning"
    
    papers = scraper.search_all(
        query=query,
        max_results_per_source=20,
        start_year=2020,
        include_abstract=True  # ← Abstracts included!
    )
    
    print(f"\n📊 RESULTS: {len(papers)} unique papers")
    print("=" * 80)
    
    for i, paper in enumerate(papers[:100], 1):
        print(f"\n{i}. {paper['title']}")
        print(f"   Source: {paper['source']}")
        print(f"   Year: {paper['year']} | Citations: {paper['citations']}")
        print(f"   Authors: {', '.join(paper['authors'][:3])}...")
        
        # Show abstract
        if paper.get('abstract'):
            abstract = paper['abstract'][:200] + "..."
            print(f"   Abstract: {abstract}")
        
        print()

if __name__ == '__main__':
    main()