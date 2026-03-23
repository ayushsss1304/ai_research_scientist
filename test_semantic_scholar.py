"""
Semantic Scholar API Key Information Tester
Tests and displays all information available with your S2 API key
"""

import requests
import json
from typing import Dict, List
import time


class SemanticScholarAPITester:
    """Test Semantic Scholar API and display all available information"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = 'https://api.semanticscholar.org/graph/v1'
        self.headers = {'x-api-key': api_key} if api_key else {}
    
    def test_api_key_status(self) -> Dict:
        """Test API key and get rate limit info"""
        print("=" * 80)
        print("TESTING SEMANTIC SCHOLAR API KEY")
        print("=" * 80)
        
        try:
            response = requests.get(
                f'{self.base_url}/paper/search',
                params={'query': 'test', 'limit': 1},
                headers=self.headers,
                timeout=10
            )
            
            print(f"\n✅ API Key Status: {'VALID' if response.status_code == 200 else 'INVALID'}")
            print(f"Status Code: {response.status_code}")
            
            # Rate limit information
            print(f"\n📊 RATE LIMIT INFORMATION:")
            print(f"  Rate Limit: {response.headers.get('x-ratelimit-limit', 'Unknown')}")
            print(f"  Remaining: {response.headers.get('x-ratelimit-remaining', 'Unknown')}")
            print(f"  Reset Time: {response.headers.get('x-ratelimit-reset', 'Unknown')}")
            
            return {
                'valid': response.status_code == 200,
                'status_code': response.status_code,
                'rate_limit': response.headers.get('x-ratelimit-limit'),
                'remaining': response.headers.get('x-ratelimit-remaining')
            }
        
        except Exception as e:
            print(f"\n❌ Error testing API key: {e}")
            return {'valid': False, 'error': str(e)}
    
    def get_paper_full_details(self, paper_id: str = 'CorpusId:470667') -> Dict:
        """Get full details of a paper including all available fields"""
        print("\n" + "=" * 80)
        print("TESTING PAPER DETAILS RETRIEVAL")
        print("=" * 80)
        
        # All available fields in S2 API
        fields = [
            'paperId', 'corpusId', 'url', 'title', 'abstract', 'venue', 
            'publicationVenue', 'year', 'referenceCount', 'citationCount',
            'influentialCitationCount', 'isOpenAccess', 'openAccessPdf',
            'fieldsOfStudy', 'publicationTypes', 'publicationDate',
            'journal', 'authors', 'citations', 'references', 
            'embedding', 'tldr', 'externalIds'
        ]
        
        try:
            response = requests.get(
                f'{self.base_url}/paper/{paper_id}',
                params={'fields': ','.join(fields)},
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"\n📄 PAPER INFORMATION:")
                print(f"  Title: {data.get('title', 'N/A')}")
                print(f"  Paper ID: {data.get('paperId', 'N/A')}")
                print(f"  Year: {data.get('year', 'N/A')}")
                print(f"  Citations: {data.get('citationCount', 0)}")
                print(f"  Influential Citations: {data.get('influentialCitationCount', 0)}")
                print(f"  Reference Count: {data.get('referenceCount', 0)}")
                
                print(f"\n📚 PUBLICATION INFO:")
                print(f"  Venue: {data.get('venue', 'N/A')}")
                print(f"  Journal: {data.get('journal', {}).get('name', 'N/A')}")
                print(f"  Publication Date: {data.get('publicationDate', 'N/A')}")
                print(f"  Open Access: {data.get('isOpenAccess', False)}")
                
                print(f"\n👥 AUTHORS ({len(data.get('authors', []))}):")
                for i, author in enumerate(data.get('authors', [])[:5], 1):
                    print(f"  {i}. {author.get('name', 'Unknown')}")
                    print(f"     Author ID: {author.get('authorId', 'N/A')}")
                    print(f"     Paper Count: {author.get('paperCount', 'N/A')}")
                    print(f"     Citation Count: {author.get('citationCount', 'N/A')}")
                
                print(f"\n🏷️ FIELDS OF STUDY:")
                for field in data.get('fieldsOfStudy', []):
                    print(f"  - {field}")
                
                print(f"\n🔗 EXTERNAL IDs:")
                external_ids = data.get('externalIds', {})
                for key, value in external_ids.items():
                    print(f"  {key}: {value}")
                
                if data.get('tldr'):
                    print(f"\n📝 TL;DR:")
                    print(f"  {data['tldr'].get('text', 'N/A')}")
                
                print(f"\n🔗 URLs:")
                print(f"  S2 URL: {data.get('url', 'N/A')}")
                if data.get('openAccessPdf'):
                    print(f"  PDF URL: {data['openAccessPdf'].get('url', 'N/A')}")
                
                return data
            
            else:
                print(f"\n❌ Failed to get paper details: {response.status_code}")
                return {}
        
        except Exception as e:
            print(f"\n❌ Error getting paper details: {e}")
            return {}
    
    def search_papers_advanced(self, query: str = 'machine learning', limit: int = 5):
        """Search papers with all available filters"""
        print("\n" + "=" * 80)
        print("TESTING ADVANCED SEARCH")
        print("=" * 80)
        
        fields = [
            'paperId', 'title', 'abstract', 'year', 'authors',
            'citationCount', 'influentialCitationCount', 'venue',
            'publicationDate', 'url', 'openAccessPdf', 'journal',
            'fieldsOfStudy', 'publicationTypes', 'externalIds', 'tldr'
        ]
        
        try:
            response = requests.get(
                f'{self.base_url}/paper/search',
                params={
                    'query': query,
                    'limit': limit,
                    'fields': ','.join(fields),
                    'year': '2020-',  # Papers from 2020 onwards
                    'minCitationCount': 10
                },
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                papers = data.get('data', [])
                
                print(f"\n🔍 SEARCH RESULTS for '{query}':")
                print(f"Total: {data.get('total', 0)} papers")
                print(f"Showing: {len(papers)} papers\n")
                
                for i, paper in enumerate(papers, 1):
                    print(f"{i}. {paper.get('title', 'No title')}")
                    print(f"   Year: {paper.get('year', 'N/A')} | Citations: {paper.get('citationCount', 0)}")
                    print(f"   Authors: {', '.join([a.get('name', '') for a in paper.get('authors', [])[:3]])}...")
                    print(f"   Venue: {paper.get('venue', 'N/A')}")
                    
                    if paper.get('abstract'):
                        abstract = paper['abstract'][:150] + "..."
                        print(f"   Abstract: {abstract}")
                    
                    if paper.get('tldr'):
                        print(f"   TL;DR: {paper['tldr'].get('text', '')}")
                    
                    print()
                
                return papers
            
            else:
                print(f"\n❌ Search failed: {response.status_code}")
                return []
        
        except Exception as e:
            print(f"\n❌ Error searching: {e}")
            return []
    
    def get_author_info(self, author_id: str = '1741101'):
        """Get detailed author information"""
        print("\n" + "=" * 80)
        print("TESTING AUTHOR INFORMATION")
        print("=" * 80)
        
        fields = [
            'authorId', 'name', 'aliases', 'affiliations',
            'homepage', 'paperCount', 'citationCount',
            'hIndex', 'papers', 'externalIds'
        ]
        
        try:
            response = requests.get(
                f'{self.base_url}/author/{author_id}',
                params={'fields': ','.join(fields)},
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"\n👤 AUTHOR INFORMATION:")
                print(f"  Name: {data.get('name', 'N/A')}")
                print(f"  Author ID: {data.get('authorId', 'N/A')}")
                print(f"  Paper Count: {data.get('paperCount', 0)}")
                print(f"  Citation Count: {data.get('citationCount', 0)}")
                print(f"  h-Index: {data.get('hIndex', 'N/A')}")
                print(f"  Homepage: {data.get('homepage', 'N/A')}")
                
                if data.get('affiliations'):
                    print(f"\n🏛️ AFFILIATIONS:")
                    for aff in data['affiliations']:
                        print(f"  - {aff}")
                
                if data.get('papers'):
                    print(f"\n📚 TOP PAPERS:")
                    for i, paper in enumerate(data['papers'][:5], 1):
                        print(f"  {i}. {paper.get('title', 'N/A')}")
                        print(f"     Citations: {paper.get('citationCount', 0)}")
                
                return data
            
            else:
                print(f"\n❌ Failed to get author info: {response.status_code}")
                return {}
        
        except Exception as e:
            print(f"\n❌ Error getting author info: {e}")
            return {}
    
    def test_bulk_operations(self):
        """Test bulk paper retrieval"""
        print("\n" + "=" * 80)
        print("TESTING BULK OPERATIONS")
        print("=" * 80)
        
        paper_ids = [
            'CorpusId:470667',
            'CorpusId:37567189',
            'CorpusId:12345678'
        ]
        
        try:
            response = requests.post(
                f'{self.base_url}/paper/batch',
                params={'fields': 'title,year,citationCount'},
                json={'ids': paper_ids},
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                papers = response.json()
                print(f"\n✅ Retrieved {len(papers)} papers in bulk")
                
                for paper in papers:
                    if paper:
                        print(f"  - {paper.get('title', 'N/A')} ({paper.get('year', 'N/A')})")
                
                return papers
            
            else:
                print(f"\n❌ Bulk operation failed: {response.status_code}")
                return []
        
        except Exception as e:
            print(f"\n❌ Error in bulk operation: {e}")
            return []
    
    def get_recommendations(self, paper_id: str = 'CorpusId:470667', limit: int = 5):
        """Get paper recommendations"""
        print("\n" + "=" * 80)
        print("TESTING PAPER RECOMMENDATIONS")
        print("=" * 80)
        
        try:
            response = requests.get(
                f'{self.base_url}/paper/{paper_id}/recommendations',
                params={
                    'fields': 'title,year,citationCount,authors',
                    'limit': limit
                },
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                papers = data.get('recommendedPapers', [])
                
                print(f"\n💡 RECOMMENDATIONS:")
                for i, paper in enumerate(papers, 1):
                    print(f"{i}. {paper.get('title', 'N/A')}")
                    print(f"   Year: {paper.get('year', 'N/A')} | Citations: {paper.get('citationCount', 0)}")
                
                return papers
            
            else:
                print(f"\n❌ Recommendations failed: {response.status_code}")
                return []
        
        except Exception as e:
            print(f"\n❌ Error getting recommendations: {e}")
            return []
    
    def run_all_tests(self):
        """Run all API tests"""
        print("\n" + "=" * 80)
        print("SEMANTIC SCHOLAR API - COMPREHENSIVE TEST")
        print("=" * 80)
        
        # Test 1: API Key Status
        self.test_api_key_status()
        time.sleep(1)
        
        # Test 2: Paper Details
        self.get_paper_full_details()
        time.sleep(1)
        
        # Test 3: Advanced Search
        self.search_papers_advanced()
        time.sleep(1)
        
        # Test 4: Author Info
        self.get_author_info()
        time.sleep(1)
        
        # Test 5: Bulk Operations
        self.test_bulk_operations()
        time.sleep(1)
        
        # Test 6: Recommendations
        self.get_recommendations()
        
        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED")
        print("=" * 80)


def main():
    """Main function"""
    print("\nSemantic Scholar API Key Tester")
    print("=" * 80)
    
    # Get API key
    api_key = input("\nEnter your Semantic Scholar API key (or press Enter to skip): ").strip()
    
    if not api_key:
        print("\n⚠️  No API key provided. Testing without authentication...")
        print("Note: Rate limits will be lower without an API key.")
    
    # Create tester
    tester = SemanticScholarAPITester(api_key)
    
    # Run all tests
    tester.run_all_tests()
    
    # Summary
    print("\n📋 AVAILABLE S2 API FEATURES:")
    print("  ✓ Paper search with advanced filters")
    print("  ✓ Full paper metadata (title, abstract, authors, citations)")
    print("  ✓ Author information and statistics")
    print("  ✓ Citation and reference networks")
    print("  ✓ Paper recommendations")
    print("  ✓ Bulk operations")
    print("  ✓ TL;DR summaries")
    print("  ✓ Fields of study categorization")
    print("  ✓ Open access PDF links")
    print("  ✓ External IDs (DOI, ArXiv, PubMed, etc.)")
    print("\n✅ With an API key, you get higher rate limits!")


if __name__ == '__main__':
    main()