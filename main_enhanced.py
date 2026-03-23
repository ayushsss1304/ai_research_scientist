"""
Enhanced Main Application
Includes both old features AND new features
"""

import argparse
import logging
from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='AI Research Scientist - Enhanced')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Search command (enhanced)
    search_parser = subparsers.add_parser('search', help='Search papers (all 8 sources)')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--max-results', type=int, default=10)
    search_parser.add_argument('--start-year', type=int)
    search_parser.add_argument('--min-citations', type=int)
    search_parser.add_argument('--sources', nargs='+', help='Specific sources to search')
    
    # RAG chatbot command (independent)
    rag_parser = subparsers.add_parser('rag', help='Standalone RAG chatbot')
    rag_parser.add_argument('--add-pdf', action='append', help='Add PDF to RAG')
    
    # Original commands
    pdf_parser = subparsers.add_parser('process-pdf', help='Process PDF')
    pdf_parser.add_argument('pdf_path')
    
    args = parser.parse_args()
    
    if args.command == 'search':
        # Use enhanced scraper
        from scrapers.enhanced_paper_scraper import EnhancedPaperScraper
        
        scraper = EnhancedPaperScraper(
            semantic_scholar_api_key=SEMANTIC_SCHOLAR_API_KEY,
            email="your.email@example.com"
        )
        
        papers = scraper.search_all(
            query=args.query,
            max_results_per_source=args.max_results,
            start_year=args.start_year,
            min_citations=args.min_citations,
            sources=args.sources,
            include_abstract=True
        )
        
        print(f"\n{'='*80}")
        print(f"Found {len(papers)} papers")
        print(f"{'='*80}\n")
        
        for i, paper in enumerate(papers[:20], 1):
            print(f"{i}. {paper['title']}")
            print(f"   {paper['source']} | {paper['year']} | {paper['citations']} citations")
            if paper.get('abstract'):
                print(f"   Abstract: {paper['abstract'][:150]}...")
            print()
    
    elif args.command == 'rag':
        # Standalone RAG
        from rag_pipeline.standalone_chatbot import StandaloneRAGChatbot
        
        chatbot = StandaloneRAGChatbot(anthropic_api_key=ANTHROPIC_API_KEY)
        
        if args.add_pdf:
            for pdf in args.add_pdf:
                chatbot.add_document_from_file(pdf)
        
        # Interactive chat
        print("\n💬 RAG Chat (type 'exit' to quit)\n")
        while True:
            query = input("You: ").strip()
            if query.lower() == 'exit':
                break
            
            response = chatbot.chat(query)
            print(f"\nAssistant: {response}\n")
    
    elif args.command == 'process-pdf':
        from pdf_processing.pdf_processor import PDFProcessor
        
        processor = PDFProcessor()
        result = processor.process_document(args.pdf_path)
        
        print(f"\n✅ Processed: {args.pdf_path}")
        print(f"   Characters: {len(result['text'])}")
        print(f"   Chunks: {len(result['chunks'])}")

if __name__ == '__main__':
    main()