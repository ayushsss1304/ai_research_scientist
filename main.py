"""
AI Research Scientist - Main Application
Entry point for the application
"""

import os
import sys
import argparse
import logging
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

# Import configurations
import config

# Import modules
from scrapers.paper_scraper import PaperScraper
from knowledge_graph.kg_manager import KnowledgeGraphManager
from pdf_processing.pdf_processor import PDFProcessor
#from chatbot.research_chatbot import ResearchChatbot
from scrapers.enhanced_paper_scraper import EnhancedPaperScraper


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ResearchScientistApp:
    """Main application class"""
    
    def __init__(self):
        """Initialize all components"""
        logger.info("Initializing AI Research Scientist...")
        
        # Initialize scrapers
        self.scraper = PaperScraper(
            ieee_api_key=config.IEEE_API_KEY,
            semantic_scholar_api_key=config.SEMANTIC_SCHOLAR_API_KEY
        )
        
        # Initialize knowledge graph (optional)
        self.kg = None
        try:
            self.kg = KnowledgeGraphManager(
                uri=config.NEO4J_URI,
                user=config.NEO4J_USER,
                password=config.NEO4J_PASSWORD
            )
            logger.info("Knowledge graph connected")
        except Exception as e:
            logger.warning(f"Knowledge graph not available: {e}")
        
        # Initialize PDF processor
        self.pdf_processor = PDFProcessor(use_ocr=config.USE_OCR)
        
        # Initialize chatbot
        self.chatbot = None
        #if config.ANTHROPIC_API_KEY and config.ANTHROPIC_API_KEY != 'your_anthropic_api_key':
            #self.chatbot = ResearchChatbot(
                #anthropic_api_key=config.ANTHROPIC_API_KEY
            #)
            #logger.info("Chatbot initialized")
        #else:
            #logger.warning("Chatbot not available - no Anthropic API key")
        
        #logger.info("Initialization complete!")
    
    def search_papers(self, query: str, **kwargs):
        """
        Search for papers across all sources
        
        Args:
            query: Search query
            **kwargs: Additional search parameters
        """
        logger.info(f"Searching for papers: {query}")
        
        papers = self.scraper.search_all(
            query=query,
            max_results_per_source=kwargs.get('max_results', 10),
            start_year=kwargs.get('start_year'),
            end_year=kwargs.get('end_year'),
            min_citations=kwargs.get('min_citations'),
            journal_filter=kwargs.get('journal_filter'),
            sources=kwargs.get('sources')
        )
        
        logger.info(f"Found {len(papers)} papers")
        
        return papers
    
    def add_papers_to_kg(self, papers):
        """Add papers to knowledge graph"""
        if not self.kg:
            logger.error("Knowledge graph not available")
            return 0
        
        return self.kg.add_papers_batch(papers)
    
    def process_pdf(self, pdf_path: str):
        """Process a PDF file"""
        logger.info(f"Processing PDF: {pdf_path}")
        
        result = self.pdf_processor.process_document(pdf_path)
        
        if result['error']:
            logger.error(f"Error processing PDF: {result['error']}")
        else:
            logger.info(f"Successfully processed PDF: {len(result['chunks'])} chunks")
        
        return result
    
    def add_pdf_to_chatbot(self, pdf_path: str, doc_id: Optional[str] = None):
        """Add PDF to chatbot"""
        if not self.chatbot:
            logger.error("Chatbot not available")
            return False
        
        if not doc_id:
            doc_id = os.path.basename(pdf_path).replace('.pdf', '')
        
        # Process PDF
        result = self.process_pdf(pdf_path)
        
        if result['error']:
            return False
        
        # Add to chatbot
        self.chatbot.add_document(
            file_path=pdf_path,
            doc_id=doc_id,
            text=result['text'],
            chunks=result['chunks']
        )
        
        logger.info(f"Added PDF to chatbot: {doc_id}")
        return True
    
    def chat(self, message: str):
        """Chat with the loaded papers"""
        if not self.chatbot:
            return "Chatbot not available - please check your Anthropic API key"
        
        return self.chatbot.chat(message)
    
    def close(self):
        """Clean up resources"""
        if self.kg:
            self.kg.close()
        logger.info("Application closed")


def cli():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description='AI Research Scientist')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for papers')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--max-results', type=int, default=10)
    search_parser.add_argument('--start-year', type=int)
    search_parser.add_argument('--end-year', type=int)
    search_parser.add_argument('--min-citations', type=int)
    search_parser.add_argument('--save-to-kg', action='store_true', 
                              help='Save results to knowledge graph')
    
    # Process PDF command
    pdf_parser = subparsers.add_parser('process-pdf', help='Process a PDF file')
    pdf_parser.add_argument('pdf_path', help='Path to PDF file')
    pdf_parser.add_argument('--add-to-chat', action='store_true',
                           help='Add to chatbot')
    
    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Chat with papers')
    chat_parser.add_argument('--pdf', action='append', help='PDF file to add')
    
    # KG query command
    kg_parser = subparsers.add_parser('kg-search', help='Search knowledge graph')
    kg_parser.add_argument('query', help='Search query')
    
    # Web interface command
    web_parser = subparsers.add_parser('web', help='Start web interface')
    web_parser.add_argument('--host', default='0.0.0.0')
    web_parser.add_argument('--port', type=int, default=5000)
    web_parser.add_argument('--debug', action='store_true')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize app
    app = ResearchScientistApp()
    
    try:
        if args.command == 'search':
            # Search for papers
            papers = app.search_papers(
                query=args.query,
                max_results=args.max_results,
                start_year=args.start_year,
                end_year=args.end_year,
                min_citations=args.min_citations
            )
            
            # Display results
            print(f"\n{'='*80}")
            print(f"Found {len(papers)} papers")
            print(f"{'='*80}\n")
            
            for i, paper in enumerate(papers, 1):
                print(f"{i}. {paper['title']}")
                print(f"   Authors: {', '.join(paper['authors'][:3])}")
                if len(paper['authors']) > 3:
                    print(f"   ... and {len(paper['authors']) - 3} more")
                print(f"   Year: {paper['year']} | Citations: {paper['citations']} | Source: {paper['source']}")
                print(f"   Journal: {paper.get('journal', 'N/A')}")
                print()
            
            if args.save_to_kg:
                count = app.add_papers_to_kg(papers)
                print(f"\nAdded {count} papers to knowledge graph")
        
        elif args.command == 'process-pdf':
            # Process PDF
            result = app.process_pdf(args.pdf_path)
            
            if not result['error']:
                print(f"\n{'='*80}")
                print(f"PDF processed successfully")
                print(f"{'='*80}\n")
                print(f"Text length: {len(result['text'])} characters")
                print(f"Number of chunks: {len(result['chunks'])}")
                print(f"Pages: {result['metadata'].get('num_pages', 'N/A')}")
                
                if args.add_to_chat:
                    doc_id = os.path.basename(args.pdf_path).replace('.pdf', '')
                    app.add_pdf_to_chatbot(args.pdf_path, doc_id)
                    print(f"\nAdded to chatbot as: {doc_id}")
        
        elif args.command == 'chat':
            # Interactive chat
            if args.pdf:
                print("Loading PDFs...")
                for pdf_path in args.pdf:
                    app.add_pdf_to_chatbot(pdf_path)
            
            print("\n" + "="*80)
            print("Research Paper Chat")
            print("="*80)
            print("Type 'exit' to quit, 'clear' to clear history")
            print()
            
            while True:
                try:
                    user_input = input("You: ").strip()
                    
                    if user_input.lower() == 'exit':
                        break
                    
                    if user_input.lower() == 'clear':
                        app.chatbot.clear_history()
                        print("History cleared\n")
                        continue
                    
                    if not user_input:
                        continue
                    
                    response = app.chat(user_input)
                    print(f"\nAssistant: {response}\n")
                
                except KeyboardInterrupt:
                    print("\nExiting...")
                    break
        
        elif args.command == 'kg-search':
            # Search knowledge graph
            if not app.kg:
                print("Knowledge graph not available")
                return
            
            papers = app.kg.search_papers(args.query)
            
            print(f"\n{'='*80}")
            print(f"Found {len(papers)} papers in knowledge graph")
            print(f"{'='*80}\n")
            
            for i, paper in enumerate(papers, 1):
                print(f"{i}. {paper['title']}")
                print(f"   Year: {paper['year']} | Citations: {paper['citations']}")
                print()
        
        elif args.command == 'web':
            # Start web interface
            from web.app import create_app
            
            app_instance = create_app(app)
            app_instance.run(
                host=args.host,
                port=args.port,
                debug=args.debug
            )
    
    finally:
        app.close()


if __name__ == '__main__':
    cli()