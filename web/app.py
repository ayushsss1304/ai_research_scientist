"""
Complete Flask Web Application with Full Frontend
Backend + Frontend integrated
"""

from flask import Flask, render_template, request, jsonify, session, send_file
from werkzeug.utils import secure_filename
import os
import sys
import json
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import all components
from scrapers.enhanced_paper_scraper import EnhancedPaperScraper
from knowledge_graph.kg_manager import KnowledgeGraphManager
from rag_pipeline.ollama_chatbot import OllamaRAGChatbot
from pdf_processing.pdf_processor import PDFProcessor

# Try to import config
try:
    import config_enhanced as config
except ImportError:
    class config:
        IEEE_API_KEY = ''
        SEMANTIC_SCHOLAR_API_KEY = ''
        ANTHROPIC_API_KEY = ''
        NEO4J_URI = 'bolt://localhost:7687'
        NEO4J_USER = 'neo4j'
        NEO4J_PASSWORD = 'password'
        USER_EMAIL = 'user@example.com'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize components (lazy loading)
scraper = None
kg_manager = None
rag_chatbot = None
pdf_processor = None

def get_scraper():
    global scraper
    if scraper is None:
        scraper = EnhancedPaperScraper(
            semantic_scholar_api_key=config.SEMANTIC_SCHOLAR_API_KEY,
            ieee_api_key=config.IEEE_API_KEY,
            email=config.USER_EMAIL
        )
    return scraper

def get_kg_manager():
    global kg_manager
    if kg_manager is None:
        try:
            kg_manager = KnowledgeGraphManager(
                uri=config.NEO4J_URI,
                user=config.NEO4J_USER,
                password=config.NEO4J_PASSWORD
            )
        except Exception as e:
            logger.warning(f"KG not available: {e}")
    return kg_manager

def get_rag_chatbot():
    global rag_chatbot
    if rag_chatbot is None:
        try:
            rag_chatbot = OllamaRAGChatbot(
                model="llama3",
                storage_dir='rag_storage_web'
            )
        except Exception as e:
            logger.warning(f"RAG not available: {e}")
    return rag_chatbot

def get_pdf_processor():
    global pdf_processor
    if pdf_processor is None:
        pdf_processor = PDFProcessor()
    return pdf_processor

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search_papers():
    """Search papers from all sources"""
    try:
        data = request.json
        query = data.get('query', '')
        max_results = int(data.get('maxResults', 10))
        start_year = int(data['startYear']) if data.get('startYear') else None
        end_year = int(data['endYear']) if data.get('endYear') else None
        min_citations = int(data['minCitations']) if data.get('minCitations') else None
        journal_filter = data.get('journal', '')
        sources = data.get('sources', ['arxiv', 'semantic_scholar', 'openalex', 'crossref', 'pubmed'])
        
        logger.info(f"Search request: {query}")
        
        scraper_instance = get_scraper()
        papers = scraper_instance.search_all(
            query=query,
            max_results_per_source=max_results,
            start_year=start_year,
            end_year=end_year,
            min_citations=min_citations,
            journal_filter=journal_filter if journal_filter else None,
            sources=sources,
            include_abstract=True
        )
        
        # Limit the total results to the exact number requested by the user
        papers = papers[:max_results]
        
        # Store in session
        session['last_search_results'] = papers
        
        return jsonify({
            'success': True,
            'papers': papers,
            'count': len(papers)
        })
    
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/papers/<int:paper_id>/summary', methods=['GET'])
def get_paper_summary(paper_id):
    """Generate summary for a paper"""
    try:
        papers = session.get('last_search_results', [])
        if paper_id >= len(papers):
            return jsonify({'success': False, 'error': 'Paper not found'}), 404
        
        paper = papers[paper_id]
        
        # Generate summary
        summary = {
            'tldr': f"A study on {paper['title'][:100]}...",
            'key_findings': [
                'Finding 1 from the paper',
                'Finding 2 from the paper',
                'Finding 3 from the paper'
            ],
            'methodology': paper.get('abstract', '')[:200] + '...',
            'impact': f"This paper has {paper.get('citations', 0)} citations"
        }
        
        return jsonify({
            'success': True,
            'summary': summary
        })
    
    except Exception as e:
        logger.error(f"Summary error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/kg/add', methods=['POST'])
def add_to_knowledge_graph():
    """Add papers to knowledge graph"""
    try:
        data = request.json
        papers = data.get('papers', [])
        
        if not papers:
            papers = session.get('last_search_results', [])
        
        kg = get_kg_manager()
        if not kg:
            return jsonify({
                'success': False,
                'error': 'Knowledge graph not available'
            }), 500
        
        count = kg.add_papers_batch(papers)
        
        return jsonify({
            'success': True,
            'message': f'Added {count} papers to knowledge graph',
            'count': count
        })
    
    except Exception as e:
        logger.error(f"KG add error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/kg/search', methods=['POST'])
def search_knowledge_graph():
    """Search within knowledge graph"""
    try:
        data = request.json
        query = data.get('query', '')
        limit = int(data.get('limit', 50))
        
        kg = get_kg_manager()
        if not kg:
            return jsonify({
                'success': False,
                'error': 'Knowledge graph not available'
            }), 500
        
        papers = kg.search_papers(query, limit)
        
        return jsonify({
            'success': True,
            'papers': papers,
            'count': len(papers)
        })
    
    except Exception as e:
        logger.error(f"KG search error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/kg/stats', methods=['GET'])
def get_kg_stats():
    """Get knowledge graph statistics"""
    try:
        kg = get_kg_manager()
        if not kg:
            return jsonify({
                'success': False,
                'error': 'Knowledge graph not available'
            }), 500
        
        stats = kg.get_statistics()
        
        return jsonify({
            'success': True,
            'stats': stats
        })
    
    except Exception as e:
        logger.error(f"KG stats error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/kg/all', methods=['GET'])
def get_all_kg_papers():
    """Get all papers from knowledge graph"""
    try:
        kg = get_kg_manager()
        if not kg:
            return jsonify({
                'success': False,
                'error': 'Knowledge graph not available'
            }), 500
        
        limit = int(request.args.get('limit', 100))
        papers = kg.get_all_papers(limit)
        
        return jsonify({
            'success': True,
            'papers': papers,
            'count': len(papers)
        })
    
    except Exception as e:
        logger.error(f"KG all papers error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/kg/visualize', methods=['GET'])
def visualize_knowledge_graph():
    """Get nodes and edges for knowledge graph visualization"""
    try:
        kg = get_kg_manager()
        if not kg:
            return jsonify({
                'success': False,
                'error': 'Knowledge graph not available'
            }), 500
        
        limit = int(request.args.get('limit', 100))
        graph_data = kg.get_graph_data(limit=limit)
        
        return jsonify({
            'success': True,
            'data': graph_data
        })
    
    except Exception as e:
        logger.error(f"KG visualization error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/pdf/upload', methods=['POST'])
def upload_pdf():
    """Upload PDF for RAG — succeeds even if Ollama is unavailable."""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'success': False, 'error': 'Only PDF files are supported'}), 400

        # Save file first — this always succeeds
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        doc_id = filename.replace('.pdf', '')

        # Try to add to RAG (optional)
        chatbot = get_rag_chatbot()
        if chatbot:
            try:
                success = chatbot.add_document_from_file(filepath, doc_id)
                if success:
                    return jsonify({
                        'success': True,
                        'message': f'PDF uploaded and indexed: {filename}',
                        'doc_id': doc_id,
                        'rag_enabled': True
                    })
                else:
                    return jsonify({
                        'success': True,
                        'message': f'PDF saved but indexing failed: {filename}. Try re-uploading.',
                        'doc_id': doc_id,
                        'rag_enabled': False
                    })
            except Exception as rag_err:
                logger.warning(f"RAG indexing failed (non-fatal): {rag_err}")
                return jsonify({
                    'success': True,
                    'message': f'PDF saved. RAG indexing failed: {rag_err}',
                    'doc_id': doc_id,
                    'rag_enabled': False
                })
        else:
            return jsonify({
                'success': True,
                'message': f'PDF saved. Start Ollama to enable AI chat.',
                'doc_id': doc_id,
                'rag_enabled': False
            })

    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/kg/clear', methods=['POST'])
def clear_knowledge_graph():
    """Clear all nodes from the knowledge graph."""
    try:
        kg = get_kg_manager()
        if not kg:
            return jsonify({'success': False, 'error': 'Knowledge graph not available'}), 500
        ok = kg.clear_graph()
        if ok:
            return jsonify({'success': True, 'message': 'Knowledge graph cleared successfully'})
        else:
            return jsonify({'success': False, 'error': 'Failed to clear graph'}), 500
    except Exception as e:
        logger.error(f"KG clear error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/kg/insights', methods=['GET'])
def get_kg_insights():
    """Generate AI insights from the knowledge graph."""
    try:
        kg = get_kg_manager()
        if not kg:
            return jsonify({'success': False, 'error': 'Knowledge graph not available'}), 500

        data = kg.get_insights_data()
        if not data:
            return jsonify({'success': False, 'error': 'No data in knowledge graph yet'}), 400

        # Build structured insight from data (always available)
        structured = {
            'top_authors': data.get('top_authors', []),
            'top_fields': data.get('top_fields', []),
            'top_cited': data.get('top_cited', []),
            'year_dist': data.get('year_dist', []),
            'stats': data.get('stats', {}),
        }

        # Try to get an AI narrative (Ollama optional)
        ai_summary = None
        chatbot = get_rag_chatbot()
        if chatbot:
            try:
                summary_prompt = f"""You are a research analyst. Analyze this knowledge graph data and provide 3 key insights, 2 trends, and 1 emerging research opportunity. Be concise (bullet points).

Top Authors: {data.get('top_authors', [])}
Top Fields: {data.get('top_fields', [])}
Most Cited Papers: {[p['title'] for p in data.get('top_cited', [])]}
Year Distribution: {data.get('year_dist', [])}
Total Papers: {data.get('stats', {}).get('total_papers', 0)}"""
                ai_summary = chatbot.client.generate(summary_prompt, temperature=0.4, max_tokens=600)
            except Exception as e:
                logger.warning(f"Ollama insight generation failed: {e}")

        return jsonify({
            'success': True,
            'structured': structured,
            'ai_summary': ai_summary
        })

    except Exception as e:
        logger.error(f"KG insights error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/research/gaps', methods=['POST'])
def find_research_gaps():
    """Use AI to identify research gaps in the last search results."""
    try:
        data = request.json or {}
        papers = data.get('papers', session.get('last_search_results', []))
        query = data.get('query', '')

        if not papers:
            return jsonify({'success': False, 'error': 'No papers to analyze. Run a search first.'}), 400

        chatbot = get_rag_chatbot()
        if not chatbot:
            return jsonify({
                'success': False,
                'error': 'Ollama not available. Start Ollama to use the Research Gap Finder.'
            }), 503

        # Build a summary of the papers
        paper_summaries = []
        for i, p in enumerate(papers[:15]):  # limit to 15 to keep prompt manageable
            paper_summaries.append(
                f"{i+1}. \"{p.get('title', '')}\" ({p.get('year', '')}) — "
                f"Citations: {p.get('citations', 0)} — Source: {p.get('source', '')}\n"
                f"   Abstract: {(p.get('abstract') or '')[:200]}"
            )

        prompt = f"""You are an expert research analyst specializing in identifying research gaps.

Research Topic: "{query}"

Papers analyzed:
{chr(10).join(paper_summaries)}

Please provide a structured analysis with EXACTLY these 4 sections using markdown:

## 🔍 Research Gaps
List 3 specific gaps or unsolved problems identified across these papers.

## 🚀 Future Research Directions
List 2 promising directions researchers should explore next.

## ⭐ Rising Star Papers
Identify 3 papers from the list that seem most impactful or novel (cite them by title).

## 💡 Novel Hypotheses
Propose 2 testable research hypotheses that could bridge the identified gaps.

Be specific, insightful, and actionable."""

        response = chatbot.client.generate(prompt, temperature=0.5, max_tokens=1200)

        return jsonify({
            'success': True,
            'analysis': response,
            'papers_analyzed': len(paper_summaries)
        })

    except Exception as e:
        logger.error(f"Research gaps error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500



@app.route('/api/rag/documents', methods=['GET'])
def get_rag_documents():
    """Get list of documents in RAG"""
    try:
        chatbot = get_rag_chatbot()
        if not chatbot:
            return jsonify({
                'success': False,
                'error': 'RAG not available'
            }), 500
        
        docs = chatbot.list_documents()
        stats = chatbot.get_statistics()
        
        return jsonify({
            'success': True,
            'documents': docs,
            'stats': stats
        })
    
    except Exception as e:
        logger.error(f"Get docs error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/rag/chat', methods=['POST'])
def rag_chat():
    """Chat with documents"""
    try:
        data = request.json
        message = data.get('message', '')
        
        if not message:
            return jsonify({'success': False, 'error': 'No message provided'}), 400
        
        chatbot = get_rag_chatbot()
        if not chatbot:
            return jsonify({
                'success': False,
                'error': 'RAG not available. Make sure Ollama is running.'
            }), 500
        
        response = chatbot.chat(
            message,
            top_k=5,
            use_reranking=True,
            temperature=0.7
        )
        
        return jsonify({
            'success': True,
            'response': response
        })
    
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/rag/clear', methods=['POST'])
def clear_rag_history():
    """Clear RAG chat history"""
    try:
        chatbot = get_rag_chatbot()
        if chatbot:
            chatbot.clear_history()
        
        return jsonify({
            'success': True,
            'message': 'History cleared'
        })
    
    except Exception as e:
        logger.error(f"Clear history error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/rag/remove/<doc_id>', methods=['DELETE'])
def remove_rag_document(doc_id):
    """Remove document from RAG"""
    try:
        chatbot = get_rag_chatbot()
        if not chatbot:
            return jsonify({'success': False, 'error': 'RAG not available'}), 500
        
        chatbot.remove_document(doc_id)
        
        return jsonify({
            'success': True,
            'message': f'Removed {doc_id}'
        })
    
    except Exception as e:
        logger.error(f"Remove doc error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/export', methods=['POST'])
def export_results():
    """Export search results"""
    try:
        data = request.json
        format_type = data.get('format', 'json')
        papers = data.get('papers', session.get('last_search_results', []))
        
        if format_type == 'json':
            return jsonify({
                'success': True,
                'data': json.dumps(papers, indent=2)
            })
        
        elif format_type == 'csv':
            import csv
            import io
            
            output = io.StringIO()
            if papers:
                writer = csv.DictWriter(output, fieldnames=papers[0].keys())
                writer.writeheader()
                writer.writerows(papers)
            
            return jsonify({
                'success': True,
                'data': output.getvalue()
            })
        
        elif format_type == 'bibtex':
            bibtex = []
            for i, paper in enumerate(papers):
                entry = f"""@article{{paper{i},
    title = {{{paper.get('title', '')}}},
    author = {{{', '.join(paper.get('authors', []))}}},
    year = {{{paper.get('year', '')}}},
    journal = {{{paper.get('journal', '')}}},
    doi = {{{paper.get('doi', '')}}}
}}
"""
                bibtex.append(entry)
            
            return jsonify({
                'success': True,
                'data': '\n'.join(bibtex)
            })
        
        else:
            return jsonify({'success': False, 'error': 'Invalid format'}), 400
    
    except Exception as e:
        logger.error(f"Export error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/trends', methods=['POST'])
def analyze_trends():
    """Analyze research trends"""
    try:
        data = request.json
        papers = data.get('papers', session.get('last_search_results', []))
        
        if not papers:
            return jsonify({'success': False, 'error': 'No papers to analyze'}), 400
        
        # Group by year
        by_year = {}
        for paper in papers:
            year = paper.get('year', 0)
            if year:
                by_year[year] = by_year.get(year, 0) + 1
        
        # Calculate growth
        years = sorted(by_year.keys())
        trend_data = [{'year': year, 'count': by_year[year]} for year in years]
        
        # Simple trend detection
        if len(years) >= 3:
            recent_growth = by_year[years[-1]] - by_year[years[-3]]
            trend = 'growing' if recent_growth > 0 else 'declining'
        else:
            trend = 'insufficient_data'
        
        return jsonify({
            'success': True,
            'trend_data': trend_data,
            'trend': trend,
            'total_papers': len(papers)
        })
    
    except Exception as e:
        logger.error(f"Trends error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/papers/summarize', methods=['POST'])
def summarize_paper():
    """Auto-summarize a paper abstract using Ollama"""
    try:
        data = request.json
        abstract = data.get('abstract', '')
        
        if not abstract:
            return jsonify({'success': False, 'error': 'No abstract provided'}), 400
        
        chatbot = get_rag_chatbot()
        if not chatbot:
            return jsonify({
                'success': False,
                'error': 'Ollama not available. Make sure Ollama is running.'
            }), 500
        
        prompt = f"Summarize this research paper abstract in 2-3 concise sentences. Focus on the key contribution and findings:\n\n{abstract}"
        summary = chatbot.client.generate(prompt, temperature=0.3)
        
        return jsonify({
            'success': True,
            'summary': summary
        })
    
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        'scraper': 'available',
        'kg': 'available' if get_kg_manager() else 'unavailable',
        'rag': 'available' if get_rag_chatbot() else 'unavailable',
        'pdf': 'available'
    }
    
    return jsonify({
        'success': True,
        'status': status,
        'timestamp': datetime.now().isoformat()
    })


# ============================================================================
# HTML TEMPLATE (embedded)
# ============================================================================

@app.route('/template')
def get_template():
    """Return the HTML template"""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Research Scientist</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .tabs {
            display: flex;
            background: #f5f5f5;
            border-bottom: 2px solid #ddd;
        }
        
        .tab {
            flex: 1;
            padding: 20px;
            text-align: center;
            cursor: pointer;
            background: #f5f5f5;
            border: none;
            font-size: 16px;
            font-weight: 600;
            transition: all 0.3s;
        }
        
        .tab:hover {
            background: #e0e0e0;
        }
        
        .tab.active {
            background: white;
            border-bottom: 3px solid #667eea;
        }
        
        .content {
            padding: 30px;
            max-height: calc(100vh - 250px);
            overflow-y: auto;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        
        input[type="text"],
        input[type="number"],
        select,
        textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 14px;
        }
        
        input:focus, select:focus, textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .filter-group {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .checkbox-group {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            margin-top: 10px;
        }
        
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
            margin-right: 10px;
            margin-bottom: 10px;
        }
        
        button:hover {
            transform: translateY(-2px);
        }
        
        button.secondary {
            background: linear-gradient(135deg, #868CFF 0%, #9795f0 100%);
        }
        
        .paper-card {
            background: #f9f9f9;
            border-left: 4px solid #667eea;
            padding: 20px;
            margin-bottom: 15px;
            border-radius: 8px;
        }
        
        .paper-card h3 {
            color: #333;
            margin-bottom: 10px;
        }
        
        .paper-meta {
            color: #666;
            font-size: 14px;
            margin-bottom: 10px;
        }
        
        .paper-abstract {
            color: #555;
            line-height: 1.6;
            margin-top: 10px;
        }
        
        .chat-container {
            display: flex;
            flex-direction: column;
            height: 600px;
            border: 2px solid #ddd;
            border-radius: 12px;
        }
        
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f9f9f9;
        }
        
        .message {
            margin-bottom: 15px;
            padding: 12px 16px;
            border-radius: 12px;
            max-width: 80%;
        }
        
        .message.user {
            background: #667eea;
            color: white;
            margin-left: auto;
        }
        
        .message.assistant {
            background: white;
            border: 2px solid #ddd;
        }
        
        .chat-input-container {
            padding: 20px;
            background: white;
            border-top: 2px solid #ddd;
        }
        
        .chat-input {
            display: flex;
            gap: 10px;
        }
        
        .chat-input textarea {
            flex: 1;
            resize: none;
        }
        
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .alert {
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        
        .alert-success {
            background: #d4edda;
            color: #155724;
        }
        
        .alert-error {
            background: #f8d7da;
            color: #721c24;
        }
        
        .alert-info {
            background: #d1ecf1;
            color: #0c5460;
        }
        
        .file-upload {
            border: 2px dashed #ddd;
            border-radius: 8px;
            padding: 30px;
            text-align: center;
            cursor: pointer;
        }
        
        .file-upload:hover {
            border-color: #667eea;
            background: #f9f9f9;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 AI Research Scientist</h1>
            <p>Search, Analyze, and Chat with Research Papers</p>
        </div>
        
        <div class="tabs">
            <button class="tab active" onclick="showTab('search')">📚 Search</button>
            <button class="tab" onclick="showTab('kg')">🕸️ Knowledge Graph</button>
            <button class="tab" onclick="showTab('rag')">💬 PDF Chat</button>
        </div>
        
        <div class="content">
            <!-- Search Tab -->
            <div id="search" class="tab-content active">
                <h2>Search Research Papers</h2>
                <form id="searchForm">
                    <div class="form-group">
                        <label>Search Query</label>
                        <input type="text" id="query" placeholder="machine learning, deep learning, NLP" required>
                    </div>
                    
                    <div class="form-group">
                        <label>Sources</label>
                        <div class="checkbox-group">
                            <label><input type="checkbox" name="sources" value="arxiv" checked> ArXiv</label>
                            <label><input type="checkbox" name="sources" value="semantic_scholar" checked> Semantic Scholar</label>
                            <label><input type="checkbox" name="sources" value="openalex" checked> OpenAlex</label>
                            <label><input type="checkbox" name="sources" value="crossref"> CrossRef</label>
                            <label><input type="checkbox" name="sources" value="pubmed"> PubMed</label>
                        </div>
                    </div>
                    
                    <div class="filter-group">
                        <div class="form-group">
                            <label>Start Year</label>
                            <input type="number" id="startYear" placeholder="2020">
                        </div>
                        <div class="form-group">
                            <label>End Year</label>
                            <input type="number" id="endYear" placeholder="2024">
                        </div>
                        <div class="form-group">
                            <label>Min Citations</label>
                            <input type="number" id="minCitations" placeholder="10">
                        </div>
                        <div class="form-group">
                            <label>Max Results</label>
                            <input type="number" id="maxResults" value="10">
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label>Journal Filter</label>
                        <input type="text" id="journal" placeholder="IEEE, Nature, Science">
                    </div>
                    
                    <button type="submit">🔍 Search</button>
                    <button type="button" class="secondary" onclick="addToKG()">➕ Add to KG</button>
                    <button type="button" class="secondary" onclick="exportResults()">💾 Export</button>
                </form>
                
                <div id="searchResults"></div>
            </div>
            
            <!-- Knowledge Graph Tab -->
            <div id="kg" class="tab-content">
                <h2>Knowledge Graph</h2>
                <form id="kgSearchForm">
                    <div class="form-group">
                        <label>Search in Graph</label>
                        <input type="text" id="kgQuery" placeholder="Search papers">
                    </div>
                    <button type="submit">🔍 Search</button>
                    <button type="button" class="secondary" onclick="viewAllKG()">📊 View All</button>
                    <button type="button" class="secondary" onclick="loadKGStats()">📈 Stats</button>
                </form>
                
                <div id="kgStats"></div>
                <div id="kgResults"></div>
            </div>
            
            <!-- RAG Tab -->
            <div id="rag" class="tab-content">
                <h2>PDF Chat (Ollama)</h2>
                
                <div class="form-group">
                    <div class="file-upload" onclick="document.getElementById('fileInput').click()">
                        <p>📄 Click to Upload PDFs</p>
                        <input type="file" id="fileInput" multiple accept=".pdf" style="display:none" onchange="uploadPDFs()">
                    </div>
                </div>
                
                <div id="uploadedFiles"></div>
                
                <div class="chat-container">
                    <div id="chatMessages" class="chat-messages"></div>
                    <div class="chat-input-container">
                        <div class="chat-input">
                            <textarea id="chatInput" rows="3" placeholder="Ask about your papers..."></textarea>
                            <button onclick="sendMessage()">Send</button>
                        </div>
                        <button class="secondary" onclick="clearChat()">🗑️ Clear</button>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let searchResults = [];
        
        function showTab(tabName) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById(tabName).classList.add('active');
        }
        
        // Search
        document.getElementById('searchForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const sources = Array.from(document.querySelectorAll('input[name="sources"]:checked')).map(cb => cb.value);
            
            const data = {
                query: document.getElementById('query').value,
                maxResults: document.getElementById('maxResults').value,
                startYear: document.getElementById('startYear').value,
                endYear: document.getElementById('endYear').value,
                minCitations: document.getElementById('minCitations').value,
                journal: document.getElementById('journal').value,
                sources: sources
            };
            
            const resultsDiv = document.getElementById('searchResults');
            resultsDiv.innerHTML = '<div class="alert alert-info"><span class="loading"></span> Searching...</div>';
            
            try {
                const response = await fetch('/api/search', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (result.success) {
                    searchResults = result.papers;
                    displayPapers(result.papers, resultsDiv);
                } else {
                    resultsDiv.innerHTML = `<div class="alert alert-error">${result.error}</div>`;
                }
            } catch (error) {
                resultsDiv.innerHTML = `<div class="alert alert-error">${error.message}</div>`;
            }
        });
        
        function displayPapers(papers, container) {
            if (!papers.length) {
                container.innerHTML = '<div class="alert alert-info">No papers found</div>';
                return;
            }
            
            container.innerHTML = `<div class="alert alert-success">✅ Found ${papers.length} papers</div>`;
            
            papers.forEach((paper, idx) => {
                const card = document.createElement('div');
                card.className = 'paper-card';
                card.innerHTML = `
                    <h3>${paper.title}</h3>
                    <div class="paper-meta">
                        📅 ${paper.year} | 📊 ${paper.citations} citations | 🏷️ ${paper.source}
                        ${paper.journal ? ` | 📖 ${paper.journal}` : ''}
                    </div>
                    <div class="paper-meta">
                        👥 ${paper.authors.slice(0, 3).join(', ')}${paper.authors.length > 3 ? ' et al.' : ''}
                    </div>
                    ${paper.abstract ? `<div class="paper-abstract">${paper.abstract.substring(0, 300)}...</div>` : ''}
                    ${paper.pdf_url ? `<a href="${paper.pdf_url}" target="_blank">📄 PDF</a>` : ''}
                `;
                container.appendChild(card);
            });
        }
        
        async function addToKG() {
            if (!searchResults.length) {
                alert('Search for papers first');
                return;
            }
            
            try {
                const response = await fetch('/api/kg/add', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({papers: searchResults})
                });
                
                const result = await response.json();
                alert(result.success ? `✅ ${result.message}` : `❌ ${result.error}`);
            } catch (error) {
                alert(`❌ ${error.message}`);
            }
        }
        
        async function exportResults() {
            const format = prompt('Export format (json/csv/bibtex):', 'json');
            if (!format) return;
            
            try {
                const response = await fetch('/api/export', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({format, papers: searchResults})
                });
                
                const result = await response.json();
                if (result.success) {
                    const blob = new Blob([result.data], {type: 'text/plain'});
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `papers.${format}`;
                    a.click();
                }
            } catch (error) {
                alert(`❌ ${error.message}`);
            }
        }
        
        // Knowledge Graph
        document.getElementById('kgSearchForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const query = document.getElementById('kgQuery').value;
            const resultsDiv = document.getElementById('kgResults');
            resultsDiv.innerHTML = '<div class="alert alert-info"><span class="loading"></span> Searching...</div>';
            
            try {
                const response = await fetch('/api/kg/search', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({query})
                });
                
                const result = await response.json();
                
                if (result.success) {
                    displayPapers(result.papers, resultsDiv);
                } else {
                    resultsDiv.innerHTML = `<div class="alert alert-error">${result.error}</div>`;
                }
            } catch (error) {
                resultsDiv.innerHTML = `<div class="alert alert-error">${error.message}</div>`;
            }
        });
        
        async function viewAllKG() {
            const resultsDiv = document.getElementById('kgResults');
            resultsDiv.innerHTML = '<div class="alert alert-info"><span class="loading"></span> Loading...</div>';
            
            try {
                const response = await fetch('/api/kg/all');
                const result = await response.json();
                
                if (result.success) {
                    displayPapers(result.papers, resultsDiv);
                } else {
                    resultsDiv.innerHTML = `<div class="alert alert-error">${result.error}</div>`;
                }
            } catch (error) {
                resultsDiv.innerHTML = `<div class="alert alert-error">${error.message}</div>`;
            }
        }
        
        async function loadKGStats() {
            const statsDiv = document.getElementById('kgStats');
            statsDiv.innerHTML = '<div class="alert alert-info"><span class="loading"></span> Loading...</div>';
            
            try {
                const response = await fetch('/api/kg/stats');
                const result = await response.json();
                
                if (result.success) {
                    const stats = result.stats;
                    statsDiv.innerHTML = `
                        <div class="alert alert-success">
                            <strong>Knowledge Graph Statistics:</strong><br>
                            📚 Papers: ${stats.total_papers}<br>
                            👥 Authors: ${stats.total_authors}<br>
                            🏷️ Fields: ${stats.total_fields}<br>
                            📊 Avg Citations: ${stats.avg_citations}
                        </div>
                    `;
                } else {
                    statsDiv.innerHTML = `<div class="alert alert-info">KG not available</div>`;
                }
            } catch (error) {
                statsDiv.innerHTML = `<div class="alert alert-error">${error.message}</div>`;
            }
        }
        
        // RAG
        async function uploadPDFs() {
            const files = document.getElementById('fileInput').files;
            const uploadedDiv = document.getElementById('uploadedFiles');
            
            for (let file of files) {
                const formData = new FormData();
                formData.append('file', file);
                
                const fileDiv = document.createElement('div');
                fileDiv.className = 'alert alert-info';
                fileDiv.textContent = `📄 ${file.name} - Uploading...`;
                uploadedDiv.appendChild(fileDiv);
                
                try {
                    const response = await fetch('/api/pdf/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    fileDiv.className = result.success ? 'alert alert-success' : 'alert alert-error';
                    fileDiv.textContent = `📄 ${file.name} - ${result.success ? '✅ Ready' : '❌ Error'}`;
                } catch (error) {
                    fileDiv.className = 'alert alert-error';
                    fileDiv.textContent = `📄 ${file.name} - ❌ ${error.message}`;
                }
            }
        }
        
        async function sendMessage() {
            const input = document.getElementById('chatInput');
            const message = input.value.trim();
            if (!message) return;
            
            const messagesDiv = document.getElementById('chatMessages');
            
            // User message
            const userMsg = document.createElement('div');
            userMsg.className = 'message user';
            userMsg.textContent = message;
            messagesDiv.appendChild(userMsg);
            
            input.value = '';
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            
            // Loading
            const loadingMsg = document.createElement('div');
            loadingMsg.className = 'message assistant';
            loadingMsg.innerHTML = '<span class="loading"></span> Thinking...';
            messagesDiv.appendChild(loadingMsg);
            
            try {
                const response = await fetch('/api/rag/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message})
                });
                
                const result = await response.json();
                
                loadingMsg.remove();
                
                const assistantMsg = document.createElement('div');
                assistantMsg.className = 'message assistant';
                assistantMsg.textContent = result.success ? result.response : `Error: ${result.error}`;
                messagesDiv.appendChild(assistantMsg);
                
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            } catch (error) {
                loadingMsg.textContent = `Error: ${error.message}`;
            }
        }
        
        async function clearChat() {
            try {
                await fetch('/api/rag/clear', {method: 'POST'});
                document.getElementById('chatMessages').innerHTML = '';
            } catch (error) {
                alert(`Error: ${error.message}`);
            }
        }
        
        document.getElementById('chatInput').addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    </script>
</body>
</html>'''

if __name__ == '__main__':
    print("=" * 80)
    print("AI RESEARCH SCIENTIST - WEB APPLICATION")
    print("=" * 80)
    print("\n🌐 Starting server on http://localhost:5000")
    print("\n📋 Available endpoints:")
    print("  - /                    : Main interface")
    print("  - /api/search          : Search papers")
    print("  - /api/kg/add          : Add to knowledge graph")
    print("  - /api/pdf/upload      : Upload PDFs")
    print("  - /api/rag/chat        : Chat with documents")
    print("\n💡 Make sure Ollama is running for RAG chat!")
    print("=" * 80)
    
    app.run(debug=True, host='0.0.0.0', port=5000)