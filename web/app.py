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
from pdf_processing.pdf_processor import PDFProcessor
from research_gap.deep_research import DeepResearchAnalyzer
from storage.research_store import ResearchStore
from rag_pipeline.openrouter_chatbot import OpenRouterRAGChatbot

# The local RAG stack pulls in PyTorch and a locally running Ollama server.
# Keep it optional so the cloud app can start without those local-only services.
_rag_default = 'false' if os.getenv('VERCEL') else 'true'
if os.getenv('ENABLE_RAG_CHATBOT', _rag_default).lower() == 'true':
    try:
        from rag_pipeline.ollama_chatbot import OllamaRAGChatbot
    except ImportError:
        OllamaRAGChatbot = None
else:
    OllamaRAGChatbot = None

try:
    from accuracy.evaluator import ChatbotAccuracyEvaluator
except ImportError:
    ChatbotAccuracyEvaluator = None

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
        ELSEVIER_API_KEY = ''
        PUBMED_API_KEY = ''
        CORE_API_KEY = ''
        SERPAPI_KEY = ''

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = getattr(config, 'FLASK_SECRET_KEY', None) or os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.config['UPLOAD_FOLDER'] = getattr(config, 'UPLOAD_DIR', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize components (lazy loading)
scraper = None
kg_manager = None
rag_chatbot = None
pdf_processor = None
accuracy_evaluator = None
deep_research_analyzer = None
research_store = None

def get_scraper():
    global scraper
    if scraper is None:
        scraper = EnhancedPaperScraper(
            semantic_scholar_api_key=config.SEMANTIC_SCHOLAR_API_KEY,
            ieee_api_key=config.IEEE_API_KEY,
            elsevier_api_key=getattr(config, 'ELSEVIER_API_KEY', ''),
            pubmed_api_key=getattr(config, 'PUBMED_API_KEY', ''),
            core_api_key=getattr(config, 'CORE_API_KEY', ''),
            serpapi_key=getattr(config, 'SERPAPI_KEY', ''),
            email=config.USER_EMAIL,
            cache_dir=os.path.join(getattr(config, 'CACHE_DIR', 'data/cache'), 'search'),
            cache_ttl=getattr(config, 'CACHE_TTL', 3600),
            max_workers=getattr(config, 'MAX_WORKERS', 5)
        )
    return scraper

def get_kg_manager():
    global kg_manager
    if not getattr(config, 'ENABLE_KNOWLEDGE_GRAPH', True):
        return None
    uri = getattr(config, 'NEO4J_URI', '')
    password = getattr(config, 'NEO4J_PASSWORD', '')
    if not uri or not password:
        return None
    if getattr(config, 'IS_VERCEL', False) and ('localhost' in uri or '127.0.0.1' in uri):
        return None
    if kg_manager is None:
        try:
            kg_manager = KnowledgeGraphManager(
                uri=uri,
                user=config.NEO4J_USER,
                password=password,
                database=getattr(config, 'NEO4J_DATABASE', '')
            )
        except Exception as e:
            logger.warning(f"KG not available: {e}")
    return kg_manager

def get_rag_chatbot():
    global rag_chatbot
    if rag_chatbot is None:
        try:
            storage_dir = os.path.join(getattr(config, 'RUNTIME_DIR', '.'), 'rag_storage_web')
            openrouter_key = os.getenv('OPENROUTER_API_KEY', '')
            if openrouter_key:
                rag_chatbot = OpenRouterRAGChatbot(
                    api_key=openrouter_key,
                    model=os.getenv('OPENROUTER_MODEL', 'google/gemma-4-31b-it:free'),
                    storage_dir=storage_dir,
                    site_url=os.getenv('APP_URL', 'https://ai-research-scientist-rho.vercel.app')
                )
            elif getattr(config, 'ENABLE_RAG_CHATBOT', True) and OllamaRAGChatbot is not None:
                rag_chatbot = OllamaRAGChatbot(
                    model=os.getenv('OLLAMA_MODEL', 'llama3'),
                    ollama_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
                    storage_dir=storage_dir
                )
        except Exception as e:
            logger.warning(f"RAG not available: {e}")
    return rag_chatbot

def get_pdf_processor():
    global pdf_processor
    if pdf_processor is None:
        pdf_processor = PDFProcessor(use_ocr=getattr(config, 'USE_OCR', False))
    return pdf_processor

def get_accuracy_evaluator():
    """Return ChatbotAccuracyEvaluator sharing the RAG embedding manager."""
    global accuracy_evaluator
    if ChatbotAccuracyEvaluator is None:
        return None
    chatbot = get_rag_chatbot()
    emb_mgr = chatbot.embedding_manager if chatbot else None
    if accuracy_evaluator is None:
        accuracy_evaluator = ChatbotAccuracyEvaluator(embedding_manager=emb_mgr)
    return accuracy_evaluator

def get_deep_research_analyzer():
    """Return a lightweight legal PDF analyzer for gap evidence."""
    global deep_research_analyzer
    if deep_research_analyzer is None:
        deep_research_analyzer = DeepResearchAnalyzer(
            email=getattr(config, 'USER_EMAIL', ''),
            max_papers=5,
            max_pdf_mb=20,
            max_chars_per_paper=10000,
            use_ocr=False,
        )
    return deep_research_analyzer

def get_research_store():
    """Return local SQLite storage for workspaces and saved papers."""
    global research_store
    if research_store is None:
        research_store = ResearchStore(getattr(config, 'SQLITE_DB', 'data/research_scientist.db'))
    return research_store

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

def create_app(research_app=None):
    """Return the configured Flask app for CLI/server entry points."""
    return app

@app.route('/api/workspaces', methods=['GET'])
def list_workspaces():
    try:
        store = get_research_store()
        workspaces = store.list_workspaces()
        current_id = session.get('current_workspace_id') or workspaces[0]['id']
        session['current_workspace_id'] = current_id
        return jsonify({
            'success': True,
            'workspaces': workspaces,
            'current_workspace_id': current_id
        })
    except Exception as e:
        logger.error(f"Workspace list error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/workspaces', methods=['POST'])
def create_workspace():
    try:
        data = request.json or {}
        workspace = get_research_store().create_workspace(
            data.get('name', ''),
            data.get('description', '')
        )
        session['current_workspace_id'] = workspace['id']
        return jsonify({'success': True, 'workspace': workspace})
    except Exception as e:
        logger.error(f"Workspace create error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/workspaces/current', methods=['POST'])
def set_current_workspace():
    try:
        data = request.json or {}
        workspace_id = int(data.get('workspace_id'))
        workspace = get_research_store().get_workspace(workspace_id)
        if not workspace:
            return jsonify({'success': False, 'error': 'Workspace not found'}), 404
        session['current_workspace_id'] = workspace_id
        return jsonify({'success': True, 'workspace': workspace})
    except Exception as e:
        logger.error(f"Workspace switch error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/workspaces/<int:workspace_id>/papers', methods=['GET'])
def list_workspace_papers(workspace_id):
    try:
        papers = get_research_store().list_saved_papers(workspace_id)
        stats = get_research_store().dashboard_stats(workspace_id)
        return jsonify({'success': True, 'papers': papers, 'stats': stats})
    except Exception as e:
        logger.error(f"Saved papers list error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/workspaces/<int:workspace_id>/papers', methods=['POST'])
def save_workspace_paper(workspace_id):
    try:
        data = request.json or {}
        saved = get_research_store().save_paper(
            workspace_id,
            data.get('paper', {}),
            data.get('notes', ''),
            data.get('tags', '')
        )
        return jsonify({'success': True, 'paper': saved})
    except Exception as e:
        logger.error(f"Save paper error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/workspaces/<int:workspace_id>/papers/<int:saved_id>', methods=['DELETE'])
def delete_workspace_paper(workspace_id, saved_id):
    try:
        ok = get_research_store().delete_saved_paper(workspace_id, saved_id)
        if not ok:
            return jsonify({'success': False, 'error': 'Saved paper not found'}), 404
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Delete saved paper error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

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
        sources = data.get('sources', ['scopus', 'ieee', 'semantic_scholar', 'openalex'])
        
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
        use_deep_research = bool(data.get('deepResearch'))

        if not papers:
            return jsonify({'success': False, 'error': 'No papers to analyze. Run a search first.'}), 400

        chatbot = get_rag_chatbot()
        if not chatbot:
            return jsonify({
                'success': False,
                'error': 'Ollama not available. Start Ollama to use the Research Gap Finder.'
            }), 503

        deep_evidence = []
        if use_deep_research:
            deep_evidence = get_deep_research_analyzer().analyze(papers)

        deep_by_key = {
            (item.get('doi') or item.get('title') or '').lower(): item
            for item in deep_evidence
        }

        # Build enriched summaries. Gap detection is weak on titles alone, so
        # include every available abstract, keyword, identifier, and source cue.
        paper_summaries = []
        for i, p in enumerate(papers[:15]):  # limit to 15 to keep prompt manageable
            deep = deep_by_key.get((p.get('doi') or p.get('title') or '').lower(), {})
            abstract = (p.get('abstract') or '').strip()
            if not abstract and deep.get('abstract'):
                abstract = deep.get('abstract', '')
            keywords = p.get('keywords') or p.get('fields') or p.get('categories') or []
            if isinstance(keywords, str):
                keywords = [keywords]
            subject_areas = p.get('subject_areas') or []
            if isinstance(subject_areas, str):
                subject_areas = [subject_areas]
            affiliations = p.get('affiliations') or []
            countries = sorted({
                a.get('country', '')
                for a in affiliations
                if isinstance(a, dict) and a.get('country')
            })
            metrics = p.get('source_metrics') or {}
            evidence_level = deep.get('evidence_level') or ('abstract/metadata' if abstract else 'title and bibliographic metadata only')
            snippets = deep.get('evidence_snippets') or []
            paper_summaries.append(
                f"{i+1}. \"{p.get('title', '')}\" ({p.get('year', '')}) — "
                f"Citations: {p.get('citations', 0)} — Source: {p.get('source', '')}\n"
                f"   DOI: {p.get('doi', '')}; EID/Paper ID: {p.get('eid') or p.get('paper_id', '')}; "
                f"Document type: {p.get('document_type', '')}\n"
                f"   Keywords/fields: {', '.join(keywords[:8])}\n"
                f"   Subject areas: {', '.join(subject_areas[:6])}\n"
                f"   Affiliation countries: {', '.join(countries[:6])}\n"
                f"   Journal metrics: CiteScore={metrics.get('cite_score', '')}, "
                f"SJR={metrics.get('sjr', '')}, SNIP={metrics.get('snip', '')}\n"
                f"   Evidence level: {evidence_level}\n"
                f"   Full text status: {deep.get('full_text_status', 'Not attempted')}\n"
                f"   Abstract: {abstract[:900] if abstract else 'Not available'}\n"
                f"   Deep evidence snippets: {' | '.join(snippets[:3]) if snippets else 'None'}"
            )

        prompt = f"""You are an expert research analyst specializing in identifying research gaps.

Research Topic: "{query}"

Papers analyzed:
{chr(10).join(paper_summaries)}

Rules:
- Do not infer specific methodology/results from title-only papers.
- Prefer gaps supported by abstracts, keywords, Scopus metadata, citation patterns, or repeated missing information.
- For every gap, cite the paper titles that support it.
- Add a confidence label: High, Medium, or Low.
- Prefer full_text evidence snippets when they are present.

Please provide a structured analysis with EXACTLY these 5 sections using markdown:

## 🔍 Research Gaps
List 3 specific gaps or unsolved problems. Each gap must include Evidence and Confidence.

## 🚀 Future Research Directions
List 2 promising directions researchers should explore next.

## ⭐ Rising Star Papers
Identify 3 papers from the list that seem most impactful or novel (cite them by title).

## 💡 Novel Hypotheses
Propose 2 testable research hypotheses that could bridge the identified gaps.

## Evidence Limitations
Briefly state whether the analysis is based on full abstracts, Scopus metadata, or title-only records.

Be specific, insightful, and actionable."""

        response = chatbot.client.generate(prompt, temperature=0.5, max_tokens=1200)

        return jsonify({
            'success': True,
            'analysis': response,
            'papers_analyzed': len(paper_summaries),
            'deep_research': use_deep_research,
            'deep_evidence': deep_evidence
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
                'error': 'AI chat is not configured.'
            }), 500
        
        response = chatbot.chat(
            message,
            top_k=getattr(config, 'RAG_TOP_K', 3),
            use_reranking=True,
            temperature=0.5
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
    """Auto-summarize a paper abstract using the configured AI provider."""
    try:
        data = request.json
        abstract = data.get('abstract', '')
        
        if not abstract:
            return jsonify({'success': False, 'error': 'No abstract provided'}), 400
        
        chatbot = get_rag_chatbot()
        if not chatbot:
            return jsonify({
                'success': False,
                'error': 'AI provider is not configured.'
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

@app.route('/api/accuracy/evaluate', methods=['POST'])
def evaluate_accuracy():
    """
    Batch evaluate chatbot accuracy against reference answers.

    Request body:
        {
          "test_cases": [
            {"question": str, "expected": str, "actual": str (optional)},
            ...
          ],
          "use_chatbot": bool   // if true and actual is missing, call chatbot
        }
    """
    try:
        data       = request.json or {}
        test_cases = data.get('test_cases', [])
        use_chatbot = data.get('use_chatbot', True)

        if not test_cases:
            return jsonify({'success': False, 'error': 'No test cases provided'}), 400

        evaluator = get_accuracy_evaluator()
        if evaluator is None:
            return jsonify({
                'success': False,
                'error': 'Accuracy evaluation is disabled in this deployment.'
            }), 503
        chatbot   = get_rag_chatbot() if use_chatbot else None

        report = evaluator.evaluate(test_cases=test_cases, chatbot=chatbot)

        return jsonify({'success': True, 'report': report})

    except Exception as e:
        logger.error(f"Accuracy evaluation error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/accuracy/quick', methods=['POST'])
def quick_accuracy_check():
    """
    Quick single-question accuracy check (no chatbot call needed).
    Compares a user-supplied actual answer vs expected answer.

    Request body: {"question": str, "expected": str, "actual": str}
    """
    try:
        data     = request.json or {}
        question = data.get('question', '').strip()
        expected = data.get('expected', '').strip()
        actual   = data.get('actual',   '').strip()

        if not expected or not actual:
            return jsonify({'success': False, 'error': 'Both expected and actual answers are required'}), 400

        evaluator = get_accuracy_evaluator()
        report    = evaluator.evaluate(
            test_cases=[{'question': question, 'expected': expected, 'actual': actual}],
            chatbot=None
        )

        result = report['results'][0] if report['results'] else {}
        return jsonify({'success': True, 'result': result, 'summary': report['summary']})

    except Exception as e:
        logger.error(f"Quick accuracy check error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


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
