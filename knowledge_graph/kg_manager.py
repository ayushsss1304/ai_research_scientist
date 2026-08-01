from neo4j import GraphDatabase
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class _DatabaseBoundDriver:
    """Delegate to a Neo4j driver while applying the Aura database by default."""

    def __init__(self, driver, database: Optional[str] = None):
        self._driver = driver
        self._database = database or None

    def session(self, *args, **kwargs):
        if self._database and 'database' not in kwargs:
            kwargs['database'] = self._database
        return self._driver.session(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._driver, name)


class KnowledgeGraphManager:
    """Manages Neo4j knowledge graph for research papers"""
    
    def __init__(self, uri: str, user: str, password: str,
                 database: Optional[str] = None):
        """Initialize Neo4j connection"""
        try:
            raw_driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver = _DatabaseBoundDriver(raw_driver, database)
            self._verify_connectivity()
            self._create_indexes()
            logger.info("Successfully connected to Neo4j")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def _verify_connectivity(self):
        """Verify database connectivity"""
        with self.driver.session() as session:
            session.run("RETURN 1")
    
    def _create_indexes(self):
        """Create indexes for better query performance"""
        with self.driver.session() as session:
            # Create indexes
            session.run("""
                CREATE INDEX paper_title IF NOT EXISTS 
                FOR (p:Paper) ON (p.title)
            """)
            session.run("""
                CREATE INDEX author_name IF NOT EXISTS 
                FOR (a:Author) ON (a.name)
            """)
            session.run("""
                CREATE INDEX paper_year IF NOT EXISTS 
                FOR (p:Paper) ON (p.year)
            """)
    
    def close(self):
        """Close database connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def add_paper(self, paper: Dict) -> bool:
        """Add a paper to the knowledge graph"""
        try:
            with self.driver.session() as session:
                session.execute_write(self._create_paper, paper)
            logger.info(f"Added paper: {paper.get('title', 'Unknown')[:50]}")
            return True
        except Exception as e:
            logger.error(f"Error adding paper: {e}")
            return False
    
    @staticmethod
    def _create_paper(tx, paper):
        """Transaction function to create paper node"""
        query = """
        MERGE (p:Paper {title: $title})
        SET p.abstract = $abstract,
            p.year = $year,
            p.published_date = $published_date,
            p.pdf_url = $pdf_url,
            p.source = $source,
            p.citations = $citations,
            p.journal = $journal,
            p.doi = $doi,
            p.url = $url,
            p.paper_id = $paper_id
        
        WITH p
        UNWIND $authors AS author_name
        MERGE (a:Author {name: author_name})
        MERGE (a)-[:AUTHORED]->(p)
        
        WITH p
        UNWIND $fields AS field_name
        MERGE (f:Field {name: field_name})
        MERGE (p)-[:IN_FIELD]->(f)
        
        RETURN p
        """
        
        tx.run(query,
               title=paper.get('title', ''),
               abstract=paper.get('abstract', ''),
               year=paper.get('year', 0),
               published_date=paper.get('published_date', ''),
               pdf_url=paper.get('pdf_url', ''),
               source=paper.get('source', ''),
               citations=paper.get('citations', 0),
               journal=paper.get('journal', ''),
               doi=paper.get('doi', ''),
               url=paper.get('url', ''),
               paper_id=paper.get('paper_id', ''),
               authors=paper.get('authors', []),
               fields=paper.get('fields', []) or paper.get('categories', []))
    
    def add_papers_batch(self, papers: List[Dict]) -> int:
        """Add multiple papers in batch"""
        success_count = 0
        for paper in papers:
            if self.add_paper(paper):
                success_count += 1
        
        logger.info(f"Successfully added {success_count}/{len(papers)} papers")
        return success_count
    
    def get_all_papers(self, limit: int = 100) -> List[Dict]:
        """Retrieve all papers from the knowledge graph"""
        try:
            with self.driver.session() as session:
                result = session.execute_read(self._get_papers, limit)
            return result
        except Exception as e:
            logger.error(f"Error retrieving papers: {e}")
            return []
    
    @staticmethod
    def _get_papers(tx, limit):
        """Transaction function to get papers"""
        query = """
        MATCH (p:Paper)
        OPTIONAL MATCH (a:Author)-[:AUTHORED]->(p)
        OPTIONAL MATCH (p)-[:IN_FIELD]->(f:Field)
        RETURN p, 
               collect(DISTINCT a.name) as authors,
               collect(DISTINCT f.name) as fields
        ORDER BY p.citations DESC, p.year DESC
        LIMIT $limit
        """
        
        result = tx.run(query, limit=limit)
        papers = []
        
        for record in result:
            paper_node = record['p']
            papers.append({
                'title': paper_node.get('title', ''),
                'abstract': paper_node.get('abstract', ''),
                'year': paper_node.get('year', 0),
                'published_date': paper_node.get('published_date', ''),
                'pdf_url': paper_node.get('pdf_url', ''),
                'source': paper_node.get('source', ''),
                'citations': paper_node.get('citations', 0),
                'journal': paper_node.get('journal', ''),
                'doi': paper_node.get('doi', ''),
                'url': paper_node.get('url', ''),
                'authors': record['authors'],
                'fields': record['fields']
            })
        
        return papers
    
    def search_papers(self, query: str, limit: int = 50) -> List[Dict]:
        """Search papers in the knowledge graph"""
        try:
            with self.driver.session() as session:
                result = session.execute_read(self._search_papers, query, limit)
            return result
        except Exception as e:
            logger.error(f"Error searching papers: {e}")
            return []
    
    @staticmethod
    def _search_papers(tx, query, limit):
        """Transaction function to search papers"""
        cypher_query = """
        MATCH (p:Paper)
        WHERE toLower(p.title) CONTAINS toLower($query)
           OR toLower(p.abstract) CONTAINS toLower($query)
           OR toLower(p.journal) CONTAINS toLower($query)
        OPTIONAL MATCH (a:Author)-[:AUTHORED]->(p)
        OPTIONAL MATCH (p)-[:IN_FIELD]->(f:Field)
        RETURN p, 
               collect(DISTINCT a.name) as authors,
               collect(DISTINCT f.name) as fields
        ORDER BY p.citations DESC
        LIMIT $limit
        """
        
        result = tx.run(cypher_query, query=query, limit=limit)
        papers = []
        
        for record in result:
            paper_node = record['p']
            papers.append({
                'title': paper_node.get('title', ''),
                'abstract': paper_node.get('abstract', ''),
                'year': paper_node.get('year', 0),
                'citations': paper_node.get('citations', 0),
                'journal': paper_node.get('journal', ''),
                'source': paper_node.get('source', ''),
                'pdf_url': paper_node.get('pdf_url', ''),
                'authors': record['authors'],
                'fields': record['fields']
            })
        
        return papers
    
    def get_author_papers(self, author_name: str) -> List[Dict]:
        """Get all papers by a specific author"""
        try:
            with self.driver.session() as session:
                result = session.execute_read(self._get_author_papers, author_name)
            return result
        except Exception as e:
            logger.error(f"Error getting author papers: {e}")
            return []
    
    @staticmethod
    def _get_author_papers(tx, author_name):
        """Transaction function to get author's papers"""
        query = """
        MATCH (a:Author {name: $author_name})-[:AUTHORED]->(p:Paper)
        OPTIONAL MATCH (other:Author)-[:AUTHORED]->(p)
        WHERE other.name <> $author_name
        RETURN p, collect(DISTINCT other.name) as coauthors
        ORDER BY p.year DESC
        """
        
        result = tx.run(query, author_name=author_name)
        papers = []
        
        for record in result:
            paper_node = record['p']
            papers.append({
                'title': paper_node.get('title', ''),
                'year': paper_node.get('year', 0),
                'citations': paper_node.get('citations', 0),
                'coauthors': record['coauthors']
            })
        
        return papers
    
    def get_coauthors(self, author_name: str) -> List[Dict]:
        """Find co-authors of a given author"""
        try:
            with self.driver.session() as session:
                result = session.execute_read(self._get_coauthors, author_name)
            return result
        except Exception as e:
            logger.error(f"Error getting co-authors: {e}")
            return []
    
    @staticmethod
    def _get_coauthors(tx, author_name):
        """Transaction function to get co-authors"""
        query = """
        MATCH (a1:Author {name: $author_name})-[:AUTHORED]->(p:Paper)<-[:AUTHORED]-(a2:Author)
        WHERE a1 <> a2
        RETURN a2.name as coauthor, count(p) as papers_together
        ORDER BY papers_together DESC
        """
        
        result = tx.run(query, author_name=author_name)
        coauthors = []
        
        for record in result:
            coauthors.append({
                'name': record['coauthor'],
                'papers_together': record['papers_together']
            })
        
        return coauthors
    
    def get_papers_by_year(self) -> Dict[int, int]:
        """Get paper count grouped by year"""
        try:
            with self.driver.session() as session:
                result = session.execute_read(self._get_papers_by_year)
            return result
        except Exception as e:
            logger.error(f"Error getting papers by year: {e}")
            return {}
    
    @staticmethod
    def _get_papers_by_year(tx):
        """Transaction function to get papers by year"""
        query = """
        MATCH (p:Paper)
        WHERE p.year > 0
        RETURN p.year as year, count(p) as count
        ORDER BY year DESC
        """
        
        result = tx.run(query)
        years = {}
        
        for record in result:
            years[record['year']] = record['count']
        
        return years
    
    def get_top_cited_papers(self, limit: int = 10) -> List[Dict]:
        """Get most cited papers"""
        try:
            with self.driver.session() as session:
                result = session.execute_read(self._get_top_cited, limit)
            return result
        except Exception as e:
            logger.error(f"Error getting top cited papers: {e}")
            return []
    
    @staticmethod
    def _get_top_cited(tx, limit):
        """Transaction function to get top cited papers"""
        query = """
        MATCH (p:Paper)
        WHERE p.citations > 0
        OPTIONAL MATCH (a:Author)-[:AUTHORED]->(p)
        RETURN p.title as title,
               p.citations as citations,
               p.year as year,
               collect(a.name) as authors
        ORDER BY citations DESC
        LIMIT $limit
        """
        
        result = tx.run(query, limit=limit)
        papers = []
        
        for record in result:
            papers.append({
                'title': record['title'],
                'citations': record['citations'],
                'year': record['year'],
                'authors': record['authors']
            })
        
        return papers
    
    def get_statistics(self) -> Dict:
        """Get overall knowledge graph statistics"""
        try:
            with self.driver.session() as session:
                stats = session.execute_read(self._get_statistics)
            return stats
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    @staticmethod
    def _get_statistics(tx):
        """Transaction function to get statistics"""
        query = """
        MATCH (p:Paper)
        OPTIONAL MATCH (a:Author)
        OPTIONAL MATCH (f:Field)
        RETURN count(DISTINCT p) as total_papers,
               count(DISTINCT a) as total_authors,
               count(DISTINCT f) as total_fields,
               avg(p.citations) as avg_citations,
               max(p.citations) as max_citations
        """
        
        result = tx.run(query)
        record = result.single()
        
        return {
            'total_papers': record['total_papers'],
            'total_authors': record['total_authors'],
            'total_fields': record['total_fields'],
            'avg_citations': round(record['avg_citations'], 2) if record['avg_citations'] else 0,
            'max_citations': record['max_citations']
        }
    
    def get_graph_data(self, limit: int = 100) -> Dict:
        """Get graph structure (nodes and edges) for visualization"""
        try:
            with self.driver.session() as session:
                result = session.execute_read(self._get_graph_data, limit)
            return result
        except Exception as e:
            logger.error(f"Error getting graph data: {e}")
            return {"nodes": [], "edges": []}
            
    @staticmethod
    def _get_graph_data(tx, limit):
        """Transaction function to get graph nodes and edges"""
        query = """
        MATCH (p:Paper)
        WITH p LIMIT $limit
        OPTIONAL MATCH (a:Author)-[r1:AUTHORED]->(p)
        OPTIONAL MATCH (p)-[r2:IN_FIELD]->(f:Field)
        
        RETURN 
            collect(DISTINCT {id: id(p), label: left(p.title, 30) + '...', title: p.title, group: 'Paper', value: coalesce(p.citations, 1)}) as paper_nodes,
            collect(DISTINCT {id: id(a), label: a.name, group: 'Author'}) as author_nodes,
            collect(DISTINCT {id: id(f), label: f.name, group: 'Field'}) as field_nodes,
            collect(DISTINCT {from: id(a), to: id(p), label: 'AUTHORED'}) as authored_edges,
            collect(DISTINCT {from: id(p), to: id(f), label: 'IN_FIELD'}) as field_edges
        """
        
        result = tx.run(query, limit=limit)
        record = result.single()
        
        if not record:
            return {"nodes": [], "edges": []}
            
        nodes = []
        # Add papers
        for n in record['paper_nodes']:
            if n['id'] is not None:
                nodes.append(n)
        
        # Add authors
        for n in record['author_nodes']:
            if n['id'] is not None:
                nodes.append(n)
                
        # Add fields
        for n in record['field_nodes']:
            if n['id'] is not None:
                nodes.append(n)
                
        # Deduplicate nodes by ID
        unique_nodes = {n['id']: n for n in nodes}.values()
                
        edges = []
        # Add authored edges
        for e in record['authored_edges']:
            if e['from'] is not None and e['to'] is not None:
                edges.append(e)
                
        # Add field edges
        for e in record['field_edges']:
            if e['from'] is not None and e['to'] is not None:
                edges.append(e)
                
        return {
            "nodes": list(unique_nodes),
            "edges": edges
        }

    def clear_graph(self) -> bool:
        """Delete all nodes and relationships from the knowledge graph."""
        try:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            logger.info("Knowledge graph cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing graph: {e}")
            return False

    def get_insights_data(self) -> Dict:
        """Retrieve raw data for AI insight generation."""
        insights = {}
        try:
            with self.driver.session() as session:
                # Top 5 authors by paper count
                result = session.run("""
                    MATCH (a:Author)-[:AUTHORED]->(p:Paper)
                    RETURN a.name AS author, count(p) AS papers
                    ORDER BY papers DESC LIMIT 5
                """)
                insights['top_authors'] = [
                    {'name': r['author'], 'papers': r['papers']} for r in result
                ]

                # Top 5 research fields
                result = session.run("""
                    MATCH (p:Paper)-[:IN_FIELD]->(f:Field)
                    RETURN f.name AS field, count(p) AS papers
                    ORDER BY papers DESC LIMIT 5
                """)
                insights['top_fields'] = [
                    {'name': r['field'], 'papers': r['papers']} for r in result
                ]

                # Top 5 most-cited papers
                result = session.run("""
                    MATCH (p:Paper)
                    WHERE p.citations IS NOT NULL AND p.citations > 0
                    RETURN p.title AS title, p.citations AS citations, p.year AS year, p.source AS source
                    ORDER BY p.citations DESC LIMIT 5
                """)
                insights['top_cited'] = [
                    {'title': r['title'], 'citations': r['citations'],
                     'year': r['year'], 'source': r['source']} for r in result
                ]

                # Year distribution
                result = session.run("""
                    MATCH (p:Paper)
                    WHERE p.year IS NOT NULL AND p.year > 2000
                    RETURN p.year AS year, count(p) AS papers
                    ORDER BY year
                """)
                insights['year_dist'] = [
                    {'year': r['year'], 'papers': r['papers']} for r in result
                ]

                # Total stats
                stats = self.get_statistics()
                insights['stats'] = stats

        except Exception as e:
            logger.error(f"Error getting insights data: {e}")
        return insights
