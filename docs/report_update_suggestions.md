# VeriSci Report Update Suggestions

Use these corrections while finalizing the report so it matches the current application.

## High-Priority Content Corrections

1. Project title
   Current title says "Multi-Model AI and Collaborative Filtering". The current implementation is stronger as:
   "VeriSci: AI-Based Research Gap Finder using Scopus-Backed Retrieval, RAG and Knowledge Graphs".
   Use this wording if your guide allows title refinement.

2. Remove plagiarism checker references
   The application no longer has a plagiarism checker feature. Replace plagiarism-related wording with:
   "result analysis, research documentation, and project presentation preparation".

3. Update source list
   Mention Scopus/Elsevier and IEEE as verified/primary sources. Mention Semantic Scholar, OpenAlex, ArXiv, CrossRef, PubMed, CORE, and Google Scholar as discovery/open-access sources.

4. Fix vector database wording
   Do not say the system uses FAISS, Pinecone, or ChromaDB. The current MVP uses a lightweight local pickle-based document store with sentence-transformer embeddings and cosine similarity.

5. Add legal full-text limitation
   Deep Research should be described as legal open-access full-text enrichment. It temporarily downloads PDFs only when a legal open PDF is available and deletes the temporary files after extraction.

6. Add workspace storage
   The MVP now includes SQLite workspaces and saved paper libraries. Add this under database design, implementation, and experimental results.

7. Update latency improvements
   Mention parallel source retrieval, local search caching, reduced RAG top-k, and OCR disabled by default as performance improvements.

8. Improve knowledge graph description
   Current graph stores Paper, Author, and Field nodes with AUTHORED and IN_FIELD relationships. Future scope can add Dataset, Method, Metric, Institution, and Research Gap nodes.

## Sections That Needed Filling

- 4.6 SOCME
- 5.3 Important Algorithms
- 5.4 Important Data Structure
- 5.5 GUI Screenshot
- Chapter 7 System Testing
- Chapter 8 Experimental Results
- Chapter 9 Conclusions and Future Scope

## Suggested Screenshots To Add

1. Dashboard workspace screen
2. Search results screen showing Scopus/IEEE and open-access source grouping
3. Paper details panel with DOI, abstract, Scopus verification, and PDF badge
4. Research Gap Finder output
5. Deep Research Gaps output showing full-text/metadata evidence
6. Knowledge Graph visualization or statistics
7. PDF Chat/RAG screen

## Suggested Result Table Columns

Experiment | Query | Sources Selected | Papers Retrieved | Duplicates Removed | Scopus Verified | Open/PDF Available | First Search Time | Cached Search Time | Remarks

