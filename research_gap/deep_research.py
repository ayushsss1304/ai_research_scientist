"""
Lightweight legal full-text enrichment for research-gap analysis.

Downloads only open-access PDFs into temporary folders, extracts compact
evidence, and deletes the PDF immediately after processing.
"""

import logging
import os
import re
import shutil
import tempfile
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests

from pdf_processing.pdf_processor import PDFProcessor

logger = logging.getLogger(__name__)


class DeepResearchAnalyzer:
    """Resolve legal PDFs, extract compact evidence, and clean up temp files."""

    def __init__(
        self,
        email: str = "",
        max_papers: int = 6,
        max_pdf_mb: int = 25,
        max_chars_per_paper: int = 12000,
        use_ocr: bool = False,
    ):
        self.email = email
        self.max_papers = max_papers
        self.max_pdf_bytes = max_pdf_mb * 1024 * 1024
        self.max_chars_per_paper = max_chars_per_paper
        self.pdf_processor = PDFProcessor(use_ocr=use_ocr)

    def analyze(self, papers: List[Dict]) -> List[Dict]:
        """Return compact deep-research evidence for the first few papers."""
        enriched = []
        for paper in papers[: self.max_papers]:
            evidence = self._metadata_evidence(paper)
            pdf_url = self.resolve_open_pdf_url(paper)

            if not pdf_url:
                evidence["evidence_level"] = "metadata_only"
                evidence["full_text_status"] = "No legal open-access PDF found"
                enriched.append(evidence)
                continue

            temp_dir = tempfile.mkdtemp(prefix="verisci_pdf_")
            try:
                pdf_path = self._download_pdf(pdf_url, temp_dir)
                text = self.pdf_processor.extract_text_from_pdf(pdf_path, use_ocr=False)
                text = self._compact_text(text)
                sections = self.extract_sections(text)

                evidence.update({
                    "evidence_level": "full_text",
                    "full_text_status": f"Downloaded temporarily from {self._domain(pdf_url)}",
                    "pdf_source": pdf_url,
                    "sections": sections,
                    "evidence_snippets": self._evidence_snippets(sections, text),
                    "text_characters_used": len(text),
                })
            except Exception as exc:
                logger.info(f"Deep research fallback for {paper.get('title', '')[:60]}: {exc}")
                evidence["evidence_level"] = "metadata_only"
                evidence["full_text_status"] = f"PDF unavailable or skipped: {exc}"
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

            enriched.append(evidence)

        return enriched

    def resolve_open_pdf_url(self, paper: Dict) -> str:
        """Find a legal open PDF URL from existing metadata or free OA APIs."""
        for candidate in [
            paper.get("pdf_url", ""),
            paper.get("open_access_url", ""),
        ]:
            if self._looks_like_pdf(candidate):
                return candidate

        doi = (paper.get("doi") or "").strip()
        if doi:
            for resolver in (self._openalex_pdf_by_doi, self._unpaywall_pdf_by_doi):
                url = resolver(doi)
                if self._looks_like_pdf(url):
                    return url

        arxiv_id = paper.get("arxiv_id", "")
        if arxiv_id:
            return f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        return ""

    def extract_sections(self, text: str) -> Dict[str, str]:
        """Extract compact sections useful for gap analysis."""
        sections = {}
        patterns = {
            "abstract": r"\babstract\b",
            "introduction": r"\b(1\.?\s*)?introduction\b",
            "methodology": r"\b(methodology|methods|materials and methods|approach)\b",
            "results": r"\b(results|experiments|evaluation)\b",
            "limitations": r"\b(limitations?|threats to validity)\b",
            "future_work": r"\b(future work|future directions)\b",
            "conclusion": r"\b(conclusion|conclusions)\b",
        }

        lowered = text.lower()
        matches = []
        for name, pattern in patterns.items():
            match = re.search(pattern, lowered)
            if match:
                matches.append((match.start(), name))

        matches.sort()
        for idx, (start, name) in enumerate(matches):
            end = matches[idx + 1][0] if idx + 1 < len(matches) else min(len(text), start + 3500)
            section_text = text[start:end].strip()
            if section_text:
                sections[name] = section_text[:2500]

        if not sections and text:
            sections["available_text"] = text[:3500]

        return sections

    def _metadata_evidence(self, paper: Dict) -> Dict:
        return {
            "title": paper.get("title", ""),
            "year": paper.get("year", ""),
            "source": paper.get("source", ""),
            "doi": paper.get("doi", ""),
            "eid": paper.get("eid") or paper.get("paper_id", ""),
            "citations": paper.get("citations", 0),
            "journal": paper.get("journal", ""),
            "document_type": paper.get("document_type", ""),
            "keywords": paper.get("keywords") or paper.get("fields") or paper.get("categories") or [],
            "subject_areas": paper.get("subject_areas") or [],
            "abstract": (paper.get("abstract") or "")[:1500],
            "scopus_verified": bool(paper.get("scopus_verified")),
            "source_metrics": paper.get("source_metrics") or {},
        }

    def _download_pdf(self, pdf_url: str, temp_dir: str) -> str:
        path = os.path.join(temp_dir, "paper.pdf")
        headers = {"User-Agent": self._user_agent()}

        with requests.get(pdf_url, headers=headers, stream=True, timeout=35, allow_redirects=True) as response:
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            content_length = int(response.headers.get("Content-Length", 0) or 0)

            if content_length and content_length > self.max_pdf_bytes:
                raise ValueError("PDF is larger than the configured size limit")

            if "pdf" not in content_type and not self._looks_like_pdf(str(response.url)):
                raise ValueError("URL did not return a PDF")

            downloaded = 0
            with open(path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 256):
                    if not chunk:
                        continue
                    downloaded += len(chunk)
                    if downloaded > self.max_pdf_bytes:
                        raise ValueError("PDF exceeded the configured size limit")
                    handle.write(chunk)

        return path

    def _openalex_pdf_by_doi(self, doi: str) -> str:
        try:
            response = requests.get(
                f"https://api.openalex.org/works/doi:{doi}",
                headers={"User-Agent": self._user_agent()},
                timeout=20,
            )
            response.raise_for_status()
            data = response.json()
            locations = data.get("locations") or []
            for location in locations:
                source_url = location.get("pdf_url") or location.get("landing_page_url") or ""
                if self._looks_like_pdf(source_url):
                    return source_url
            return data.get("open_access", {}).get("oa_url", "")
        except Exception:
            return ""

    def _unpaywall_pdf_by_doi(self, doi: str) -> str:
        if not self.email:
            return ""
        try:
            response = requests.get(
                f"https://api.unpaywall.org/v2/{doi}",
                params={"email": self.email},
                timeout=20,
            )
            response.raise_for_status()
            data = response.json()
            best = data.get("best_oa_location") or {}
            return best.get("url_for_pdf") or best.get("url") or ""
        except Exception:
            return ""

    def _compact_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text or "").strip()
        return text[: self.max_chars_per_paper]

    def _evidence_snippets(self, sections: Dict[str, str], text: str) -> List[str]:
        snippets = []
        for name in ["limitations", "future_work", "conclusion", "abstract", "methodology", "results"]:
            if sections.get(name):
                snippets.append(f"{name}: {sections[name][:900]}")
        if not snippets and text:
            snippets.append(text[:900])
        return snippets[:4]

    def _looks_like_pdf(self, url: Optional[str]) -> bool:
        if not url:
            return False
        parsed = urlparse(url)
        return parsed.scheme in {"http", "https"} and (
            parsed.path.lower().endswith(".pdf") or "pdf" in parsed.path.lower()
        )

    def _domain(self, url: str) -> str:
        return urlparse(url).netloc or "open web"

    def _user_agent(self) -> str:
        if self.email:
            return f"VeriSciResearchBot/1.0 (mailto:{self.email})"
        return "VeriSciResearchBot/1.0"
