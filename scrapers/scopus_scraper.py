"""
Scopus / Elsevier scraper.

Uses Elsevier's Scopus Search API for verified Scopus-indexed records and,
when available, the Abstract Retrieval API to enrich records for research-gap
analysis.
"""

import logging
import re
import time
from typing import Dict, List, Optional

import requests

from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class ScopusScraper(BaseScraper):
    """Scraper for Scopus-indexed papers through Elsevier APIs."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or ""
        self.search_url = "https://api.elsevier.com/content/search/scopus"
        self.abstract_eid_url = "https://api.elsevier.com/content/abstract/eid"
        self.abstract_doi_url = "https://api.elsevier.com/content/abstract/doi"
        self.serial_issn_url = "https://api.elsevier.com/content/serial/title/issn"
        self.headers = {
            "Accept": "application/json",
            "X-ELS-APIKey": self.api_key,
        }

    def search(
        self,
        query: str,
        max_results: int = 10,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        min_citations: Optional[int] = None,
        **kwargs,
    ) -> List[Dict]:
        """Search Scopus for papers."""
        if not self.api_key:
            logger.warning("Elsevier API key required for Scopus - skipping")
            return []

        scopus_query = self._build_query(query, start_year, end_year, min_citations)
        papers: List[Dict] = []
        start = 0
        page_size = min(max_results, 25)

        while len(papers) < max_results:
            try:
                entries = self._search_entries(
                    query=scopus_query,
                    start=start,
                    count=min(page_size, max_results - len(papers)),
                )
                if not entries:
                    break

                for entry in entries:
                    paper = self._parse_search_entry(entry)
                    if paper and paper.get("eid"):
                        details = self.get_paper_details(eid=paper["eid"], doi=paper.get("doi", ""))
                        paper.update({k: v for k, v in details.items() if v})
                    if paper and paper.get("issn"):
                        source_metrics = self.get_source_metrics(paper["issn"])
                        if source_metrics:
                            paper["source_metrics"] = source_metrics
                    if paper:
                        papers.append(paper)

                if len(entries) < page_size:
                    break

                start += len(entries)
                time.sleep(0.25)

            except Exception as exc:
                logger.error(f"Scopus search error: {exc}")
                break

        return self._apply_filters(
            papers[:max_results],
            start_year=start_year,
            end_year=end_year,
            min_citations=min_citations,
        )

    def _search_entries(self, query: str, start: int, count: int) -> List[Dict]:
        rich_fields = [
            "dc:identifier", "dc:title", "dc:description", "dc:creator",
            "author", "affiliation", "prism:publicationName", "prism:coverDate",
            "prism:doi", "prism:issn", "prism:isbn", "citedby-count", "eid",
            "prism:url", "subtype", "subtypeDescription", "openaccess",
            "authkeywords", "fund-sponsor",
        ]
        standard_fields = [
            "dc:identifier", "dc:title", "dc:creator", "prism:publicationName",
            "prism:coverDate", "prism:doi", "citedby-count", "eid",
            "prism:url", "subtypeDescription", "openaccess",
        ]

        attempts = [
            {"view": "COMPLETE", "field": ",".join(rich_fields)},
            {"view": "STANDARD", "field": ",".join(standard_fields)},
            {"view": "STANDARD"},
        ]

        last_error = None
        for attempt in attempts:
            params = {
                "query": query,
                "start": start,
                "count": count,
                "sort": "-citedby-count",
                **attempt,
            }
            try:
                response = requests.get(
                    self.search_url,
                    headers=self.headers,
                    params=params,
                    timeout=30,
                )
                response.raise_for_status()
                return response.json().get("search-results", {}).get("entry", [])
            except requests.HTTPError as exc:
                last_error = exc
                status = exc.response.status_code if exc.response is not None else None
                if status in {400, 401, 403}:
                    logger.debug(f"Scopus search retrying with lower view after HTTP {status}")
                    continue
                raise

        if last_error:
            raise last_error
        return []

    def get_paper_details(self, eid: str = "", doi: str = "") -> Dict:
        """Fetch abstract/details using free metadata views before restricted views."""
        if not eid and not doi:
            return {}

        for url in self._abstract_urls(eid, doi):
            for view in ("META_ABS", "META", "FULL"):
                try:
                    response = requests.get(
                        url,
                        headers=self.headers,
                        params={"view": view},
                        timeout=30,
                    )
                    response.raise_for_status()
                    data = response.json().get("abstracts-retrieval-response", {})
                    if data:
                        return self._parse_abstract_response(data)
                except Exception as exc:
                    logger.debug(f"Scopus abstract lookup failed for {url} view={view}: {exc}")
                    continue

        return {}

    def get_source_metrics(self, issn: str) -> Dict:
        """Fetch free source/journal metadata by ISSN when available."""
        clean_issn = re.sub(r"[^0-9Xx]", "", issn or "")
        if not clean_issn:
            return {}

        try:
            response = requests.get(
                f"{self.serial_issn_url}/{clean_issn}",
                headers=self.headers,
                params={"view": "ENHANCED"},
                timeout=30,
            )
            response.raise_for_status()
            entries = response.json().get("serial-metadata-response", {}).get("entry", [])
            if isinstance(entries, dict):
                entries = [entries]
            if not entries:
                return {}
            return self._parse_source_metrics(entries[0])
        except Exception as exc:
            logger.debug(f"Scopus source metrics lookup failed for ISSN {issn}: {exc}")
            return {}

    def search_by_doi(self, doi: str) -> Optional[Dict]:
        """Verify/enrich one DOI against Scopus."""
        clean_doi = (doi or "").strip()
        if not clean_doi:
            return None

        try:
            entries = self._search_entries(
                query=f'DOI("{clean_doi}")',
                start=0,
                count=1,
            )
            if not entries:
                return None

            paper = self._parse_search_entry(entries[0])
            if paper and paper.get("eid"):
                details = self.get_paper_details(eid=paper["eid"], doi=paper.get("doi", clean_doi))
                paper.update({k: v for k, v in details.items() if v})
            if paper and paper.get("issn"):
                metrics = self.get_source_metrics(paper["issn"])
                if metrics:
                    paper["source_metrics"] = metrics
            return paper
        except Exception as exc:
            logger.debug(f"Scopus DOI verification failed for {clean_doi}: {exc}")
            return None

    def _abstract_urls(self, eid: str, doi: str) -> List[str]:
        urls = []
        if eid:
            urls.append(f"{self.abstract_eid_url}/{eid}")
        if doi:
            urls.append(f"{self.abstract_doi_url}/{doi}")
        return urls

    def _parse_abstract_response(self, data: Dict) -> Dict:
        coredata = data.get("coredata", {})
        authors = self._parse_authors(data.get("authors", {}).get("author", []))
        keywords = self._parse_keywords(data)
        subject_areas = self._parse_subject_areas(data)
        affiliations = self._parse_affiliations(data)

        return {
            "abstract": coredata.get("dc:description", ""),
            "authors": authors,
            "fields": subject_areas or keywords,
            "keywords": keywords,
            "subject_areas": subject_areas,
            "affiliations": affiliations,
            "url": coredata.get("prism:url", ""),
            "issn": coredata.get("prism:issn", ""),
            "isbn": coredata.get("prism:isbn", ""),
            "document_type": coredata.get("subtypeDescription", ""),
            "open_access": str(coredata.get("openaccess", "")).lower() in {"1", "true", "yes"},
        }

    def _parse_authors(self, raw_authors) -> List[str]:
        if isinstance(raw_authors, dict):
            raw_authors = [raw_authors]

        authors = []
        for author in raw_authors or []:
            name = (
                author.get("ce:indexed-name")
                or author.get("preferred-name", {}).get("ce:indexed-name")
                or author.get("authname")
            )
            if name:
                authors.append(name)
        return authors

    def _parse_keywords(self, data: Dict) -> List[str]:
        keywords = []
        raw_keywords = data.get("authkeywords", {}).get("author-keyword", [])
        if not raw_keywords:
            raw_keywords = (
                data.get("item", {})
                .get("bibrecord", {})
                .get("head", {})
                .get("citation-info", {})
                .get("author-keywords", {})
                .get("author-keyword", [])
            )
        if isinstance(raw_keywords, dict):
            raw_keywords = [raw_keywords]
        for keyword in raw_keywords or []:
            value = keyword.get("$") if isinstance(keyword, dict) else keyword
            if value:
                keywords.append(str(value).strip())
        return keywords

    def _parse_subject_areas(self, data: Dict) -> List[str]:
        raw_areas = data.get("subject-areas", {}).get("subject-area", [])
        if isinstance(raw_areas, dict):
            raw_areas = [raw_areas]

        areas = []
        for area in raw_areas or []:
            label = area.get("$") if isinstance(area, dict) else area
            if label:
                areas.append(str(label).strip())
        return areas

    def _parse_affiliations(self, data: Dict) -> List[Dict]:
        raw_affiliations = data.get("affiliation", [])
        if isinstance(raw_affiliations, dict):
            raw_affiliations = [raw_affiliations]

        affiliations = []
        for aff in raw_affiliations or []:
            if not isinstance(aff, dict):
                continue
            name = aff.get("affilname") or aff.get("affiliation-name")
            if not name:
                continue
            affiliations.append({
                "name": name,
                "city": aff.get("affiliation-city") or aff.get("city", ""),
                "country": aff.get("affiliation-country") or aff.get("country", ""),
                "afid": aff.get("afid") or aff.get("affiliation-id", ""),
            })
        return affiliations

    def _parse_source_metrics(self, entry: Dict) -> Dict:
        metrics = {
            "title": entry.get("dc:title", ""),
            "publisher": entry.get("dc:publisher", ""),
            "source_type": entry.get("prism:aggregationType", ""),
            "open_access": str(entry.get("openaccess", "")).lower() in {"1", "true", "yes"},
            "homepage": self._extract_link(entry, "homepage"),
        }

        metrics["subject_areas"] = self._list_values(entry.get("subject-area", []))
        metrics["sjr"] = self._first_nested_value(entry.get("SJRList", {}), ["SJR", "$"])
        metrics["snip"] = self._first_nested_value(entry.get("SNIPList", {}), ["SNIP", "$"])
        cite_score_info = entry.get("citeScoreYearInfoList", {}).get("citeScoreCurrentMetric", "")
        metrics["cite_score"] = cite_score_info

        return {k: v for k, v in metrics.items() if v not in ("", [], None)}

    def _build_query(
        self,
        query: str,
        start_year: Optional[int],
        end_year: Optional[int],
        min_citations: Optional[int],
    ) -> str:
        parts = [f'TITLE-ABS-KEY("{query}")']
        if start_year:
            parts.append(f"PUBYEAR > {start_year - 1}")
        if end_year:
            parts.append(f"PUBYEAR < {end_year + 1}")
        return " AND ".join(parts)

    def _parse_search_entry(self, entry: Dict) -> Optional[Dict]:
        try:
            cover_date = entry.get("prism:coverDate", "")
            year = int(cover_date[:4]) if cover_date[:4].isdigit() else 0
            keywords = entry.get("authkeywords", "")
            if isinstance(keywords, str):
                keywords = [k.strip() for k in keywords.split("|") if k.strip()]
            authors = self._parse_authors(entry.get("author", []))

            creator = entry.get("dc:creator", "")
            if not authors and creator:
                authors = [creator]
            eid = entry.get("eid", "")
            affiliations = self._parse_search_affiliations(entry.get("affiliation", []))

            return {
                "title": entry.get("dc:title", ""),
                "authors": authors,
                "abstract": entry.get("dc:description", ""),
                "year": year,
                "published_date": cover_date,
                "pdf_url": "",
                "source": "Scopus",
                "citations": int(entry.get("citedby-count", 0) or 0),
                "journal": entry.get("prism:publicationName", ""),
                "doi": entry.get("prism:doi", ""),
                "issn": entry.get("prism:issn", ""),
                "isbn": entry.get("prism:isbn", ""),
                "url": entry.get("prism:url", ""),
                "paper_id": eid,
                "eid": eid,
                "scopus_id": entry.get("dc:identifier", "").replace("SCOPUS_ID:", ""),
                "document_type_code": entry.get("subtype", ""),
                "document_type": entry.get("subtypeDescription", ""),
                "fields": keywords,
                "keywords": keywords,
                "affiliations": affiliations,
                "funding_sponsors": self._list_values(entry.get("fund-sponsor", [])),
                "scopus_verified": True,
                "source_tier": "verified",
                "open_access": str(entry.get("openaccess", "")).lower() in {"1", "true", "yes"},
            }
        except Exception as exc:
            logger.warning(f"Error parsing Scopus entry: {exc}")
            return None

    def _parse_search_affiliations(self, raw_affiliations) -> List[Dict]:
        if isinstance(raw_affiliations, dict):
            raw_affiliations = [raw_affiliations]

        affiliations = []
        for aff in raw_affiliations or []:
            if not isinstance(aff, dict):
                continue
            name = aff.get("affilname") or aff.get("affiliation-name")
            if not name:
                continue
            affiliations.append({
                "name": name,
                "city": aff.get("affiliation-city", ""),
                "country": aff.get("affiliation-country", ""),
                "afid": aff.get("afid", ""),
            })
        return affiliations

    def _list_values(self, raw_value) -> List[str]:
        if isinstance(raw_value, str):
            return [raw_value] if raw_value else []
        if isinstance(raw_value, dict):
            raw_value = [raw_value]

        values = []
        for item in raw_value or []:
            if isinstance(item, dict):
                value = item.get("$") or item.get("value") or item.get("name")
            else:
                value = item
            if value:
                values.append(str(value).strip())
        return values

    def _first_nested_value(self, data: Dict, path: List[str]):
        current = data
        for key in path:
            if isinstance(current, list):
                current = current[0] if current else {}
            if not isinstance(current, dict):
                return current
            current = current.get(key, "")
        return current

    def _extract_link(self, entry: Dict, ref: str) -> str:
        links = entry.get("link", [])
        if isinstance(links, dict):
            links = [links]
        for link in links or []:
            if link.get("@ref") == ref:
                return link.get("@href", "")
        return ""
