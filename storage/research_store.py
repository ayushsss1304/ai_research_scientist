"""SQLite storage for VeriSci workspaces and saved papers."""

import json
import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional


class ResearchStore:
    """Small local SQLite store for MVP workflows."""

    def __init__(self, db_path: str = "data/research_scientist.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS workspaces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS saved_papers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workspace_id INTEGER NOT NULL,
                    identity_key TEXT NOT NULL,
                    title TEXT NOT NULL,
                    doi TEXT DEFAULT '',
                    source TEXT DEFAULT '',
                    year INTEGER DEFAULT 0,
                    citations INTEGER DEFAULT 0,
                    journal TEXT DEFAULT '',
                    paper_json TEXT NOT NULL,
                    notes TEXT DEFAULT '',
                    tags TEXT DEFAULT '',
                    saved_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
                )
            """)
            conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_saved_paper_identity
                ON saved_papers (workspace_id, identity_key)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_saved_workspace
                ON saved_papers (workspace_id, saved_at DESC)
            """)
            if not conn.execute("SELECT id FROM workspaces LIMIT 1").fetchone():
                now = self._now()
                conn.execute(
                    "INSERT INTO workspaces (name, description, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    ("My Research Workspace", "Default local research project", now, now),
                )

    def list_workspaces(self) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT w.*,
                       COUNT(p.id) AS paper_count
                FROM workspaces w
                LEFT JOIN saved_papers p ON p.workspace_id = w.id
                GROUP BY w.id
                ORDER BY w.updated_at DESC, w.id DESC
            """).fetchall()
            return [dict(row) for row in rows]

    def create_workspace(self, name: str, description: str = "") -> Dict:
        name = (name or "").strip()
        if not name:
            raise ValueError("Workspace name is required")
        now = self._now()
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO workspaces (name, description, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (name, description.strip(), now, now),
            )
            workspace_id = cursor.lastrowid
        return self.get_workspace(workspace_id)

    def get_workspace(self, workspace_id: int) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM workspaces WHERE id = ?", (workspace_id,)).fetchone()
            return dict(row) if row else None

    def get_default_workspace(self) -> Dict:
        workspaces = self.list_workspaces()
        return workspaces[0]

    def save_paper(self, workspace_id: int, paper: Dict, notes: str = "", tags: str = "") -> Dict:
        if not self.get_workspace(workspace_id):
            raise ValueError("Workspace not found")
        title = (paper.get("title") or "").strip()
        if not title:
            raise ValueError("Paper title is required")

        now = self._now()
        payload = json.dumps(paper, ensure_ascii=False)
        doi = paper.get("doi", "") or ""
        identity_key = doi.strip().lower() or title.lower()

        with self._connect() as conn:
            conn.execute("""
                INSERT INTO saved_papers (
                    workspace_id, identity_key, title, doi, source, year, citations, journal,
                    paper_json, notes, tags, saved_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(workspace_id, identity_key)
                DO UPDATE SET
                    title = excluded.title,
                    doi = excluded.doi,
                    source = excluded.source,
                    year = excluded.year,
                    citations = excluded.citations,
                    journal = excluded.journal,
                    paper_json = excluded.paper_json,
                    notes = CASE WHEN excluded.notes != '' THEN excluded.notes ELSE saved_papers.notes END,
                    tags = CASE WHEN excluded.tags != '' THEN excluded.tags ELSE saved_papers.tags END,
                    updated_at = excluded.updated_at
            """, (
                workspace_id,
                identity_key,
                title,
                doi,
                paper.get("source", "") or "",
                int(paper.get("year") or 0),
                int(paper.get("citations") or 0),
                paper.get("journal", "") or "",
                payload,
                notes,
                tags,
                now,
                now,
            ))
            conn.execute(
                "UPDATE workspaces SET updated_at = ? WHERE id = ?",
                (now, workspace_id),
            )
            row = conn.execute("""
                SELECT * FROM saved_papers
                WHERE workspace_id = ? AND identity_key = ?
            """, (workspace_id, identity_key)).fetchone()
            return self._paper_row(row)

    def list_saved_papers(self, workspace_id: int) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT * FROM saved_papers
                WHERE workspace_id = ?
                ORDER BY saved_at DESC, id DESC
            """, (workspace_id,)).fetchall()
            return [self._paper_row(row) for row in rows]

    def delete_saved_paper(self, workspace_id: int, saved_id: int) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM saved_papers WHERE workspace_id = ? AND id = ?",
                (workspace_id, saved_id),
            )
            return cursor.rowcount > 0

    def dashboard_stats(self, workspace_id: int) -> Dict:
        papers = self.list_saved_papers(workspace_id)
        return {
            "saved_papers": len(papers),
            "scopus_verified": sum(1 for p in papers if p["paper"].get("scopus_verified")),
            "open_access": sum(1 for p in papers if p["paper"].get("open_access") or p["paper"].get("pdf_url")),
            "sources": sorted({p["paper"].get("source", "Unknown") for p in papers if p["paper"].get("source")}),
        }

    def _paper_row(self, row) -> Dict:
        data = dict(row)
        data["paper"] = json.loads(data.pop("paper_json") or "{}")
        return data

    def _now(self) -> str:
        return datetime.utcnow().isoformat(timespec="seconds") + "Z"
