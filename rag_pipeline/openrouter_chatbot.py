"""Lightweight OpenRouter-backed chat and document retrieval for cloud deployments."""

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests

from pdf_processing.pdf_processor import PDFProcessor


logger = logging.getLogger(__name__)


class OpenRouterClient:
    """Small OpenAI-compatible client for OpenRouter."""

    def __init__(self, api_key: str, model: str = "google/gemma-4-31b-it:free",
                 site_url: str = "", site_name: str = "VeriSci"):
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not configured")
        self.api_key = api_key
        self.model = model
        self.site_url = site_url
        self.site_name = site_name

    def generate(self, prompt: str, system: str = "", temperature: float = 0.5,
                 max_tokens: int = 1200, image_data_url: Optional[str] = None) -> str:
        content = [{"type": "text", "text": prompt}]
        if image_data_url:
            content.append({"type": "image_url", "image_url": {"url": image_data_url}})

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content if image_data_url else prompt})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=90,
        )
        if not response.ok:
            try:
                detail = response.json().get("error", {}).get("message", response.text)
            except ValueError:
                detail = response.text
            raise RuntimeError(f"OpenRouter request failed ({response.status_code}): {detail[:300]}")

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("OpenRouter returned no response")
        content = choices[0].get("message", {}).get("content", "")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return content.strip()


class OpenRouterRAGChatbot:
    """Cloud-friendly RAG with lexical retrieval and no ML runtime dependency."""

    def __init__(self, api_key: str, model: str = "google/gemma-4-31b-it:free",
                 storage_dir: str = "rag_storage", site_url: str = ""):
        self.client = OpenRouterClient(api_key, model, site_url, "VeriSci")
        self.storage_dir = storage_dir
        self.store_path = os.path.join(storage_dir, "documents.json")
        self.embedding_manager = None
        self.conversation_history: List[Dict[str, str]] = []
        os.makedirs(storage_dir, exist_ok=True)
        self.documents = self._load_documents()

    def _load_documents(self) -> Dict:
        try:
            with open(self.store_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def _save_documents(self) -> None:
        with open(self.store_path, "w", encoding="utf-8") as handle:
            json.dump(self.documents, handle, ensure_ascii=False)

    def add_document_from_file(self, file_path: str, doc_id: Optional[str] = None,
                               chunk_size: int = 1400, chunk_overlap: int = 200) -> bool:
        doc_id = doc_id or os.path.splitext(os.path.basename(file_path))[0]
        processor = PDFProcessor(use_ocr=False)
        text = processor.extract_text_from_pdf(file_path, use_ocr=False)
        if not text.strip():
            return False
        chunks = processor.chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
        self.documents[doc_id] = {
            "chunks": chunks,
            "filename": os.path.basename(file_path),
            "added_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_documents()
        return True

    def list_documents(self) -> List[str]:
        return list(self.documents.keys())

    def get_statistics(self) -> Dict:
        return {
            "total_documents": len(self.documents),
            "total_chunks": sum(len(doc.get("chunks", [])) for doc in self.documents.values()),
            "provider": "OpenRouter",
            "model": self.client.model,
        }

    def remove_document(self, doc_id: str) -> None:
        if self.documents.pop(doc_id, None) is not None:
            self._save_documents()

    def clear_history(self) -> None:
        self.conversation_history = []

    @staticmethod
    def _tokens(text: str) -> set:
        return set(re.findall(r"[a-z0-9]{3,}", text.lower()))

    def _retrieve(self, query: str, top_k: int) -> List[Dict]:
        query_tokens = self._tokens(query)
        ranked = []
        for doc_id, document in self.documents.items():
            for chunk in document.get("chunks", []):
                chunk_tokens = self._tokens(chunk)
                score = len(query_tokens & chunk_tokens) / max(len(query_tokens), 1)
                ranked.append({"doc_id": doc_id, "text": chunk, "score": score})
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[:top_k]

    def chat(self, message: str, top_k: int = 3, use_reranking: bool = True,
             temperature: float = 0.5) -> str:
        matches = self._retrieve(message, top_k)
        context = "\n\n".join(
            f"[Document: {item['doc_id']}]\n{item['text']}" for item in matches
        )
        if context:
            prompt = (
                "Answer the question using the supplied research-document context. "
                "Cite document names in square brackets. If the context is insufficient, say so.\n\n"
                f"Context:\n{context}\n\nQuestion: {message}"
            )
        else:
            prompt = message
        answer = self.client.generate(
            prompt,
            system="You are VeriSci, a careful research assistant. Be concise and evidence-aware.",
            temperature=temperature,
        )
        self.conversation_history.extend([
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ])
        return answer
