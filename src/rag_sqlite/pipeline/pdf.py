from pathlib import Path
from typing import List
import fitz  # PyMuPDF
from loguru import logger

from rag_sqlite.utils.chunker import SemanticChunker, make_chunk_id
from rag_sqlite.core.embeddings import embedding_client


async def extract_pdf_text(file_path: str) -> str:
    """
    Extrai texto nativo de PDF via PyMuPDF.
    Páginas sem texto (escaneadas) são ignoradas — o embedding direto do texto
    existente é suficiente para a maioria dos casos.
    """
    doc = fitz.open(file_path)
    parts: List[str] = []

    for page in doc:
        text = page.get_text().strip()
        if text:
            parts.append(text)

    doc.close()
    full_text = "\n\n".join(parts)
    logger.info(f"PDF {Path(file_path).name}: {len(full_text)} chars, {len(parts)} pages with text")
    return full_text


async def process_pdf(file_path: str, source_id: str, user_id: int) -> List[dict]:
    """Extrai texto, chunka, gera embeddings."""
    text = await extract_pdf_text(file_path)
    if not text:
        logger.warning(f"No text in {file_path}")
        return []

    chunker = SemanticChunker()
    chunks = chunker.chunk(text, source_id)
    embeddings = await embedding_client.embed([c.text for c in chunks])

    records = []
    for c, emb in zip(chunks, embeddings):
        records.append({
            "chunk_id": make_chunk_id(source_id, c.index),
            "user_id": user_id,
            "content": c.text,
            "embedding": emb,
            "source_type": "pdf",
            "source_id": source_id,
            "module": "general",
            "tags": c.tags,
            "is_public": False,
            "confidence": 1.0,
        })
    return records
