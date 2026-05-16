from pathlib import Path
from typing import List
import base64
import httpx
from loguru import logger

from rag_sqlite.core.config import settings
from rag_sqlite.utils.chunker import SemanticChunker, make_chunk_id
from rag_sqlite.core.embeddings import embedding_client


async def describe_image(file_path: str) -> str:
    """
    Usa modelo local Ollama com vision (gemma4:e4b) para descrever imagem.
    Sempre usa o modelo local — é rápido e não esquenta a GPU.
    """
    path = Path(file_path)
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")

    # Sempre local para imagens — gemma4:e4b é eficiente
    base_url = settings.ollama.local_base_url
    model = settings.ollama.local_chat_model
    headers = {}

    prompt = (
        "Analise esta imagem. "
        "Se houver texto, transcreva integralmente. "
        "Se houver tabelas, extraia os dados. "
        "Se houver gráficos ou desenhos técnicos, descreva. "
        "Responda apenas com o conteúdo extraído."
    )

    timeout = float(settings.ollama.timeout) if hasattr(settings.ollama, 'timeout') else 300.0

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "images": [b64],
                    "stream": False,
                    "options": {"temperature": 0.2},
                },
                headers=headers,
            )
            resp.raise_for_status()
            text = resp.json().get("response", "").strip()
            if text:
                logger.info(f"Image {path.name} described ({model}): {len(text)} chars")
                return text
    except Exception as e:
        logger.error(f"Vision failed for {path.name}: {type(e).__name__}: {e}")

    return ""


async def process_image(file_path: str, source_id: str, user_id: int, context: str = "local") -> List[dict]:
    """Descreve imagem via Ollama vision, chunka e retorna registros."""
    description = await describe_image(file_path)
    if not description:
        return []

    chunker = SemanticChunker()
    chunks = chunker.chunk(description, source_id)
    embeddings = await embedding_client.embed([c.text for c in chunks])

    records = []
    for c, emb in zip(chunks, embeddings):
        records.append({
            "chunk_id": make_chunk_id(source_id, c.index),
            "user_id": user_id,
            "content": c.text,
            "embedding": emb,
            "source_type": "image",
            "source_id": source_id,
            "module": "general",
            "tags": c.tags,
            "is_public": False,
            "confidence": 1.0,
        })
    return records
