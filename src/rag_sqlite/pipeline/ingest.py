from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from loguru import logger

from rag_sqlite.core.database import get_db
from rag_sqlite.core.embeddings import embedding_client
from rag_sqlite.utils.chunker import IntentChunker, SemanticChunker, make_chunk_id
from rag_sqlite.pipeline.storage import save_media
from rag_sqlite.pipeline.audio import transcribe_audio
from rag_sqlite.pipeline.pdf import process_pdf
from rag_sqlite.pipeline.image import process_image


def generate_source_id(user_id: int, media_type: str) -> str:
    """Gera ID único para uma mensagem/mídia. Usa microsegundos + random para evitar colisão."""
    import random, string
    now = datetime.now()
    suffix = ''.join(random.choices(string.ascii_lowercase, k=4))
    return f"u{user_id}_{now.strftime('%Y%m%d_%H%M%S')}_{now.microsecond:06d}_{suffix}"


async def ingest_text(user_id: int, text: str, is_public: bool = False) -> Dict[str, Any]:
    """
    Ingestão de texto puro (mensagem do Telegram ou texto colado).
    1. Intent chunking (nível 1) para classificação.
    2. Semantic chunking (nível 2) para indexação.
    3. Embeddings + insert no DB.
    """
    db = get_db()
    source_id = generate_source_id(user_id, "message")
    tag = db.generate_next_tag(user_id)

    # Salvar mensagem bruta
    db.insert_message(source_id, user_id, text, media_type="text", role="user", tag=tag)

    # Nível 1: intent chunks (coarse) - útil para classify_intent no RAG
    intent_chunker = IntentChunker()
    intent_chunks = intent_chunker.chunk(text, source_id)
    # Nota: intent chunks não são embedados, apenas guardados como referência ou usados em memória

    # Nível 2: semantic chunks (fine) - embedados e indexados
    semantic_chunker = SemanticChunker()
    sem_chunks = semantic_chunker.chunk(text, source_id)

    embeddings = await embedding_client.embed([c.text for c in sem_chunks])

    records = []
    for c, emb in zip(sem_chunks, embeddings):
        db.insert_chunk(
            chunk_id=make_chunk_id(source_id, c.index),
            user_id=user_id,
            content=c.text,
            embedding=emb,
            source_type="message",
            source_id=source_id,
            module="general",
            tags=c.tags,
            is_public=is_public,
            confidence=1.0,
            source_path=None,  # mensagens de texto não têm arquivo
        )
        records.append({
            "chunk_id": make_chunk_id(source_id, c.index),
            "content": c.text,
            "tags": c.tags,
        })

    logger.info(f"Ingested text {source_id}: {len(sem_chunks)} chunks")
    return {
        "source_id": source_id,
        "intent_chunks": [{"text": ic.text, "tags": ic.tags} for ic in intent_chunks],
        "semantic_chunks": records,
    }


async def ingest_audio(user_id: int, file_name: str, file_data: bytes,
                       context: str = "local", is_public: bool = False) -> Dict[str, Any]:
    """
    1. Salva áudio localmente.
    2. Transcreve via Whisper API.
    3. Chunka e indexa a transcrição como texto.
    """
    # Salvar
    media_path = save_media(user_id, file_name, file_data, media_type="audio")
    db = get_db()

    # Transcrever
    raw_transcription = await transcribe_audio(media_path)

    # Indexar a transcrição como texto (sem correção gramatical para evitar chamada extra ao Ollama)
    result = await ingest_text(user_id, raw_transcription, is_public)
    result["raw_transcription"] = raw_transcription
    result["corrected_transcription"] = raw_transcription
    result["media_path"] = media_path
    return result


async def ingest_document(user_id: int, file_name: str, file_data: bytes,
                          context: str = "local", is_public: bool = False) -> Dict[str, Any]:
    """
    PDF, DOCX, TXT, etc.
    Salva localmente, extrai texto, chunka e indexa.
    """
    ext = Path(file_name).suffix.lower()
    media_path = save_media(user_id, file_name, file_data, media_type="document")

    db = get_db()
    source_id = generate_source_id(user_id, "document")
    tag = db.generate_next_tag(user_id)
    db.insert_message(source_id, user_id, f"Documento: {file_name}", media_type="document", media_path=media_path, role="user", tag=tag)

    records: List[dict] = []
    if ext == ".pdf":
        records = await process_pdf(media_path, source_id, user_id)
    elif ext in (".txt", ".md", ".csv"):
        text = Path(media_path).read_text(encoding="utf-8")
        result = await ingest_text(user_id, text, is_public)
        result["source_id"] = source_id
        result["media_path"] = media_path
        return result
    elif ext in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"):
        # Imagens enviadas como documento → processar como imagem
        records = await process_image(media_path, source_id, user_id)
    else:
        logger.warning(f"Unsupported document type: {ext}")
        return {"source_id": source_id, "media_path": media_path, "chunks": []}

    # Inserir chunks no DB
    for r in records:
        db.insert_chunk(
            chunk_id=r["chunk_id"],
            user_id=r["user_id"],
            content=r["content"],
            embedding=r["embedding"],
            source_type=r["source_type"],
            source_id=r["source_id"],
            module=r["module"],
            tags=r["tags"],
            is_public=r["is_public"],
            confidence=r["confidence"],
            source_path=media_path,
        )

    logger.info(f"Ingested document {source_id} ({file_name}): {len(records)} chunks")
    return {
        "source_id": source_id,
        "media_path": media_path,
        "chunks": [{"chunk_id": r["chunk_id"], "content": r["content"]} for r in records],
    }


async def ingest_image(user_id: int, file_name: str, file_data: bytes,
                       context: str = "local", is_public: bool = False) -> Dict[str, Any]:
    """
    Salva imagem, descreve via modelo multimodal/OCR, chunka e indexa.
    Retorna descrição para o usuário.
    """
    media_path = save_media(user_id, file_name, file_data, media_type="image")

    db = get_db()
    source_id = generate_source_id(user_id, "image")
    tag = db.generate_next_tag(user_id)
    db.insert_message(source_id, user_id, f"Imagem: {file_name}", media_type="image", media_path=media_path, role="user", tag=tag)

    records = await process_image(media_path, source_id, user_id, context)

    for r in records:
        db.insert_chunk(
            chunk_id=r["chunk_id"],
            user_id=r["user_id"],
            content=r["content"],
            embedding=r["embedding"],
            source_type=r["source_type"],
            source_id=r["source_id"],
            module=r["module"],
            tags=r["tags"],
            is_public=r["is_public"],
            confidence=r["confidence"],
            source_path=media_path,
        )

    description = records[0]["content"] if records else ""
    logger.info(f"Ingested image {source_id}: {len(records)} chunks")
    return {
        "source_id": source_id,
        "media_path": media_path,
        "description": description,
        "chunks": [{"chunk_id": r["chunk_id"], "content": r["content"]} for r in records],
    }
