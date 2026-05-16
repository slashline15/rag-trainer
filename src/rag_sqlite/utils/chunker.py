import re
from typing import List, Dict, Any
from dataclasses import dataclass

from rag_sqlite.core.config import settings


@dataclass
class Chunk:
    text: str
    index: int
    source_id: str
    tags: str = ""


def extract_hashtags(text: str) -> str:
    """Extrai hashtags e retorna como string separada por espaços."""
    tags = re.findall(r"#\w+", text)
    return " ".join(tags)


def strip_hashtags(text: str) -> str:
    """Remove hashtags do texto para chunking limpo."""
    return re.sub(r"#\w+", "", text).strip()


class IntentChunker:
    """
    Nível 1: chunking grosseiro para classify_intent.
    Preserva contexto amplo. Pode retornar a mensagem inteira
    ou dividir em blocos grandes se exceder limite.
    """

    def __init__(self, max_length: int = None, overlap: int = None):
        self.max_length = max_length or settings.chunking.intent_max_length
        self.overlap = overlap or settings.chunking.intent_overlap

    def chunk(self, text: str, source_id: str) -> List[Chunk]:
        clean = strip_hashtags(text)
        tags = extract_hashtags(text)

        if len(clean) <= self.max_length:
            return [Chunk(text=clean, index=0, source_id=source_id, tags=tags)]

        chunks: List[Chunk] = []
        start = 0
        idx = 0
        while start < len(clean):
            end = start + self.max_length
            # Buscar quebra natural (fim de parágrafo, sentença)
            if end < len(clean):
                for sep in ["\n\n", ". ", "\n", " "]:
                    pos = clean.rfind(sep, start, end)
                    if pos > start:
                        end = pos + len(sep)
                        break
            chunk_text = clean[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(text=chunk_text, index=idx, source_id=source_id, tags=tags))
                idx += 1
            start = end - self.overlap
        return chunks


class SemanticChunker:
    """
    Nível 2: chunking refinado pós-classificação.
    Divide por mudança de tópico/assunto, comentários e avaliações.
    """

    def __init__(self, chunk_size: int = None, overlap: int = None):
        self.chunk_size = chunk_size or settings.chunking.semantic_chunk_size
        self.overlap = overlap or settings.chunking.semantic_overlap

    def _split_paragraphs(self, text: str) -> List[str]:
        """Divide em parágrafos mantendo delimitadores."""
        return [p.strip() for p in text.split("\n\n") if p.strip()]

    def _maybe_split_long(self, paragraph: str) -> List[str]:
        """Se um parágrafo for muito longo, quebra por sentenças."""
        if len(paragraph) <= self.chunk_size:
            return [paragraph]
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        parts: List[str] = []
        current = ""
        for s in sentences:
            if len(current) + len(s) + 1 <= self.chunk_size:
                current = f"{current} {s}".strip() if current else s
            else:
                if current:
                    parts.append(current)
                current = s
        if current:
            parts.append(current)
        return parts

    def chunk(self, text: str, source_id: str) -> List[Chunk]:
        """
        Fluxo:
        1. Extrair hashtags e limpar texto.
        2. Dividir em parágrafos.
        3. Parágrafos longos quebrar por sentenças.
        4. Agrupar sentenças próximas com overlap para contexto.
        """
        tags = extract_hashtags(text)
        clean = strip_hashtags(text)
        paragraphs = self._split_paragraphs(clean)

        raw_chunks: List[str] = []
        for p in paragraphs:
            raw_chunks.extend(self._maybe_split_long(p))

        # Agrupar com overlap para janela de contexto
        result: List[Chunk] = []
        buf = ""
        idx = 0
        for rc in raw_chunks:
            if len(buf) + len(rc) + 2 <= self.chunk_size:
                buf = f"{buf}\n\n{rc}".strip() if buf else rc
            else:
                if buf:
                    result.append(Chunk(text=buf, index=idx, source_id=source_id, tags=tags))
                    idx += 1
                    # overlap: últimas sentenças do chunk anterior
                    overlap_text = self._tail(buf, self.overlap)
                    buf = f"{overlap_text}\n\n{rc}".strip() if overlap_text else rc
                else:
                    buf = rc
        if buf:
            result.append(Chunk(text=buf, index=idx, source_id=source_id, tags=tags))

        return result

    def _tail(self, text: str, length: int) -> str:
        """Retorna as últimas N caracteres (aproximadamente última sentença)."""
        if len(text) <= length:
            return text
        # Buscar início de sentença próximo ao corte
        cutoff = len(text) - length
        pos = text.find(". ", cutoff)
        if pos == -1:
            pos = text.find("\n", cutoff)
        if pos == -1:
            pos = cutoff
        return text[pos:].strip()


def make_chunk_id(source_id: str, index: int) -> str:
    return f"{source_id}#n{index + 1}"
