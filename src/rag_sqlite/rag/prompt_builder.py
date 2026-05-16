from typing import List, Dict, Any, Tuple


SYSTEM_PROMPT = """Você é um assistente de conhecimento pessoal inteligente.
Você tem acesso a uma base de dados local do usuário E ao seu próprio conhecimento.

Regras:
1. Priorize a base do usuário quando houver trechos relevantes abaixo.
2. Combine informações de TODOS os trechos disponíveis. Não use apenas um.
3. Se a pergunta for de conhecimento geral e não houver trechos relevantes, responda com seu conhecimento próprio.
4. Se citar um trecho da base, mencione a tag (ex: #m5). Nunca invente tags.
5. Responda em português, de forma natural e direta.
6. Não adicione blocos de referências no final.
7. Encapsule trechos de código, python, json, SQL ou outros estritamente utilizando três crases consecutivas (```) no início e no fim.

"""


def build_rag_prompt(query: str, chunks: List[Dict[str, Any]],
                     ratings_context: List[Dict[str, Any]] = None,
                     structured_data: List[Dict[str, Any]] = None,
                     history: List[Dict[str, Any]] = None) -> Tuple[str, Dict[str, str]]:
    """Monta o prompt final para a LLM."""
    parts = [SYSTEM_PROMPT]

    # Histórico de conversa
    if history:
        parts.append("--- Histórico ---")
        for h in reversed(history):
            role_label = "Usuário" if h.get("role") == "user" else "Assistente"
            tag = h.get("tag", "")
            tag_str = f" {tag}" if tag else ""
            content = h.get("raw_content", "")[:500]
            parts.append(f"[{role_label}{tag_str}] {content}")
        parts.append("---\n")

    # Correções do usuário — ensina o modelo a evitar erros passados
    if ratings_context:
        corrections = [
            r for r in ratings_context
            if r.get("rating", 5) <= 2 and (r.get("correction_text") or "").strip()
        ]
        good_corrections = [
            r for r in ratings_context
            if r.get("correction_type") == "ajuste" and (r.get("correction_text") or "").strip()
        ]
        if good_corrections:
            parts.append("--- Correções do usuário (use como referência) ---")
            for r in good_corrections[:3]:
                parts.append(f"Correção: {r['correction_text'][:300]}")
            parts.append("---\n")
        elif corrections:
            parts.append("--- Erros a evitar ---")
            for r in corrections[:2]:
                parts.append(f"Feedback negativo: {r['correction_text'][:300]}")
            parts.append("---\n")

    # Chunks com tags reais
    chunk_aliases: Dict[str, str] = {}
    if chunks:
        parts.append("--- Trechos da base ---")
        for idx, c in enumerate(chunks, start=1):
            # Buscar tag real do source na base
            tag = _get_chunk_tag(c)
            chunk_aliases[c["id"]] = tag
            parts.append(f"[{tag}] {c['content']}\n")
        parts.append("---\n")
    else:
        parts.append("(Nenhum trecho relevante na base.)\n")

    # Dados estruturados
    if structured_data:
        parts.append("--- Dados estruturados ---")
        for row in structured_data[:10]:
            parts.append(str(row))
        parts.append("---\n")

    parts.append(f"Pergunta: {query}\n\nResposta:")
    return "\n".join(parts), chunk_aliases


def _get_chunk_tag(chunk: Dict[str, Any]) -> str:
    """Extrai a tag real (#mX) de um chunk. Nunca retorna '#trecho'."""
    # Tags do chunk (ex: "#m5 #m6")
    tags = chunk.get("tags", "")
    if tags:
        # Pegar a primeira tag que começa com #m
        for t in tags.split():
            if t.startswith("#m"):
                return t
        # Se tem tags mas nenhuma #m, usa a primeira
        first = tags.split()[0]
        if first.startswith("#"):
            return first

    # Fallback: extrair do source_id (formato "uXXX_YYYYMMDD_HHMMSS")
    source_id = chunk.get("source_id", "")
    chunk_id = chunk.get("id", "")
    if "#n" in chunk_id:
        # chunk_id formato "source#n1" — usar o número
        n = chunk_id.split("#n")[-1]
        return f"#c{n}"

    return f"#r{hash(chunk_id) % 100}"


def build_ingest_prompt(text: str) -> str:
    return f"""Analise o texto e extraia entidades estruturadas.
Responda com JSON: {{"module": "finance|obra|shopping|general", "fields": {{"campo": "valor"}}}}
Se não houver dados estruturados: {{"module": "general", "fields": {{}}}}

Texto: {text}
JSON:"""
