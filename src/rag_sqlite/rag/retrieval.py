from typing import List, Dict, Any, Optional
import sqlite3
from loguru import logger

from rag_sqlite.core.database import get_db
from rag_sqlite.core.embeddings import embedding_client


# Pesos para o re-ranking híbrido.
# O usuário relatou que a busca parecia literal (FTS pesado),
# então priorizamos a busca vetorial (semântica).
VEC_WEIGHT = 0.7
FTS_WEIGHT = 0.3

# Limiar mínimo de distância para filtrar auto-match
# (quando o chunk é a própria query do usuário, distance ≈ 0.0)
MIN_DISTANCE_THRESHOLD = 0.05


def _deduplicate_merge(
    vec_results: List[Dict[str, Any]],
    fts_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Merge e deduplica resultados de vec e FTS, combinando scores ponderados.
    Se um chunk aparece nos dois, seus scores são somados com pesos.
    """
    scores: Dict[str, float] = {}
    sources: Dict[str, Dict[str, Any]] = {}

    for r in vec_results:
        cid = r.get("id")
        if not cid:
            continue
        norm = r.get("norm_score", 0.0)
        scores[cid] = scores.get(cid, 0.0) + norm * VEC_WEIGHT
        if cid not in sources:
            sources[cid] = r

    for r in fts_results:
        cid = r.get("id")
        if not cid:
            continue
        norm = r.get("norm_score", 0.0)
        scores[cid] = scores.get(cid, 0.0) + norm * FTS_WEIGHT
        if cid not in sources:
            sources[cid] = r

    return scores, sources


def _normalize_scores(results: List[Dict[str, Any]], score_key: str, invert: bool = True) -> List[Dict[str, Any]]:
    """Normaliza scores para 0-1. Se invert=True, assume que menor é melhor (distância)."""
    if not results:
        return results
    vals = [r.get(score_key, 0) for r in results]
    min_v, max_v = min(vals), max(vals)
    if max_v == min_v:
        for r in results:
            r["norm_score"] = 1.0
        return results
    for r in results:
        v = r.get(score_key, 0)
        if invert:
            r["norm_score"] = 1.0 - (v - min_v) / (max_v - min_v)
        else:
            r["norm_score"] = (v - min_v) / (max_v - min_v)
    return results


class HybridRetriever:
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.db = get_db()

    async def retrieve(self, user_id: int, query_text: str,
                       module: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Busca híbrida combinando:
        1. Busca vetorial (sqlite-vec) — peso 0.7
        2. Busca FTS5 (texto exato) — peso 0.3
        3. Filtra auto-match (chunks que são a própria query)
        4. Re-ranqueia por score combinado ponderado
        """
        # 1. Embedding da query
        query_emb = await embedding_client.embed_single(query_text)

        # 2. Busca vetorial
        vec_results = self.db.search_vec(user_id, query_emb, top_k=self.top_k * 3, module=module)

        # Filtrar auto-match: chunks com distance muito baixa (≈ a própria query)
        vec_results = [
            r for r in vec_results
            if r.get("distance", 0) > MIN_DISTANCE_THRESHOLD
        ]
        vec_results = _normalize_scores(vec_results, "distance", invert=True)

        # 3. Busca FTS5
        fts_results = self.db.search_fts(user_id, query_text, top_k=self.top_k * 2)
        fts_results = _normalize_scores(fts_results, "rank", invert=True)

        # 4. Merge ponderado
        scores, sources = _deduplicate_merge(vec_results, fts_results)

        # 5. Ordenar por score combinado decrescente
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        final = []
        for cid, score in ranked[:self.top_k]:
            item = dict(sources[cid])
            item["combined_score"] = round(score, 4)
            final.append(item)

        logger.info(
            f"Hybrid retrieval for user {user_id}: {len(final)} chunks "
            f"(vec={len(vec_results)}, fts={len(fts_results)}, "
            f"vec_weight={VEC_WEIGHT}, fts_weight={FTS_WEIGHT})"
        )
        for f in final:
            logger.debug(f"  -> {f['id']} score={f.get('combined_score')} src={f.get('source_type')} module={f.get('module')}")
        return final

    def retrieve_structured(self, user_id: int, sql_query: str, params: tuple) -> List[Dict[str, Any]]:
        """
        Executa SQL customizado sobre módulos (ex: module_finance).
        Deve ser usado quando o classificador detectar intenção de consulta estruturada.
        """
        try:
            with self.db._connect() as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(sql_query, params).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Structured query failed: {e}")
            return []

    def get_recent_ratings_for_context(self, user_id: int, limit: int = 5) -> List[Dict[str, Any]]:
        """Recupera avaliações recentes para RAG contrastivo (exemplos positivos/negativos)."""
        return self.db.get_ratings_for_context(user_id, limit)
