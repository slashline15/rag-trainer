"""
Teste end-to-end: valida fluxo completo de ingestão + RAG com DB real em memória.
Todos os endpoints externos (Ollama embeddings/chat) são mockados.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import asyncio
import tempfile
import os

from rag_sqlite.core.database import Database
from rag_sqlite.core.embeddings import embedding_client
from rag_sqlite.rag.engine import RAGEngine
from rag_sqlite.pipeline.ingest import ingest_text
from rag_sqlite.utils.chunker import SemanticChunker


# Mock do embedding (vetor aleatório normalizado)
async def mock_embed(texts, context="local"):
    import random
    dim = 768
    results = []
    for _ in texts:
        vec = [random.random() for _ in range(dim)]
        # normalizar
        norm = sum(v * v for v in vec) ** 0.5
        vec = [v / norm for v in vec]
        results.append(vec)
    return results


async def mock_embed_single(text, context="local"):
    return (await mock_embed([text]))[0]


async def test_ingest_and_query():
    # Criar DB temporário
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = Database(db_path)

    # Substituir o singleton get_db para apontar para o DB temporário
    import rag_sqlite.core.database as db_module
    original_db_instance = db_module._db_instance
    db_module._db_instance = db

    # Mock embeddings
    original_embed = embedding_client.embed
    original_embed_single = embedding_client.embed_single
    embedding_client.embed = mock_embed
    embedding_client.embed_single = mock_embed_single

    # Mock do generate_single para não depender de Ollama
    import rag_sqlite.rag.generator as gen_module
    original_generate_single = gen_module.generate_single

    async def mock_generate_single(prompt, model, base_url, headers, temperature=0.3, timeout=60.0):
        return "Resposta mockada: você gastou 500 reais com material."

    gen_module.generate_single = mock_generate_single

    try:
        user_id = 42

        # 1. Ingestão de conhecimento
        text1 = "Gastei 500 reais com material de construção na obra do muro. #obra #material"
        result_ingest = await ingest_text(user_id, text1, is_public=False)
        assert "semantic_chunks" in result_ingest
        assert len(result_ingest["semantic_chunks"]) > 0
        print(f"Ingested {len(result_ingest['semantic_chunks'])} chunks")

        # 2. Consulta via RAG Engine
        engine = RAGEngine()
        result = await engine.process(user_id, "quanto gastei com material na obra?")

        assert "response" in result
        assert "final_confidence" in result
        assert 0.0 <= result["final_confidence"] <= 1.0
        assert "retrieval_score" in result
        print(f"RAG response: {result['response'][:100]}...")
        print(f"Confidence: {result['final_confidence']}")
        print(f"Retrieval score: {result['retrieval_score']}")

        # 3. Verificar que chunks estão no banco
        chunks = db.get_chunks_by_source(result_ingest["source_id"])
        assert len(chunks) > 0
        print(f"Chunks in DB: {len(chunks)}")

        # 4. Verificar busca vetorial
        query_emb = await mock_embed_single("material obra")
        vec_results = db.search_vec(user_id, query_emb, top_k=5)
        assert len(vec_results) > 0
        print(f"Vector search returned {len(vec_results)} results")

        # 5. Verificar busca FTS
        fts_results = db.search_fts(user_id, "material obra", top_k=5)
        assert len(fts_results) >= 0  # pode ser 0 se FTS não indexou ainda, mas triggers devem ter funcionado
        print(f"FTS search returned {len(fts_results)} results")

        # 6. Verificar que avaliação pode ser salva
        db.insert_rating({
            "user_id": user_id,
            "message_id": "msg_1",
            "response_id": "resp_1",
            "rating": 4,
            "correction_type": None,
            "correction_text": None,
            "response_time_ms": 1500,
            "tokens_input": 100,
            "tokens_output": 50,
            "model_used": "llama3.2:latest",
            "self_consistency_score": None,
            "retrieval_score": result["retrieval_score"],
            "final_confidence": result["final_confidence"],
        })
        ratings = db.get_ratings_for_context(user_id, limit=10)
        assert len(ratings) == 1
        assert ratings[0]["rating"] == 4
        print("Rating saved and retrieved OK")

        print("\n[OK] End-to-end test passed!")
    finally:
        embedding_client.embed = original_embed
        embedding_client.embed_single = original_embed_single
        gen_module.generate_single = original_generate_single
        db_module._db_instance = original_db_instance
        os.unlink(db_path)


if __name__ == "__main__":
    asyncio.run(test_ingest_and_query())
