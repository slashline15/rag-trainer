"""Quick validation test for all the fixes."""
import sys
sys.path.insert(0, "src")

from rag_sqlite.core.config import settings
from rag_sqlite.core.database import get_db
from rag_sqlite.core.embeddings import EmbeddingClient
from rag_sqlite.utils.chunker import IntentChunker, SemanticChunker, make_chunk_id
from rag_sqlite.rag.engine import RAGEngine
from rag_sqlite.rag.retrieval import HybridRetriever, VEC_WEIGHT, FTS_WEIGHT
from rag_sqlite.rag.intent import classify_intent, _heuristic_intent, _extract_json
from rag_sqlite.rag.prompt_builder import build_rag_prompt, build_ingest_prompt
from rag_sqlite.rag.generator import generate_response
from rag_sqlite.pipeline.ingest import ingest_text, ingest_audio, ingest_document, ingest_image
from rag_sqlite.tg.bot import main as bot_main
from rag_sqlite.tg.flows import build_rating_buttons
print("All imports OK!")

# Test heuristic intent
r = _heuristic_intent("Qual a capital da França?")
assert r["intent"] == "query", f"Expected query, got {r['intent']}"
r = _heuristic_intent("Anote que gastei 500")
assert r["intent"] == "ingest", f"Expected ingest, got {r['intent']}"
r = _heuristic_intent("Anote que gastei 500 e quanto tenho no total?")
assert r["intent"] == "mixed", f"Expected mixed, got {r['intent']}"
print("Heuristic intent OK!")

# Test JSON extraction
r = _extract_json('{"intent": "query", "target_module": "general"}')
assert r["intent"] == "query"
r = _extract_json('Sure! Here is the JSON: {"intent": "ingest", "target_module": "finance"} that you asked.')
assert r["intent"] == "ingest"
print("JSON extraction OK!")

# Test chunking with new size
chunker = SemanticChunker()
assert chunker.chunk_size == 1024, f"Expected 1024, got {chunker.chunk_size}"
assert chunker.overlap == 150, f"Expected 150, got {chunker.overlap}"
print(f"Chunk size: {chunker.chunk_size}, overlap: {chunker.overlap} — OK!")

# Test retrieval weights
assert VEC_WEIGHT == 0.7, f"Expected 0.7, got {VEC_WEIGHT}"
assert FTS_WEIGHT == 0.3, f"Expected 0.3, got {FTS_WEIGHT}"
print(f"Retrieval weights: vec={VEC_WEIGHT}, fts={FTS_WEIGHT} — OK!")

# Test DB has source_path column
db = get_db()
cols = db.get_table_columns("chunks")
assert "source_path" in cols, f"source_path not in chunks columns: {cols}"
print(f"Chunks columns: {cols} — OK!")

# Test prompt builder - no context
prompt, aliases = build_rag_prompt("O que são planetas?", [])
assert "Nenhum trecho relevante" in prompt, "Expected 'no context' indicator in prompt"
print("Prompt builder (empty context) OK!")

# Test prompt builder - with chunks
chunks = [
    {"id": "c1", "content": "Terra é um planeta", "tags": "#m1", "combined_score": 0.85, "source_id": "s1"},
    {"id": "c2", "content": "Marte é o quarto planeta", "tags": "#m2", "combined_score": 0.72, "source_id": "s2"},
]
prompt, aliases = build_rag_prompt("Quais planetas?", chunks)
assert "#m1" in prompt, "Expected #m1 tag in prompt"
assert "#m2" in prompt, "Expected #m2 tag in prompt"
assert "síntese" not in prompt.lower() or "Combine" in prompt or "múltiplos" in prompt
print("Prompt builder (with chunks) OK!")

# Test system prompt allows general knowledge
assert "Conhecimento geral é permitido" in prompt or "conhecimento" in prompt.lower()
print("System prompt allows general knowledge OK!")

print("\n=== ALL TESTS PASSED ===")
