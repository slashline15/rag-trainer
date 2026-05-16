import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import asyncio
from rag_sqlite.rag.engine import RAGEngine
from rag_sqlite.rag.intent import _heuristic_intent
from rag_sqlite.rag.retrieval import _deduplicate, _normalize_scores


def test_heuristic_intent():
    assert _heuristic_intent("quanto gastei ontem?")["intent"] in ("query", "mixed")
    assert _heuristic_intent("gastei 500 com material")["intent"] == "ingest"
    assert _heuristic_intent("anote que gastei 500 e me diga o total")["intent"] == "mixed"
    assert _heuristic_intent("qual foi o total de gastos?")["intent"] == "query"
    print("Heuristic intent OK")


def test_deduplicate():
    data = [
        {"id": "a", "content": "foo"},
        {"id": "a", "content": "foo2"},
        {"id": "b", "content": "bar"},
    ]
    out = _deduplicate(data)
    assert len(out) == 2
    assert out[0]["id"] == "a"
    assert out[1]["id"] == "b"
    print("Deduplicate OK")


def test_normalize_scores():
    data = [
        {"id": "a", "distance": 0.1},
        {"id": "b", "distance": 0.5},
        {"id": "c", "distance": 0.9},
    ]
    out = _normalize_scores(data, "distance", invert=True)
    scores = [o["norm_score"] for o in out]
    assert scores[0] > scores[1] > scores[2]
    print("Normalize scores OK")


async def test_engine_structure():
    import rag_sqlite.rag.generator as gen_module
    original_generate_single = gen_module.generate_single

    async def mock_generate_single(prompt, model, base_url, headers, temperature=0.3, timeout=60.0):
        return "Resposta mockada para teste."

    gen_module.generate_single = mock_generate_single
    try:
        engine = RAGEngine()
        result = await engine.process(123, "quanto gastei ontem?")
        assert "response" in result
        assert "final_confidence" in result
        assert 0.0 <= result["final_confidence"] <= 1.0
        print(f"Engine structure OK (conf={result['final_confidence']})")
    finally:
        gen_module.generate_single = original_generate_single


if __name__ == "__main__":
    test_heuristic_intent()
    test_deduplicate()
    test_normalize_scores()
    asyncio.run(test_engine_structure())
