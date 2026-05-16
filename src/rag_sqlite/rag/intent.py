import httpx
import json
import re
from typing import Literal
from loguru import logger

from rag_sqlite.core.config import settings

IntentType = Literal["ingest", "query", "mixed"]


INTENT_PROMPT = """Classifique a intenção da mensagem em uma das categorias:
- "ingest": o usuário está informando, anotando, registrando ou descrevendo algo.
- "query": o usuário está perguntando, buscando, resumindo ou pedindo informação.
- "mixed": o usuário está fazendo ambos.

Responda APENAS com JSON:
{{"intent": "ingest|query|mixed", "target_module": "general|finance|obra|shopping"}}

Mensagem: {message}
JSON:"""


def _extract_json(raw: str) -> dict:
    """
    Extrai o primeiro objeto JSON válido de uma string,
    mesmo que contenha texto antes/depois do JSON.
    """
    # Tentar parse direto
    raw = raw.strip()
    if raw.startswith("{"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

    # Tentar encontrar JSON com regex
    match = re.search(r'\{[^{}]+\}', raw)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Tentar encontrar JSON multilinhas
    match = re.search(r'\{.*?\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON found in: {raw[:200]}")


async def _call_ollama(base_url: str, model: str, prompt: str, headers: dict) -> dict:
    async with httpx.AsyncClient(timeout=45.0) as client:
        resp = await client.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_ctx": 4096},
            },
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("response", "").strip()
        if not raw:
            raise ValueError("Ollama returned empty response")
        return _extract_json(raw)


async def classify_intent(message: str, context: str = "local") -> dict:
    """
    Classifica intenção da mensagem usando LLM leve via Ollama.
    Retorna dict com intent, entities, confidence, target_module.
    """
    base_url = settings.ollama.local_base_url if context == "local" else settings.ollama.cloud_base_url
    headers = {}
    if context == "cloud" and settings.ollama.cloud_api_key:
        headers["Authorization"] = f"Bearer {settings.ollama.cloud_api_key}"

    model = settings.ollama.local_chat_model if context == "local" else settings.ollama.cloud_chat_model

    # Truncar mensagem longa para classificação (não precisa do texto inteiro)
    truncated = message[:1500] if len(message) > 1500 else message
    prompt = INTENT_PROMPT.format(message=truncated)

    try:
        result = await _call_ollama(base_url, model, prompt, headers)
    except (httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadTimeout) as e:
        logger.warning(f"Intent failed at {base_url} ({type(e).__name__}: {e})")
        cloud_url = settings.ollama.cloud_base_url
        if cloud_url and cloud_url != base_url:
            cloud_headers = {}
            if settings.ollama.cloud_api_key:
                cloud_headers["Authorization"] = f"Bearer {settings.ollama.cloud_api_key}"
            try:
                result = await _call_ollama(
                    cloud_url,
                    settings.ollama.cloud_chat_model,
                    prompt, cloud_headers,
                )
            except Exception as e2:
                logger.error(f"Intent cloud fallback failed: {e2}. Falling back to heuristic.")
                result = _heuristic_intent(message)
        else:
            logger.warning(f"No different cloud URL for fallback. Using heuristic.")
            result = _heuristic_intent(message)
    except Exception as e:
        logger.error(f"Intent classification failed: {type(e).__name__}: {e}. Falling back to heuristic.")
        result = _heuristic_intent(message)

    # Normalizar
    intent = result.get("intent", "query")
    if intent not in ("ingest", "query", "mixed"):
        intent = "query"

    logger.info(f"Intent: {intent} for message: {message[:60]}...")
    return {
        "intent": intent,
        "entities": result.get("entities", {}),
        "confidence": result.get("confidence", 0.5),
        "target_module": result.get("target_module", "general"),
    }


def _heuristic_intent(message: str) -> dict:
    """Fallback rápido se LLM falhar."""
    text = message.lower()
    ingest_words = ["anote", "salve", "registre", "gastei", "recebi", "comprei", "fiz", "aconteceu"]
    query_words = ["quanto", "qual", "quando", "onde", "como", "me diga", "me fale", "resuma", "busque", "liste",
                   "o que", "quais", "por que", "explique", "?"]

    ingest_score = sum(1 for w in ingest_words if w in text)
    query_score = sum(1 for w in query_words if w in text)

    if ingest_score > 0 and query_score > 0:
        intent = "mixed"
    elif ingest_score > query_score:
        intent = "ingest"
    else:
        intent = "query"

    return {"intent": intent, "entities": {}, "confidence": 0.5, "target_module": "general"}
