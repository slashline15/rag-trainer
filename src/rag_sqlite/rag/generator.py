import httpx
import numpy as np
from typing import List, Dict, Any
from loguru import logger

from rag_sqlite.core.config import settings
from rag_sqlite.core.embeddings import embedding_client


async def _try_generate(prompt: str, model: str, base_url: str,
                        headers: dict, temperature: float = 0.3,
                        timeout: float = 60.0) -> str:
    """Tenta gerar uma resposta em um endpoint Ollama."""
    if not base_url:
        raise httpx.ConnectError("No base_url provided")
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_ctx": 4096,
                },
            },
            headers=headers,
        )
        if resp.status_code == 500:
            text = resp.text.lower()
            if "memory" in text or "requires more" in text:
                raise httpx.HTTPStatusError(
                    f"Ollama 500: modelo '{model}' requer mais memória do que disponível. "
                    f"Tente um modelo menor ou libere RAM.",
                    request=resp.request, response=resp,
                )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()


async def generate_single(prompt: str, model: str, base_url: str,
                          headers: dict, temperature: float = 0.3,
                          timeout: float = 60.0) -> str:
    """
    Gera uma única resposta via Ollama.
    Se falhar (404/connection error), tenta fallback para cloud se configurado.
    """
    try:
        return await _try_generate(prompt, model, base_url, headers, temperature, timeout)
    except (httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadTimeout) as e:
        logger.warning(f"Ollama failed at {base_url} ({type(e).__name__}: {e})")
        # Só tenta fallback se a URL cloud for diferente da que já falhou
        cloud_url = settings.ollama.cloud_base_url
        if cloud_url and cloud_url != base_url:
            cloud_headers = {}
            if settings.ollama.cloud_api_key:
                cloud_headers["Authorization"] = f"Bearer {settings.ollama.cloud_api_key}"
            try:
                return await _try_generate(
                    prompt, settings.ollama.cloud_chat_model,
                    cloud_url, cloud_headers,
                    temperature, timeout,
                )
            except Exception as e2:
                logger.error(f"Cloud fallback also failed: {e2}")
                raise e
        raise


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


async def compute_consistency_score(responses: List[str]) -> float:
    """
    Calcula consistência interna entre N respostas.
    Embed cada resposta e calcula similaridade cosseno média entre todos os pares.
    """
    if len(responses) < 2:
        return 1.0

    embeddings = await embedding_client.embed(responses)
    sims = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = _cosine_similarity(embeddings[i], embeddings[j])
            sims.append(sim)
    return float(np.mean(sims)) if sims else 1.0


def _select_best_response(responses: List[str], embeddings: List[List[float]]) -> str:
    """Seleciona a resposta com maior similaridade média com as outras (mais representativa)."""
    if len(responses) == 1:
        return responses[0]

    best_idx = 0
    best_score = -1.0
    for i in range(len(responses)):
        scores = []
        for j in range(len(responses)):
            if i == j:
                continue
            scores.append(_cosine_similarity(embeddings[i], embeddings[j]))
        avg = float(np.mean(scores))
        if avg > best_score:
            best_score = avg
            best_idx = i
    return responses[best_idx]


async def generate_response(prompt: str, context: str = "local") -> Dict[str, Any]:
    """
    Gera resposta principal. Se self-consistency estiver habilitado, gera N amostras,
    calcula consistência e retorna a mais representativa.

    Retorna dict com:
    - response: texto final
    - self_consistency_score: float ou None
    - num_samples: int
    - model_used: str
    """
    base_url = settings.ollama.local_base_url if context == "local" else settings.ollama.cloud_base_url
    headers = {}
    if context == "cloud" and settings.ollama.cloud_api_key:
        headers["Authorization"] = f"Bearer {settings.ollama.cloud_api_key}"

    model = settings.ollama.local_chat_model if context == "local" else settings.ollama.cloud_chat_model

    sc_cfg = settings.self_consistency
    temp = settings.ollama.temperature
    tout = settings.ollama.timeout

    if sc_cfg.enabled and sc_cfg.num_samples > 1:
        # Gerar N respostas
        responses: List[str] = []
        for i in range(sc_cfg.num_samples):
            text = await generate_single(
                prompt, model, base_url, headers,
                temperature=sc_cfg.temperature, timeout=tout,
            )
            responses.append(text)
            logger.info(f"Self-consistency sample {i+1}/{sc_cfg.num_samples}: {len(text)} chars")

        consistency_score = await compute_consistency_score(responses)
        embeddings = await embedding_client.embed(responses)
        best = _select_best_response(responses, embeddings)

        logger.info(f"Self-consistency score: {consistency_score:.3f}")
        return {
            "response": best,
            "self_consistency_score": round(consistency_score, 4),
            "num_samples": len(responses),
            "model_used": model,
            "all_responses": responses,
        }
    else:
        text = await generate_single(prompt, model, base_url, headers, temperature=temp, timeout=tout)
        return {
            "response": text,
            "self_consistency_score": None,
            "num_samples": 1,
            "model_used": model,
            "all_responses": [text],
        }
