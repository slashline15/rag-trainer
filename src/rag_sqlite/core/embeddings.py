import httpx
from typing import List
from loguru import logger

from rag_sqlite.core.config import settings


class EmbeddingClient:
    def __init__(self):
        self.local_url = settings.ollama.local_base_url
        self.cloud_url = settings.ollama.cloud_base_url
        self.cloud_key = settings.ollama.cloud_api_key
        self.model = settings.ollama.embedding_model
        self._client = httpx.AsyncClient(timeout=60.0)

    async def embed(self, texts: List[str], context: str = "local") -> List[List[float]]:
        if not texts:
            return []

        base_url = self.local_url if context == "local" else self.cloud_url
        headers = {}
        if context == "cloud" and self.cloud_key:
            headers["Authorization"] = f"Bearer {self.cloud_key}"

        embeddings: List[List[float]] = []
        for text in texts:
            emb = await self._embed_single(text, base_url, headers)
            if emb is None and context == "local" and self.cloud_url:
                logger.warning("Local embedding failed, trying cloud fallback...")
                cloud_headers = {}
                if self.cloud_key:
                    cloud_headers["Authorization"] = f"Bearer {self.cloud_key}"
                emb = await self._embed_single(text, self.cloud_url, cloud_headers)
            if emb is None:
                dim = settings.database.vec_dimension
                emb = [0.0] * dim
            embeddings.append(emb)
        return embeddings

    async def _embed_single(self, text: str, base_url: str, headers: dict) -> List[float] | None:
        try:
            resp = await self._client.post(
                f"{base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["embedding"]
        except Exception as e:
            logger.error(f"Embedding failed for text snippet: {e}")
            return None

    async def embed_single(self, text: str, context: str = "local") -> List[float]:
        results = await self.embed([text], context)
        return results[0] if results else [0.0] * settings.database.vec_dimension


embedding_client = EmbeddingClient()
