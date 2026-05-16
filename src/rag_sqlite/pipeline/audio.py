import httpx
from pathlib import Path
from typing import Optional
from loguru import logger

from rag_sqlite.core.config import settings


async def transcribe_audio(file_path: str) -> str:
    """Transcreve áudio via OpenAI Whisper API usando requests diretos."""
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {settings.openai.api_key}"}

    path = Path(file_path)
    mime_types = {
        ".ogg": "audio/ogg",
        ".oga": "audio/ogg",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/m4a",
        ".webm": "audio/webm",
        ".mp4": "audio/mp4",
    }
    mime = mime_types.get(path.suffix.lower(), "audio/mpeg")
    logger.info(f"Sending {path.name} ({mime}) to Whisper API...")
    with open(path, "rb") as f:
        files = {"file": (path.name, f, mime)}
        data = {"model": settings.openai.whisper_model, "language": "pt"}

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, headers=headers, files=files, data=data)
            logger.info(f"Whisper API status: {resp.status_code}")
            if resp.status_code != 200:
                logger.error(f"Whisper API error body: {resp.text[:500]}")
            resp.raise_for_status()
            result = resp.json()

    transcription = result.get("text", "").strip()
    logger.info(f"Transcribed {path.name}: {len(transcription)} chars")
    return transcription


async def light_grammar_fix(text: str, context: str = "local") -> str:
    """
    Envia transcrição para modelo Ollama para leve correção gramatical.
    Preserva conteúdo factual, apenas ajusta pontuação e concordância.
    """
    prompt = (
        "Você é um corretor de textos. Faça APENAS correções gramaticais e de pontuação leves. "
        "NÃO altere o conteúdo factual, nomes próprios, números ou datas. "
        "NÃO adicione comentários. Responda apenas com o texto corrigido.\n\n"
        f"Texto:\n{text}\n\nTexto corrigido:"
    )

    base_url = settings.ollama.local_base_url if context == "local" else settings.ollama.cloud_base_url
    headers = {}
    if context == "cloud" and settings.ollama.cloud_api_key:
        headers["Authorization"] = f"Bearer {settings.ollama.cloud_api_key}"

    model = settings.ollama.local_chat_model if context == "local" else settings.ollama.cloud_chat_model

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{base_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.1}},
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

    corrected = data.get("response", text).strip()
    if not corrected:
        corrected = text
    logger.info(f"Grammar fix applied: {len(text)} -> {len(corrected)} chars")
    return corrected
