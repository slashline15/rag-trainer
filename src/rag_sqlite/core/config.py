import os
from dotenv import load_dotenv
from pathlib import Path

# Carregar .env explicitamente
env_path = Path(".env")
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_bool(key: str, default: bool = False) -> bool:
    val = os.environ.get(key, "").lower()
    return val in ("1", "true", "yes", "on") if val else default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


class OllamaConfig:
    local_base_url: str = _env("OLLAMA_LOCAL_BASE_URL", "http://localhost:11434")
    cloud_base_url: str = _env("OLLAMA_CLOUD_BASE_URL", "")
    cloud_api_key: str = _env("OLLAMA_CLOUD_API_KEY", "")
    embedding_model: str = _env("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:v1.5")
    local_chat_model: str = _env("OLLAMA_LOCAL_CHAT_MODEL", "llama3.2:latest")
    cloud_chat_model: str = _env("OLLAMA_CLOUD_CHAT_MODEL", "gemma4:31b-cloud")
    default_context: str = _env("OLLAMA_DEFAULT_CONTEXT", "local")
    temperature: float = _env_float("TEMPERATURE", 0.3)
    timeout: float = _env_float("OLLAMA_TIMEOUT", 300.0)


class SelfConsistencyConfig:
    enabled: bool = _env_bool("SELF_CONSISTENCY_ENABLED", False)
    num_samples: int = _env_int("SELF_CONSISTENCY_NUM_SAMPLES", 3)
    temperature: float = _env_float("SELF_CONSISTENCY_TEMPERATURE", 0.7)


class TelegramConfig:
    bot_token: str = _env("TELEGRAM_BOT_TOKEN", "")


class OpenAIConfig:
    api_key: str = _env("OPENAI_API_KEY", "")
    whisper_model: str = _env("OPENAI_WHISPER_MODEL", "whisper-1")


class ChunkingConfig:
    intent_max_length: int = _env_int("CHUNKING_INTENT_MAX_LENGTH", 2000)
    intent_overlap: int = _env_int("CHUNKING_INTENT_OVERLAP", 100)
    semantic_chunk_size: int = _env_int("CHUNKING_SEMANTIC_CHUNK_SIZE", 2048)
    semantic_overlap: int = _env_int("CHUNKING_SEMANTIC_OVERLAP", 200)


class DatabaseConfig:
    db_path: str = _env("DATABASE_DB_PATH", "data/memory.db")
    vec_dimension: int = _env_int("DATABASE_VEC_DIMENSION", 768)


class AppConfig:
    app_name: str = _env("APP_NAME", "rag-sqlite")
    debug: bool = _env_bool("DEBUG", False)
    ollama: OllamaConfig = OllamaConfig()
    self_consistency: SelfConsistencyConfig = SelfConsistencyConfig()
    telegram: TelegramConfig = TelegramConfig()
    openai: OpenAIConfig = OpenAIConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    database: DatabaseConfig = DatabaseConfig()


settings = AppConfig()
