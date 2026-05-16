import shutil
from pathlib import Path
from typing import Optional
from datetime import datetime

from rag_sqlite.core.config import settings


MEDIA_BASE = Path("data/users")


def save_media(user_id: int, file_name: str, file_data: bytes,
               media_type: str = "document") -> str:
    """Salva arquivo localmente e retorna caminho absoluto."""
    now = datetime.now()
    folder = MEDIA_BASE / str(user_id) / media_type / now.strftime("%Y/%m")
    folder.mkdir(parents=True, exist_ok=True)

    # Sanitizar nome
    safe_name = Path(file_name).name.replace(" ", "_")
    path = folder / safe_name

    # Evitar colisão
    counter = 1
    stem = path.stem
    suffix = path.suffix
    while path.exists():
        path = folder / f"{stem}_{counter}{suffix}"
        counter += 1

    path.write_bytes(file_data)
    return str(path)


def get_media_path(relative_or_absolute: str) -> Path:
    p = Path(relative_or_absolute)
    if p.is_absolute():
        return p
    return Path(settings.database.db_path).parent.parent / p
