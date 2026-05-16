import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_sqlite.core.database import Database
from rag_sqlite.core.config import settings


def test_database_init():
    import tempfile, os
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = Database(path)
    assert db.db_path is not None
    os.unlink(path)
    print("Database init OK")


def test_insert_and_search():
    import tempfile, os
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = Database(path)
    db.insert_message("msg_1", 123, "Teste de mensagem #obra", "text")
    db.insert_chunk("msg_1#n1", 123, "Conteúdo de teste", [0.1] * 768, "message", "msg_1", "general", "#obra")

    rows = db.get_chunks_by_source("msg_1")
    assert len(rows) == 1
    assert rows[0]["content"] == "Conteúdo de teste"
    os.unlink(path)
    print("Insert and retrieve OK")


if __name__ == "__main__":
    test_database_init()
    test_insert_and_search()
