import sqlite3
import sqlite_vec
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
from loguru import logger

from rag_sqlite.core.config import settings


SCHEMA_SQL = """
-- Mensagens brutas
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    role TEXT DEFAULT 'user',
    tag TEXT,
    raw_content TEXT,
    media_type TEXT,
    media_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Chunks (conteúdo textual indexado)
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    source_type TEXT,
    source_id TEXT,
    source_path TEXT,
    module TEXT DEFAULT 'general',
    tags TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_public BOOLEAN DEFAULT 0,
    confidence_at_index REAL
);

-- Tabela virtual sqlite-vec para embeddings
-- Criada dinamicamente no init_db

-- Tabela virtual FTS5 para busca textual
-- Criada dinamicamente no init_db

-- Avaliações
CREATE TABLE IF NOT EXISTS ratings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    message_id TEXT NOT NULL,
    response_id TEXT NOT NULL,
    rating INTEGER CHECK(rating BETWEEN 1 AND 5),
    correction_type TEXT,
    correction_text TEXT,
    response_time_ms INTEGER,
    tokens_input INTEGER,
    tokens_output INTEGER,
    model_used TEXT,
    self_consistency_score REAL,
    retrieval_score REAL,
    final_confidence REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Módulo financeiro (template)
CREATE TABLE IF NOT EXISTS module_finance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    transaction_date DATE,
    category TEXT,
    amount REAL,
    description TEXT,
    source_message_id TEXT,
    is_public BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


class Database:
    def __init__(self, db_path: str = settings.database.db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self._connect() as conn:
            # Schema base
            conn.executescript(SCHEMA_SQL)

            # Tabela virtual sqlite-vec (nomic-embed-text v1.5 = 768 dim)
            dim = settings.database.vec_dimension
            conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
                    id TEXT PRIMARY KEY,
                    user_id INTEGER,
                    content_vec FLOAT[{dim}],
                    +source_type TEXT,
                    +module TEXT,
                    +tags TEXT,
                    +is_public INTEGER,
                    +source_id TEXT
                );
            """)

            # Tabela virtual FTS5
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    content,
                    source_id UNINDEXED,
                    tags UNINDEXED,
                    content='chunks',
                    content_rowid='rowid'
                );
            """)

            # Triggers para manter FTS5 sincronizado
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunks_fts(rowid, content, source_id, tags)
                    VALUES (new.rowid, new.content, new.source_id, new.tags);
                END;
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, content, source_id, tags)
                    VALUES ('delete', old.rowid, old.content, old.source_id, old.tags);
                END;
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, content, source_id, tags)
                    VALUES ('delete', old.rowid, old.content, old.source_id, old.tags);
                    INSERT INTO chunks_fts(rowid, content, source_id, tags)
                    VALUES (new.rowid, new.content, new.source_id, new.tags);
                END;
            """)

            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")

    def insert_message(self, msg_id: str, user_id: int, raw_content: str,
                       media_type: str = "text", media_path: Optional[str] = None,
                       role: str = "user", tag: Optional[str] = None) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO messages (id, user_id, role, tag, raw_content, media_type, media_path)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (msg_id, user_id, role, tag, raw_content, media_type, media_path),
            )
            conn.commit()

    def generate_next_tag(self, user_id: int) -> str:
        """Gera tag sequencial curta (#m1, #m2, ...) por usuário."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(CAST(SUBSTR(tag, 3) AS INTEGER)) FROM messages WHERE user_id = ? AND tag LIKE '#m%'",
                (user_id,),
            ).fetchone()
            n = (row[0] or 0) + 1
            return f"#m{n}"

    def get_recent_history(self, user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Retorna as últimas N mensagens (user + assistant) para contexto de conversa."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT role, tag, raw_content, created_at
                   FROM messages
                   WHERE user_id = ? AND role IN ('user', 'assistant')
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (user_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def insert_chunk(self, chunk_id: str, user_id: int, content: str,
                     embedding: List[float], source_type: str = "message",
                     source_id: str = "", module: str = "general",
                     tags: str = "", is_public: bool = False,
                     confidence: float = 1.0,
                     source_path: Optional[str] = None) -> None:
        with self._connect() as conn:
            # Inserir na tabela base
            conn.execute(
                """INSERT INTO chunks (id, user_id, content, source_type, source_id, source_path, module, tags, is_public, confidence_at_index)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (chunk_id, user_id, content, source_type, source_id, source_path, module, tags, is_public, confidence),
            )
            # Inserir na tabela virtual vec
            conn.execute(
                """INSERT INTO chunks_vec (id, user_id, content_vec, source_type, module, tags, is_public, source_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (chunk_id, user_id, sqlite_vec.serialize_float32(embedding), source_type, module, tags, is_public, source_id),
            )
            conn.commit()

    def search_vec(self, user_id: int, embedding: List[float], top_k: int = 5,
                   module: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Busca vetorial KNN via sqlite-vec.
        Como vec0 não permite WHERE em colunas auxiliares no KNN,
        buscamos os top-k globais e filtramos no JOIN com a tabela base.
        """
        with self._connect() as conn:
            # KNN query na virtual table (sem WHERE em colunas auxiliares)
            knn_sql = """
                SELECT id, distance
                FROM chunks_vec
                WHERE content_vec MATCH ? AND k = ?
                ORDER BY distance
            """
            knn_rows = conn.execute(knn_sql, (
                sqlite_vec.serialize_float32(embedding), top_k * 4,  # buscar mais para filtrar
            )).fetchall()

            if not knn_rows:
                return []

            ids = [r[0] for r in knn_rows]
            distances = {r[0]: r[1] for r in knn_rows}

            # Filtrar na tabela base
            placeholders = ",".join("?" * len(ids))
            query = f"""
                SELECT
                    id, content, source_type, source_id, source_path, module, tags,
                    is_public, confidence_at_index
                FROM chunks
                WHERE id IN ({placeholders})
                  AND (user_id = ? OR is_public = 1)
            """
            params: List[Any] = ids + [user_id]

            if module:
                query += " AND module = ?"
                params.append(module)

            rows = conn.execute(query, params).fetchall()
            results = []
            for r in rows:
                d = dict(r)
                d["distance"] = distances.get(d["id"], 999.0)
                results.append(d)

            # Ordenar por distância e limitar a top_k
            results.sort(key=lambda x: x["distance"])
            return results[:top_k]

    @staticmethod
    def _sanitize_fts_query(query_text: str) -> str:
        """
        Remove caracteres especiais do FTS5 para evitar syntax error.
        Mantém apenas palavras alfanuméricas e espaços.
        """
        import re
        # Remover pontuação FTS5-problemática: ? " * ^ . = < > ( ) , & | ~ ! / \
        cleaned = re.sub(r'[^\w\s#]', ' ', query_text)
        # Normalizar espaços
        cleaned = ' '.join(cleaned.split())
        return cleaned

    def search_fts(self, user_id: int, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Busca FTS5 com filtro de segurança por user_id.
        Inclui user_id na projeção para que o filtro pós-query funcione corretamente.
        """
        safe_query = self._sanitize_fts_query(query_text)
        if not safe_query:
            return []

        # Escapar aspas duplas remanescentes
        safe_query = safe_query.replace('"', '""')
        limit = top_k * 3
        with self._connect() as conn:
            sql = f"""
                SELECT
                    c.id,
                    c.user_id,
                    c.content,
                    c.source_type,
                    c.source_id,
                    c.source_path,
                    c.module,
                    c.tags,
                    c.is_public,
                    c.confidence_at_index,
                    rank
                FROM chunks_fts AS fts
                JOIN chunks AS c ON c.rowid = fts.rowid
                WHERE chunks_fts MATCH "{safe_query}"
                ORDER BY rank
                LIMIT {limit}
            """
            rows = conn.execute(sql).fetchall()
            # Filtrar por user_id/is_public em Python (segurança)
            results = []
            for r in rows:
                d = dict(r)
                if d.get("user_id") == user_id or d.get("is_public") == 1:
                    results.append(d)
                if len(results) >= top_k:
                    break
            return results

    def get_chunks_by_source(self, source_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE source_id = ? ORDER BY id", (source_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_message_by_id(self, msg_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM messages WHERE id = ?", (msg_id,)).fetchone()
            return dict(row) if row else None

    def insert_rating(self, data: Dict[str, Any]) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO ratings
                   (user_id, message_id, response_id, rating, correction_type, correction_text,
                    response_time_ms, tokens_input, tokens_output, model_used,
                    self_consistency_score, retrieval_score, final_confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    data["user_id"], data["message_id"], data["response_id"],
                    data.get("rating"), data.get("correction_type"), data.get("correction_text"),
                    data.get("response_time_ms"), data.get("tokens_input"), data.get("tokens_output"),
                    data.get("model_used"), data.get("self_consistency_score"),
                    data.get("retrieval_score"), data.get("final_confidence"),
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_ratings_for_context(self, user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM ratings
                   WHERE user_id = ?
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (user_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_table_columns(self, table_name: str) -> List[str]:
        """Retorna lista de nomes de colunas de uma tabela."""
        with self._connect() as conn:
            rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            return [r[1] for r in rows]


# Singleton
_db_instance: Optional[Database] = None


def get_db() -> Database:
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance
