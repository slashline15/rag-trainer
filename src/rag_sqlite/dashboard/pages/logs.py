import streamlit as st
from pathlib import Path
from datetime import datetime


LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "rag_sqlite.log"


def render():
    st.title("📋 Logs do Sistema")

    # Logs em arquivo (se existir)
    if LOG_FILE.exists():
        st.subheader("Arquivo de Log")
        log_text = LOG_FILE.read_text(encoding="utf-8", errors="replace")
        lines = log_text.strip().split("\n")

        # Filtros
        col1, col2 = st.columns(2)
        with col1:
            level_filter = st.selectbox("Nível", ["Todos", "ERROR", "WARNING", "INFO", "DEBUG"])
        with col2:
            num_lines = st.slider("Últimas N linhas", 50, 1000, 200, 50)

        # Filtrar e mostrar últimas N linhas
        filtered = lines
        if level_filter != "Todos":
            filtered = [l for l in lines if f"| {level_filter}" in l]

        display = filtered[-num_lines:]
        st.text_area(
            "Log",
            value="\n".join(display),
            height=500,
            disabled=True,
        )
        st.caption(f"Mostrando {len(display)} de {len(filtered)} linhas (total: {len(lines)})")

        if st.button("🔄 Atualizar"):
            st.rerun()
    else:
        st.info(
            f"Nenhum arquivo de log encontrado em `{LOG_FILE}`.\n\n"
            "Para habilitar logs em arquivo, adicione no início do `main.py`:\n"
            "```python\n"
            "from loguru import logger\n"
            "logger.add('logs/rag_sqlite.log', rotation='10 MB')\n"
            "```"
        )

    st.divider()

    # Logs do banco — últimas interações
    st.subheader("Últimas Interações (do Banco)")
    from rag_sqlite.core.database import get_db
    db = get_db()

    with db._connect() as conn:
        rows = conn.execute("""
            SELECT r.created_at, r.model_used, r.response_time_ms,
                   r.tokens_input, r.tokens_output, r.rating,
                   r.retrieval_score, m.raw_content as pergunta
            FROM ratings r
            LEFT JOIN messages m ON m.id = r.message_id
            ORDER BY r.created_at DESC
            LIMIT 50
        """).fetchall()

    if rows:
        import pandas as pd
        df = pd.DataFrame([dict(r) for r in rows])
        st.dataframe(df, width="stretch")
    else:
        st.info("Nenhuma interação registrada ainda.")
