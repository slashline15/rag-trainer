import streamlit as st
import pandas as pd
from rag_sqlite.core.database import get_db


def render():
    st.title("Visão Geral")
    db = get_db()

    with db._connect() as conn:
        total_users = conn.execute("SELECT COUNT(DISTINCT user_id) FROM messages").fetchone()[0] or 0
        total_messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] or 0
        total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] or 0
        total_ratings = conn.execute("SELECT COUNT(*) FROM ratings").fetchone()[0] or 0
        avg_rating = conn.execute("SELECT AVG(rating) FROM ratings").fetchone()[0] or 0
        avg_response_time = conn.execute("SELECT AVG(response_time_ms) FROM ratings").fetchone()[0] or 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Usuários", total_users)
    col2.metric("Mensagens", total_messages)
    col3.metric("Chunks", total_chunks)
    col4.metric("Avaliações", total_ratings)

    col5, col6 = st.columns(2)
    col5.metric("Nota Média", f"{avg_rating:.2f}/5")
    col6.metric("Tempo Médio de Resposta", f"{avg_response_time:.0f}ms")

    st.divider()

    st.subheader("Distribuição por Tipo de Mídia")
    with db._connect() as conn:
        rows = conn.execute(
            "SELECT media_type, COUNT(*) as cnt FROM messages GROUP BY media_type"
        ).fetchall()
    if rows:
        df = pd.DataFrame([dict(r) for r in rows])
        df.columns = ["Tipo", "Quantidade"]
        st.bar_chart(df.set_index("Tipo"))
    else:
        st.info("Nenhuma mensagem registrada ainda.")

    st.subheader("Últimas Mensagens")
    with db._connect() as conn:
        recent = conn.execute(
            "SELECT role, tag, raw_content, media_type, created_at FROM messages ORDER BY created_at DESC LIMIT 10"
        ).fetchall()
    if recent:
        df = pd.DataFrame([dict(r) for r in recent])
        st.dataframe(df, width="stretch")
