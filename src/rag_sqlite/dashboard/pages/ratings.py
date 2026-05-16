import streamlit as st
import pandas as pd
from rag_sqlite.core.database import get_db


def render():
    st.title("Avaliações e Correções")
    db = get_db()

    with db._connect() as conn:
        rows = conn.execute(
            """SELECT id, user_id, message_id, response_id, rating,
                      correction_type, correction_text, response_time_ms,
                      tokens_input, tokens_output, model_used,
                      self_consistency_score, retrieval_score, final_confidence,
                      created_at
               FROM ratings
               ORDER BY created_at DESC
               LIMIT 500"""
        ).fetchall()

    if not rows:
        st.info("Nenhuma avaliação registrada.")
        return

    df = pd.DataFrame(rows, columns=[
        "id", "user_id", "message_id", "response_id", "rating",
        "correction_type", "correction_text", "response_time_ms",
        "tokens_input", "tokens_output", "model_used",
        "self_consistency_score", "retrieval_score", "final_confidence",
        "created_at"
    ])
    st.dataframe(df, width="stretch")

    st.divider()
    st.subheader("Distribuição de Notas")
    hist = df["rating"].value_counts().sort_index()
    st.bar_chart(hist)

    st.subheader("Correções Pendentes")
    corrections = df[df["correction_type"].notna()]
    if not corrections.empty:
        st.dataframe(corrections, width="stretch")
    else:
        st.info("Nenhuma correção registrada.")
