import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from rag_sqlite.core.database import get_db


def render():
    st.title("Métricas de Performance")
    db = get_db()

    with db._connect() as conn:
        rows = conn.execute(
            """SELECT created_at, response_time_ms, tokens_input, tokens_output,
                      model_used, self_consistency_score, retrieval_score, final_confidence, rating
               FROM ratings
               ORDER BY created_at"""
        ).fetchall()

    if not rows:
        st.info("Nenhuma métrica disponível ainda. Interaja com o bot para gerar dados.")
        return

    df = pd.DataFrame(rows, columns=[
        "created_at", "response_time_ms", "tokens_input", "tokens_output",
        "model_used", "self_consistency_score", "retrieval_score", "final_confidence", "rating"
    ])
    df["created_at"] = pd.to_datetime(df["created_at"])

    st.subheader("Confiança ao Longo do Tempo")
    st.line_chart(df.set_index("created_at")[["final_confidence", "retrieval_score"]])

    st.subheader("Tempo de Resposta (ms)")
    st.line_chart(df.set_index("created_at")["response_time_ms"])

    st.subheader("Consumo de Tokens")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Input", int(df["tokens_input"].sum()))
    with col2:
        st.metric("Total Output", int(df["tokens_output"].sum()))
    st.bar_chart(df[["tokens_input", "tokens_output"]])

    st.subheader("Uso de Modelos")
    model_counts = df["model_used"].value_counts()
    st.bar_chart(model_counts)

    st.subheader("Self-Consistency Score (quando habilitado)")
    sc_df = df[df["self_consistency_score"].notna()]
    if not sc_df.empty:
        st.line_chart(sc_df.set_index("created_at")["self_consistency_score"])
    else:
        st.info("Self-consistency não habilitado ou sem amostras.")

    st.subheader("Relação Confiança × Nota do Usuário")
    corr = df[["final_confidence", "rating"]].corr().iloc[0, 1]
    st.caption(f"Correlação: {corr:.3f}")
    st.scatter_chart(df, x="final_confidence", y="rating")
