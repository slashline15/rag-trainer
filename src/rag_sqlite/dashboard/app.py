import streamlit as st
from rag_sqlite.core.database import get_db
from rag_sqlite.dashboard.pages import (
    overview, tables, chunks, ratings, metrics,
    conversation, settings, ollama_status, chat_test, logs,
)

st.set_page_config(page_title="RAG SQLite Dashboard", layout="wide")

PAGES = {
    "📊 Visão Geral": overview,
    "💬 Chat de Teste": chat_test,
    "🗂️ Tabelas": tables,
    "🧩 Chunks": chunks,
    "⭐ Avaliações": ratings,
    "📈 Métricas": metrics,
    "💭 Conversa": conversation,
    "🤖 Ollama Status": ollama_status,
    "📋 Logs": logs,
    "⚙️ Configurações": settings,
}


def main():
    st.sidebar.title("RAG SQLite")
    page_name = st.sidebar.radio("Navegação", list(PAGES.keys()))
    page = PAGES[page_name]
    page.render()


main()
