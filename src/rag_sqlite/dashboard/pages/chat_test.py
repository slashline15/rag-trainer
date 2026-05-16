import asyncio
import streamlit as st
from rag_sqlite.core.database import get_db
from rag_sqlite.core.config import settings
from rag_sqlite.rag.engine import RAGEngine
from rag_sqlite.pipeline.ingest import ingest_text


def render():
    st.title("💬 Chat de Teste")
    st.caption("Envie mensagens de teste direto pelo dashboard — sem precisar do Telegram.")

    db = get_db()

    # Selecionar user_id para teste
    test_user = st.number_input("User ID para teste", value=0, min_value=0, step=1)
    if test_user == 0:
        st.info("Defina um User ID para começar o chat de teste.")
        return

    # Carregar histórico existente
    history = db.get_recent_history(test_user, limit=30)

    # Mostrar histórico em formato de chat
    for h in reversed(history):
        role = h.get("role", "user")
        tag = h.get("tag", "")
        content = h.get("raw_content", "")
        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
                if tag:
                    st.caption(f"{tag} • {h.get('created_at', '')}")
        else:
            with st.chat_message("assistant"):
                st.markdown(content)
                if tag:
                    st.caption(f"{tag} • {h.get('created_at', '')}")

    # Input de mensagem
    user_input = st.chat_input("Digite uma mensagem de teste...")
    if user_input:
        # Mostrar mensagem do usuário imediatamente
        with st.chat_message("user"):
            st.markdown(user_input)

        # Processar
        with st.chat_message("assistant"):
            with st.spinner("Processando..."):
                try:
                    response = _process_message(test_user, user_input)
                    st.markdown(response["response"])

                    # Métricas da resposta
                    col1, col2, col3 = st.columns(3)
                    col1.caption(f"⏱️ {response.get('response_time_ms', 0)}ms")
                    col2.caption(f"🤖 {response.get('model_used', '?')}")
                    col3.caption(f"📊 {len(response.get('chunks_used', []))} chunks")
                except Exception as e:
                    st.error(f"Erro: {e}")

        st.rerun()


def _process_message(user_id: int, text: str) -> dict:
    """Executa o pipeline RAG completo de forma síncrona (para Streamlit)."""
    engine = RAGEngine()
    db = get_db()

    loop = asyncio.new_event_loop()
    try:
        # Ingestão
        loop.run_until_complete(ingest_text(user_id, text))

        # Histórico
        history = db.get_recent_history(user_id, limit=10)

        # RAG
        result = loop.run_until_complete(engine.process(user_id, text, history=history))

        # Salvar resposta do bot
        bot_tag = db.generate_next_tag(user_id)
        db.insert_message(
            msg_id=f"dashboard_bot_{user_id}_{bot_tag}",
            user_id=user_id,
            raw_content=result["response"],
            media_type="text",
            role="assistant",
            tag=bot_tag,
        )

        return result
    finally:
        loop.close()
