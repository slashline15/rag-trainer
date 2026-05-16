import streamlit as st
import pandas as pd
from rag_sqlite.core.database import get_db


def render():
    st.title("💭 Histórico de Conversa")
    db = get_db()

    with db._connect() as conn:
        user_rows = conn.execute(
            "SELECT DISTINCT user_id FROM messages ORDER BY user_id"
        ).fetchall()

    if not user_rows:
        st.info("Nenhuma mensagem registrada.")
        return

    user_ids = [r[0] for r in user_rows]
    selected_user = st.selectbox("Selecione o usuário", user_ids)

    with db._connect() as conn:
        rows = conn.execute(
            """SELECT id, role, tag, raw_content, media_type, created_at
               FROM messages
               WHERE user_id = ? AND role IN ('user', 'assistant')
               ORDER BY created_at DESC
               LIMIT 100""",
            (selected_user,),
        ).fetchall()

    if not rows:
        st.info("Nenhuma mensagem para este usuário.")
        return

    # Chat view
    st.subheader("Chat")
    for row in reversed(rows):
        msg_id, role, tag, content, media_type, created_at = row
        if role == "user":
            with st.chat_message("user"):
                st.write(content[:1000])
                st.caption(f"{tag or ''} | {media_type} | {created_at}")
        else:
            with st.chat_message("assistant"):
                st.write(content[:1000])
                st.caption(f"{tag or ''} | {created_at}")

    st.divider()

    # Editor de mensagens
    st.subheader("✏️ Editar Mensagem")
    st.caption("Edite mensagens para corrigir dados de treinamento RAG.")

    msg_ids = [r[0] for r in rows]
    msg_tags = [f"{r[2] or r[0]} ({r[1]})" for r in rows]

    selected_idx = st.selectbox("Mensagem", range(len(msg_ids)), format_func=lambda i: msg_tags[i])
    selected_msg = rows[selected_idx]

    msg_id = selected_msg[0]
    current_content = selected_msg[3] or ""

    new_content = st.text_area("Conteúdo", current_content, height=200, key=f"edit_{msg_id}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Salvar Edição", type="primary"):
            with db._connect() as conn:
                conn.execute(
                    "UPDATE messages SET raw_content = ? WHERE id = ?",
                    (new_content, msg_id),
                )
                conn.commit()
            st.success("Mensagem atualizada!")
            st.rerun()

    with col2:
        if st.button("🗑️ Deletar Mensagem", type="secondary"):
            with db._connect() as conn:
                conn.execute("DELETE FROM messages WHERE id = ?", (msg_id,))
                conn.commit()
            st.success("Mensagem deletada!")
            st.rerun()
