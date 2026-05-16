import streamlit as st
import pandas as pd
from rag_sqlite.core.database import get_db


def render():
    st.title("Chunks e Embeddings")
    db = get_db()

    col1, col2, col3 = st.columns(3)
    with col1:
        user_filter = st.text_input("user_id", "")
    with col2:
        module_filter = st.selectbox("Módulo", ["Todos", "general", "finance", "obra", "shopping"])
    with col3:
        source_type = st.selectbox("Tipo de fonte", ["Todos", "message", "pdf", "image", "audio"])

    with db._connect() as conn:
        query = "SELECT id, user_id, content, source_type, source_id, module, tags, is_public, confidence_at_index, created_at FROM chunks WHERE 1=1"
        params = []

        if user_filter:
            query += " AND user_id = ?"
            params.append(int(user_filter))
        if module_filter != "Todos":
            query += " AND module = ?"
            params.append(module_filter)
        if source_type != "Todos":
            query += " AND source_type = ?"
            params.append(source_type)

        query += " ORDER BY created_at DESC LIMIT 200"
        rows = conn.execute(query, params).fetchall()

    if not rows:
        st.info("Nenhum chunk encontrado.")
        return

    df = pd.DataFrame(rows, columns=[
        "id", "user_id", "content", "source_type", "source_id",
        "module", "tags", "is_public", "confidence", "created_at"
    ])
    st.dataframe(df, width="stretch")

    st.divider()
    st.subheader("Editar Chunk")
    chunk_id = st.text_input("ID do chunk para editar")
    if chunk_id:
        with db._connect() as conn:
            row = conn.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        if row:
            d = dict(row)
            new_content = st.text_area("Conteúdo", d.get("content", ""), height=200)
            new_tags = st.text_input("Tags", d.get("tags", ""))
            new_public = st.checkbox("Público", bool(d.get("is_public", 0)))
            if st.button("Salvar alterações"):
                with db._connect() as conn:
                    conn.execute(
                        "UPDATE chunks SET content = ?, tags = ?, is_public = ? WHERE id = ?",
                        (new_content, new_tags, int(new_public), chunk_id)
                    )
                    conn.commit()
                st.success("Chunk atualizado!")
        else:
            st.warning("Chunk não encontrado.")
