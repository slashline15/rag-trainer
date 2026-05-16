import streamlit as st
import pandas as pd
from rag_sqlite.core.database import get_db


TABLES = ["messages", "chunks", "ratings", "module_finance"]


def render():
    st.title("Tabelas do Banco")
    db = get_db()

    selected_table = st.selectbox("Selecione a tabela", TABLES)

    with db._connect() as conn:
        # Obter schema
        schema_rows = conn.execute(
            f"PRAGMA table_info({selected_table})"
        ).fetchall()
        col_names = [r[1] for r in schema_rows]
        schema = pd.DataFrame(schema_rows, columns=["cid", "name", "type", "notnull", "dflt_value", "pk"])
        st.caption("Schema")
        st.dataframe(schema[["name", "type", "pk"]], width="stretch")

        # Filtros
        col1, col2 = st.columns(2)
        with col1:
            limit = st.number_input("Limite de linhas", min_value=10, max_value=5000, value=100)
        with col2:
            user_filter = st.text_input("Filtrar por user_id (deixe vazio para todos)", "")

        query = f"SELECT * FROM {selected_table}"
        params = []
        if user_filter:
            query += " WHERE user_id = ?"
            params.append(int(user_filter))
        query += f" ORDER BY rowid DESC LIMIT {int(limit)}"

        rows = conn.execute(query, params).fetchall()
        if rows:
            # Usar nomes de colunas do PRAGMA para garantir cabeçalhos corretos
            df = pd.DataFrame([dict(r) for r in rows])
            st.dataframe(df, width="stretch")

            # Exportar
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Exportar para CSV",
                data=csv,
                file_name=f"{selected_table}.csv",
                mime="text/csv",
            )
        else:
            st.info("Nenhum dado encontrado.")
