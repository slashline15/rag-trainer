import streamlit as st
import httpx
from rag_sqlite.core.config import settings


def render():
    st.title("Status do Ollama")

    local_url = settings.ollama.local_base_url
    cloud_url = settings.ollama.cloud_base_url

    # --- Status dos endpoints ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖥️ Local")
        _show_endpoint_status(local_url)

    with col2:
        st.subheader("☁️ Cloud")
        if cloud_url and cloud_url != local_url:
            headers = {}
            if settings.ollama.cloud_api_key:
                headers["Authorization"] = f"Bearer {settings.ollama.cloud_api_key}"
            _show_endpoint_status(cloud_url, headers)
        else:
            st.info("Cloud = Local (mesmo endpoint)")

    st.divider()

    # --- Modelos carregados em memória ---
    st.subheader("Modelos em Memória")
    _show_running_models(local_url)

    st.divider()

    # --- Todos os modelos disponíveis ---
    st.subheader("Modelos Disponíveis")
    _show_available_models(local_url)


def _show_endpoint_status(base_url: str, headers: dict = None):
    try:
        r = httpx.get(f"{base_url}/api/tags", timeout=5.0, headers=headers or {})
        r.raise_for_status()
        models = r.json().get("models", [])
        st.success(f"✅ Online — {len(models)} modelos")
    except httpx.ConnectError:
        st.error("❌ Offline — não foi possível conectar")
    except Exception as e:
        st.error(f"❌ Erro: {e}")


def _show_running_models(base_url: str):
    try:
        r = httpx.get(f"{base_url}/api/ps", timeout=5.0)
        r.raise_for_status()
        data = r.json()
        models = data.get("models", [])
        if not models:
            st.info("Nenhum modelo carregado na memória.")
            return
        for m in models:
            name = m.get("name", m.get("model", "?"))
            size = m.get("size", 0)
            size_gb = size / (1024**3) if size else 0
            vram = m.get("size_vram", 0)
            vram_gb = vram / (1024**3) if vram else 0
            expires = m.get("expires_at", "")

            col1, col2, col3 = st.columns(3)
            col1.metric("Modelo", name)
            col2.metric("Tamanho", f"{size_gb:.1f} GB")
            col3.metric("VRAM", f"{vram_gb:.1f} GB")
            if expires:
                st.caption(f"Expira em: {expires}")
    except Exception as e:
        st.warning(f"Não foi possível consultar modelos em memória: {e}")


def _show_available_models(base_url: str):
    try:
        r = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        r.raise_for_status()
        models = r.json().get("models", [])
        if not models:
            st.info("Nenhum modelo instalado.")
            return
        for m in models:
            name = m.get("name", m.get("model", "?"))
            size = m.get("size", 0)
            size_gb = size / (1024**3) if size else 0
            family = m.get("details", {}).get("family", "")
            params = m.get("details", {}).get("parameter_size", "")
            quant = m.get("details", {}).get("quantization_level", "")

            with st.expander(f"**{name}** — {params} {quant} ({size_gb:.1f} GB)"):
                st.json(m.get("details", {}))
    except Exception as e:
        st.warning(f"Não foi possível listar modelos: {e}")
