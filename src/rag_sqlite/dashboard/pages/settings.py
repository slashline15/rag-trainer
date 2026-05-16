import os
import streamlit as st
from pathlib import Path
from dotenv import set_key, load_dotenv
import httpx

from rag_sqlite.core.config import settings

ENV_PATH = Path(".env")


def _fetch_local_models(base_url: str) -> list:
    try:
        r = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        r.raise_for_status()
        return [m.get("name", "") for m in r.json().get("models", []) if m.get("name")]
    except Exception:
        return []


def render():
    st.title("⚙️ Configurações")
    load_dotenv(ENV_PATH, override=True)

    # Ler .env atual
    current = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                current[k] = v.strip().strip('"').strip("'")

    st.subheader("Modelos Ollama")

    local_url = st.text_input(
        "URL Local Ollama",
        value=current.get("OLLAMA_LOCAL_BASE_URL", "http://localhost:11434"),
    )

    local_models = _fetch_local_models(local_url)
    local_model = st.selectbox(
        "Modelo Local (chat + vision)",
        options=local_models if local_models else [current.get("OLLAMA_LOCAL_CHAT_MODEL", "gemma4:e4b")],
    )
    if not local_models:
        st.caption("⚠️ Ollama local não respondeu.")

    cloud_model = st.text_input(
        "Modelo Cloud",
        value=current.get("OLLAMA_CLOUD_CHAT_MODEL", ""),
    )

    st.subheader("Endpoints")
    cloud_url = st.text_input(
        "URL Cloud Ollama",
        value=current.get("OLLAMA_CLOUD_BASE_URL", ""),
    )

    st.subheader("Comportamento")
    col1, col2, col3 = st.columns(3)
    with col1:
        default_ctx = st.selectbox(
            "Contexto padrão",
            ["local", "cloud"],
            index=0 if current.get("OLLAMA_DEFAULT_CONTEXT", "local") == "local" else 1,
        )
    with col2:
        temperature = st.slider(
            "Temperatura", 0.0, 2.0,
            float(current.get("TEMPERATURE", "0.3")), 0.05,
        )
    with col3:
        timeout = st.number_input(
            "Timeout (s)", min_value=30, max_value=600,
            value=int(float(current.get("OLLAMA_TIMEOUT", "300"))),
            step=30,
        )

    st.subheader("APIs Externas")
    openai_key = st.text_input(
        "OpenAI API Key", value=current.get("OPENAI_API_KEY", ""), type="password",
    )
    cloud_key = st.text_input(
        "Ollama Cloud API Key", value=current.get("OLLAMA_CLOUD_API_KEY", ""), type="password",
    )

    if st.button("💾 Salvar Configurações", type="primary"):
        set_key(ENV_PATH, "OLLAMA_LOCAL_CHAT_MODEL", local_model)
        set_key(ENV_PATH, "OLLAMA_CLOUD_CHAT_MODEL", cloud_model)
        set_key(ENV_PATH, "OLLAMA_LOCAL_BASE_URL", local_url)
        set_key(ENV_PATH, "OLLAMA_CLOUD_BASE_URL", cloud_url)
        set_key(ENV_PATH, "OLLAMA_DEFAULT_CONTEXT", default_ctx)
        set_key(ENV_PATH, "TEMPERATURE", str(temperature))
        set_key(ENV_PATH, "OLLAMA_TIMEOUT", str(timeout))
        if openai_key:
            set_key(ENV_PATH, "OPENAI_API_KEY", openai_key)
        if cloud_key:
            set_key(ENV_PATH, "OLLAMA_CLOUD_API_KEY", cloud_key)

        # Recarregar env vars no processo atual
        os.environ["OLLAMA_LOCAL_CHAT_MODEL"] = local_model
        os.environ["OLLAMA_CLOUD_CHAT_MODEL"] = cloud_model
        os.environ["OLLAMA_LOCAL_BASE_URL"] = local_url
        os.environ["OLLAMA_CLOUD_BASE_URL"] = cloud_url
        os.environ["OLLAMA_DEFAULT_CONTEXT"] = default_ctx
        os.environ["TEMPERATURE"] = str(temperature)
        os.environ["OLLAMA_TIMEOUT"] = str(timeout)

        # Atualizar settings em memória
        settings.ollama.local_chat_model = local_model
        settings.ollama.cloud_chat_model = cloud_model
        settings.ollama.local_base_url = local_url
        settings.ollama.cloud_base_url = cloud_url
        settings.ollama.default_context = default_ctx
        settings.ollama.temperature = temperature
        settings.ollama.timeout = float(timeout)

        st.success("✅ Configurações salvas e aplicadas! O dashboard já usa os novos valores.")
        st.info("⚠️ O bot do Telegram precisa ser reiniciado manualmente para aplicar as mudanças.")

    st.divider()
    st.caption("Valores atuais em memória:")
    st.json({
        "model_local": settings.ollama.local_chat_model,
        "model_cloud": settings.ollama.cloud_chat_model,
        "context": settings.ollama.default_context,
        "temperature": settings.ollama.temperature,
        "timeout": settings.ollama.timeout,
    })
