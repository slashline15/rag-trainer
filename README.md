# RAG Trainer

Sistema de **Retrieval-Augmented Generation** pessoal com SQLite, embeddings vetoriais (`sqlite-vec`), busca FTS5 e dashboard Streamlit para visualização e treinamento.

## Funcionalidades

- **Telegram Bot** — ingestão de texto, áudio (Whisper), imagens (Ollama vision), PDFs
- **RAG Híbrido** — busca vetorial (70%) + FTS5 (30%) com re-ranking ponderado
- **Dashboard Streamlit** — chat de teste, visualização de tabelas/chunks, métricas, logs, status Ollama, edição de mensagens
- **Multi-modelo** — local (gemma4:e4b) + cloud, com fallback automático
- **Chunking semântico** — chunks de 2048 chars com overlap de 200
- **Avaliações** — feedback 1-5 com correções que alimentam o prompt

## Setup

```bash
# Clonar
git clone https://github.com/slashline15/rag-trainer.git
cd rag-trainer

# Ambiente virtual
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Dependências
pip install -r requirements.txt
pip install -e .

# Configurar
cp .env.example .env
# Editar .env com seus tokens
```

## Pré-requisitos

- **Python 3.10+**
- **Ollama** rodando localmente com modelos instalados:
  ```bash
  ollama pull nomic-embed-text:v1.5  # embeddings (obrigatório)
  ollama pull gemma4:e4b             # chat + vision local
  ```
- **Telegram Bot Token** via [@BotFather](https://t.me/BotFather)
- **OpenAI API Key** (para transcrição de áudio via Whisper)

## Uso

```bash
# Bot Telegram
python main.py

# Dashboard (separado)
streamlit run src/rag_sqlite/dashboard/app.py
```

## Dashboard

| Página | Descrição |
|--------|-----------|
| 📊 Visão Geral | Métricas globais, distribuição de mídia |
| 💬 Chat de Teste | Testar RAG sem Telegram |
| 🗂️ Tabelas | Browser de todas as tabelas com export CSV |
| 🧩 Chunks | Filtrar e editar chunks indexados |
| ⭐ Avaliações | Histórico de ratings e correções |
| 📈 Métricas | Tempo de resposta, tokens, modelos |
| 💭 Conversa | Histórico de chat com edição de mensagens |
| 🤖 Ollama Status | Modelos em memória, VRAM, endpoints |
| 📋 Logs | Visualizador de logs com filtro por nível |
| ⚙️ Configurações | Modelo, temperatura, timeout (aplicado ao vivo) |

## Arquitetura

```
src/rag_sqlite/
├── core/           # Config, Database (sqlite-vec + FTS5), Embeddings
├── rag/            # Intent classifier, Hybrid retrieval, Prompt builder, Generator
├── pipeline/       # Ingest: text, audio (Whisper), PDF (PyMuPDF), image (Ollama vision)
├── tg/             # Telegram bot: handlers, flows, callbacks
├── dashboard/      # Streamlit: 10 páginas
└── utils/          # Chunker semântico
```

## Configuração via .env

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `OLLAMA_LOCAL_CHAT_MODEL` | `gemma4:e4b` | Modelo local para chat e vision |
| `OLLAMA_CLOUD_CHAT_MODEL` | — | Modelo cloud (fallback) |
| `OLLAMA_DEFAULT_CONTEXT` | `local` | `local` ou `cloud` |
| `TEMPERATURE` | `0.3` | Temperatura da geração |
| `OLLAMA_TIMEOUT` | `300` | Timeout em segundos |
| `CHUNKING_SEMANTIC_CHUNK_SIZE` | `2048` | Tamanho máximo de chunk |
| `CHUNKING_SEMANTIC_OVERLAP` | `200` | Overlap entre chunks |

## Licença

MIT
