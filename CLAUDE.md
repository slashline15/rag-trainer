# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a **multi-user RAG (Retrieval-Augmented Generation) system** built on SQLite with vector search (`sqlite-vec`), full-text search (FTS5), and structured SQL metadata. It is designed for personal knowledge management with data ingestion from Telegram (text, audio, images, PDFs). The system runs entirely local or hybrid (Ollama local + cloud), with a Streamlit dashboard for visualization and control.

**Key design principle**: zero commands for the end user. The AI classifies intent, decides whether to ingest or query, picks the right tables/tools, and responds naturally. All data is isolated per user via `user_id` filtering; public data is flagged with `is_public=1`.

---

## Common Commands

### Setup
```bash
pip install -r requirements.txt
pip install -e .
```

### Run Tests (use venv recommended)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
pip install -r requirements.txt

python tests/test_fixes.py
python tests/test_database.py
python tests/test_rag_engine.py
python tests/test_telegram_import.py
python tests/test_dashboard_import.py
python tests/test_end_to_end.py
```

### Run Telegram Bot
```bash
python -m rag_sqlite.tg.bot
```

### Run Streamlit Dashboard
```bash
streamlit run src/rag_sqlite/dashboard/app.py
```

### Environment Variables
Copy `.env.example` to `.env` and fill:
- `TELEGRAM_BOT_TOKEN`
- `OPENAI_API_KEY` (for Whisper transcription)
- `OLLAMA_CLOUD_BASE_URL` and `OLLAMA_CLOUD_API_KEY` (optional, for heavy cloud models)

---

## High-Level Architecture

### 1. Two-Level Chunking

Every incoming message/document goes through **two chunking stages**:

- **Intent Chunker (Level 1)**: coarse-grained, preserves broad context. Used by the `classify_intent` step to understand if the user wants to ingest data, ask a question, or do both. Config: `intent_max_length=2000`.
- **Semantic Chunker (Level 2)**: fine-grained, splits by paragraphs and sentences, detects multiple topics within the same message. Each chunk gets an embedding via `nomic-embed-text:v1.5` (768d). Config: `semantic_chunk_size=1024`, `semantic_overlap=150`.

Chunks are stored with sequential IDs like `msg_123#n1`, `msg_123#n2`, so the AI and the user can reference specific fragments with `#n1` syntax.

### 2. RAG Hybrid Engine

The retrieval layer combines three strategies with **weighted re-ranking**:

1. **Vector search** (`sqlite-vec`): semantic similarity over chunk embeddings. **Weight: 0.7**
2. **Full-text search** (FTS5): exact keyword/hashtag matching (e.g., `#obra`, `#material`). **Weight: 0.3**
3. **Structured SQL**: queries over module tables (`module_finance`, `module_obra`, etc.) for exact dates, amounts, categories.

All searches are filtered by: `user_id = ? OR is_public = 1`. Private data from other users never leaks into prompts.

**Auto-match filtering**: chunks whose vector distance is ≤ 0.05 from the query are filtered out to prevent the user's own message from appearing as the top result.

### 3. Database Schema (Single SQLite File)

Tables:
- `messages`: raw incoming content (text, transcriptions, file paths).
- `chunks` + `chunks_vec` (virtual, sqlite-vec) + `chunks_fts` (virtual, FTS5): semantic chunks with embeddings. Includes `source_path` for file traceability.
- `ratings`: user feedback (1–5), corrections, timing/token metrics, and retrieval scores.
- `module_*` (extensible): structured tables per domain (finance, construction, shopping).

**Important**: `sqlite-vec` virtual tables do **not** accept `BOOLEAN` as an extra column type. Use `INTEGER` for `is_public` in `vec0` declarations. The base SQLite tables can still use `BOOLEAN`.

**Schema changes**: In v1, **delete the .db file and recreate** instead of writing migrations. User explicitly prefers this approach.

### 4. Self-Consistency (Optional)

When enabled in config (`self_consistency.enabled = true`), each query is answered N times (default 3). The final response is chosen by majority/consensus, and an internal consistency score is computed from embedding similarity across the N answers.

### 5. Response Generation

The system prompt instructs the LLM to:
- **Prioritize the user's knowledge base** when relevant context exists
- **Combine information from multiple chunks** for richer answers
- **Use general knowledge** when no relevant context is found (don't refuse to answer)
- **Cite tags inline** using the real `#mX` format from message tags

Confidence scores are tracked internally for the dashboard but **not exposed in responses** (per user feedback).

### 6. Configuration System

Configuration lives in `src/rag_sqlite/core/config.py` using plain Python classes with env var loading. It supports switching between local and cloud Ollama endpoints, toggling self-consistency, and tuning chunking parameters.

---

## Project Structure

```
src/rag_sqlite/
  core/
    config.py         # Settings from env vars
    database.py       # SQLite + sqlite-vec + FTS5 abstraction
    embeddings.py     # Async Ollama embedding client
  utils/
    chunker.py        # IntentChunker + SemanticChunker + hashtag parser
  pipeline/           # Ingestion: Whisper, PDF, image, text
  rag/                # Intent classifier + hybrid retrieval + response generator
  tg/                 # python-telegram-bot handlers + inline buttons (sub-modules: router, text, media, callbacks, corrections, ratings)
  dashboard/          # Streamlit visualizer
  modules/            # Domain-specific SQL schemas and parsers
```

---

## Notes for Future Development

- **Ollama models**: `nomic-embed-text:v1.5` is hardcoded for embeddings (768d). Chat models are swappable via config (`local_chat_model`, `cloud_chat_model`).
- **Telegram flow**: every AI response ends with inline buttons 1–5 (layout: 3 on top, 2 on bottom, using emojis 😡😠😐 / 🙂😁). If user clicks 1 or 2, a correction flow opens (`[sim]` / `[não]` → `[ajuste]` / `[comentário]`). All buttons disappear after click.
- **Hashtag references**: the AI cites `#mX` tags in responses. In Telegram, tapping the hashtag jumps to the referenced message.
- **Metrics stored**: every interaction logs `response_time_ms`, `tokens_input/output`, `model_used`, and retrieval scores in `ratings`.
- **Intent classification**: uses regex-based JSON extraction as fallback when Ollama returns malformed JSON. Falls back to keyword-based heuristic if LLM is completely unavailable.
