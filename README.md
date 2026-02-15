# Local Telegram RAG

Local RAG app for Telegram history with:
- Qdrant (Docker) with binary quantization
- llama-cpp-python for embeddings and generation
- Telethon ingestion (full/sync/full_then_sync)
- Streamlit UI with full debug inspector

## 1) Setup

1. Create virtual environment and install dependencies:
   - Windows (PowerShell):
     - `python -m venv .venv`
     - `.venv\Scripts\Activate.ps1`
     - `pip install -r requirements.txt`
   - Linux:
     - `python3 -m venv .venv`
     - `source .venv/bin/activate`
     - `pip install -r requirements.txt`
2. Install llama-cpp-python:
   - Linux: `pip install -r requirements-llm.txt`
   - Windows: install "Visual Studio Build Tools" (Desktop development with C++), then run:
     - `pip install -r requirements-llm.txt`
3. Copy `.env.example` to `.env` and fill values.
4. Put GGUF models into `./models/`.
5. Ensure Docker Desktop / Docker Engine is running.

## 2) Run

- Server (Qdrant + ingestion sync): `python run_server.py --ingest-mode sync`
- Client (Streamlit): `python run_client.py`
- All-in-one: `python run_all.py`

Shell wrappers:
- Windows: `scripts/start_server.ps1`, `scripts/start_client.ps1`, `scripts/start_all.ps1`
- Linux: `scripts/start_server.sh`, `scripts/start_client.sh`, `scripts/start_all.sh`

## 3) Ingestion Modes

- `full`: full historical backfill
- `sync`: catch-up first, then live updates
- `full_then_sync`: full backfill then sync

## 4) Tests

- `pytest -q`
