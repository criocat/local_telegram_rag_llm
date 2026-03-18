# Local Telegram RAG

*A fully local, privacy-first Retrieval-Augmented Generation (RAG) application to chat with your own Telegram history.*

---

## 🌟 Overview

**Local Telegram RAG** lets you index your personal Telegram chat history into a local vector database and talk to it using open-source Large Language Models (LLMs). Everything runs locally on your machine—no data is sent to external APIs like OpenAI or Anthropic.

The application automatically decides whether to search your Telegram history or answer from its general knowledge, remembers conversation context, and streams responses to a web interface.

## ✨ Key Features

- **🔒 100% Local & Private:** Uses `llama-cpp-python` and GGUF models for both embeddings and text generation. No internet connection required for the LLM!
- **📥 Smart Telegram Ingestion:** Supports full historical backfills, date-filtered ingestion (`--start-date`), and live sync of new messages.
- **💬 Streamlit Multi-Chat UI:** A sleek web interface supporting multiple conversation threads, chat history memory, and a "Debug Inspector" to view retrieved contexts and scores.
- **⚡ GPU Acceleration:** Natively supports offloading to GPU for blazing-fast token generation.
- **🐳 Dockerized Deployment:** Easy to run using Docker Compose, spinning up isolated containers for the web app, ingestion worker, and Qdrant vector database.

---

## 🏗️ Architecture

1. **Ingestion Service (Telethon):** Connects to your Telegram account, downloads messages, embeds them using an embedding model, and pushes them to Qdrant.
2. **Vector Database (Qdrant):** Runs locally via Docker, storing message embeddings with binary quantization for fast and efficient similarity search.
3. **LLM Engine (llama.cpp):** Handles user queries, generates search vectors, re-ranks results using MMR (Maximal Marginal Relevance), and streams the final generated text.
4. **Web Client (Streamlit):** Serves the chat interface where users submit queries and read the AI's responses.

---

## 🛠️ Prerequisites

- **Python 3.12+**
- **Docker & Docker Compose** (for Qdrant and containerized setup)
- **uv** (Python package installer)
- **Telegram API Credentials** (API ID & API Hash)

---

## 🚀 Setup & Installation

### 1. Install `uv`

If you haven't already, install `uv` package manager:

- **Linux/macOS:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Windows (PowerShell):** `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

### 2. Clone and Install Dependencies

```bash
# Create a virtual environment
uv venv --python 3.12

# Sync all dependencies from pyproject.toml
uv sync
```

*(Note for Windows users: If `llama-cpp-python` fails to build, install "Visual Studio Build Tools" with C++ Desktop development and run `uv sync` again.)*

### 3. Environment Variables

Copy the example environment file and fill in your details:

```bash
cp .env.example .env
```

**Crucial `.env` fields:**

- `TELEGRAM_API_ID` & `TELEGRAM_API_HASH`: Get these from [my.telegram.org](https://my.telegram.org/apps).
- `TELEGRAM_PHONE`: Your phone number with country code (e.g., `+1234567890`).
- `LLAMA_N_GPU_LAYERS`: Set to `-1` to offload all layers to GPU (if you have one), or `0` for CPU-only.

### 4. Download LLM Models

The app requires GGUF format models in the `./models/` directory. By default, it expects:

1. **Main Generation Model:** e.g., `Qwen2.5-14B-Instruct-Q4_K_M.gguf`
2. **Embedding Model:** e.g., `gte-Qwen2-1.5B-instruct-Q5_K_M.gguf`

**To download automatically:**
Define `MAIN_MODEL_REPO` and `EMBED_MODEL_REPO` in your `.env` and run:

```bash
uv run python download_models.py
```

---

## 💻 Usage

You have two primary ways to run the project: locally (directly on your host machine) or fully containerized via Docker.

### Option A: Fully Dockerized (Recommended)

Run everything (Qdrant, Ingestion, and the Streamlit App) inside Docker containers.

```bash
docker-compose up --build -d
```

*Note: On your very first run, you will need to authenticate with Telegram. The `ingest` container is set up with `tty: true`. You may need to attach to it to input your Telegram 2FA code.*

```bash
docker attach telegram-rag-ingest
```

### Option B: Local Python Execution

Ensure Docker is running (for Qdrant) before executing these commands.

**1. Start the Server & Ingestion:**
Starts Qdrant and begins syncing your Telegram messages.

```bash
uv run python run_server.py --ingest-mode sync
```

**2. Start the Web UI:**
In a separate terminal, launch the Streamlit frontend.

```bash
uv run python run_client.py
```

**Alternatively, run everything at once:**

```bash
uv run python run_all.py
```

*Shell scripts are also available in `./scripts/` (e.g., `./scripts/start_all.sh` or `.\scripts\start_all.ps1`).*

---

## ⚙️ Ingestion Modes & Filtering

When running `run_server.py` or editing your `docker-compose.yml`, you can control how messages are ingested:


| Mode             | Description                                                   |
| ---------------- | ------------------------------------------------------------- |
| `sync`           | Fetches new messages since the last run. Best for daily use.  |
| `full`           | Complete historical backfill of your entire Telegram account. |
| `full_then_sync` | Does a full backfill, then switches to live sync mode.        |
| `none`           | Skips ingestion entirely (just starts Qdrant).                |


**Date Filtering:**
You can tell the ingestion engine to only fetch messages newer than a specific date to save time and storage space.

```bash
uv run python run_server.py --ingest-mode full --start-date "2024-01-01"
```

---

## 🎛️ The Web UI (Streamlit)

Once running, access the web client at **[http://localhost:8501](http://localhost:8501)**.

- **Multi-Chat:** Click `➕ New Chat` in the sidebar to start a fresh context. The system automatically names your chats based on your first query.
- **Debug Inspector:** Open the sidebar expander to see *exactly* what search query the LLM generated, whether it decided to use history, and view the raw retrieved chunks with their relevance scores!

---

## 🧪 Testing

To run the test suite:

```bash
uv run pytest -q
```

