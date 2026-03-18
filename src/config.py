from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class Settings:
    telegram_api_id: int
    telegram_api_hash: str
    telegram_phone: str
    telegram_session_name: str
    qdrant_host: str
    qdrant_port: int
    qdrant_collection: str
    qdrant_vector_size: int
    main_model_path: Path
    embed_model_path: Path
    llama_n_ctx: int
    llama_n_threads: int
    llama_n_batch: int
    llama_n_gpu_layers: int
    ingest_state_path: Path


def _env(key: str, default: str | None = None) -> str:
    value = os.getenv(key, default)
    if value is None:
        raise ValueError(f"Missing required environment variable: {key}")
    return value


def load_settings() -> Settings:
    load_dotenv()

    base_dir = Path.cwd()

    return Settings(
        telegram_api_id=int(_env("TELEGRAM_API_ID", "0")),
        telegram_api_hash=_env("TELEGRAM_API_HASH", ""),
        telegram_phone=_env("TELEGRAM_PHONE", ""),
        telegram_session_name=_env("TELEGRAM_SESSION_NAME", "telegram_rag"),
        qdrant_host=_env("QDRANT_HOST", "localhost"),
        qdrant_port=int(_env("QDRANT_PORT", "6333")),
        qdrant_collection=_env("QDRANT_COLLECTION", "telegram_history"),
        qdrant_vector_size=int(_env("QDRANT_VECTOR_SIZE", "1536")),
        main_model_path=(base_dir / _env("MAIN_MODEL_PATH", "./models/Qwen-2.5-14B-Instruct-Q4_K_M.gguf")).resolve(),
        embed_model_path=(base_dir / _env("EMBED_MODEL_PATH", "./models/gte-Qwen2-1.5B-instruct-Q5_K_M.gguf")).resolve(),
        llama_n_ctx=int(_env("LLAMA_N_CTX", "8192")),
        llama_n_threads=int(_env("LLAMA_N_THREADS", "8")),
        llama_n_batch=int(_env("LLAMA_N_BATCH", "256")),
        llama_n_gpu_layers=int(_env("LLAMA_N_GPU_LAYERS", "0")), # Set to 0 by default for CPU-only systems
        ingest_state_path=(base_dir / _env("INGEST_STATE_PATH", "./data/ingest_state.json")).resolve(),
    )
