from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from tqdm import tqdm as base_tqdm

from src.config import load_settings


class HFTqdm(base_tqdm):
    def __init__(self, *args, **kwargs):
        # huggingface_hub may pass internal metadata fields unsupported by tqdm.
        kwargs.pop("name", None)
        super().__init__(*args, **kwargs)


def _require(value: str, env_name: str) -> str:
    if value.strip():
        return value.strip()
    raise ValueError(f"Missing required value for {env_name}")


def download_models() -> tuple[Path, Path]:
    load_dotenv()
    settings = load_settings()

    main_repo = _require(os.getenv("MAIN_MODEL_REPO", ""), "MAIN_MODEL_REPO")
    main_filename = _require(
        os.getenv("MAIN_MODEL_FILENAME", settings.main_model_path.name),
        "MAIN_MODEL_FILENAME",
    )
    embed_repo = _require(os.getenv("EMBED_MODEL_REPO", ""), "EMBED_MODEL_REPO")
    embed_filename = _require(
        os.getenv("EMBED_MODEL_FILENAME", settings.embed_model_path.name),
        "EMBED_MODEL_FILENAME",
    )
    hf_token = os.getenv("HF_TOKEN", "").strip() or None

    settings.main_model_path.parent.mkdir(parents=True, exist_ok=True)
    settings.embed_model_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading main model from {main_repo}/{main_filename} ...")
    main_downloaded = Path(
        hf_hub_download(
            repo_id=main_repo,
            filename=main_filename,
            local_dir=str(settings.main_model_path.parent),
            token=hf_token,
            tqdm_class=HFTqdm,
        )
    )
    print(f"Main model download complete: {main_downloaded}")

    print(f"Downloading embedding model from {embed_repo}/{embed_filename} ...")
    embed_downloaded = Path(
        hf_hub_download(
            repo_id=embed_repo,
            filename=embed_filename,
            local_dir=str(settings.embed_model_path.parent),
            token=hf_token,
            tqdm_class=HFTqdm,
        )
    )
    print(f"Embedding model download complete: {embed_downloaded}")
    return main_downloaded, embed_downloaded


def main() -> None:
    try:
        main_model, embed_model = download_models()
    except ValueError as exc:
        print(f"Configuration error: {exc}")
        print("Set these values in .env before running downloader:")
        print("  MAIN_MODEL_REPO=<owner/repo>")
        print("  EMBED_MODEL_REPO=<owner/repo>")
        print("  MAIN_MODEL_FILENAME=<file.gguf>   # optional")
        print("  EMBED_MODEL_FILENAME=<file.gguf>  # optional")
        print("  HF_TOKEN=<token>                  # optional/private repos")
        raise SystemExit(1)

    print(f"Main model: {main_model}")
    print(f"Embed model: {embed_model}")


if __name__ == "__main__":
    main()
