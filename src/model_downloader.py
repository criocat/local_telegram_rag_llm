from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from src.config import load_settings


def _require(value: str, env_name: str) -> str:
    if value.strip():
        return value.strip()
    raise ValueError(f"Missing required value for {env_name}")


def download_models() -> tuple[Path, Path]:
    load_dotenv()
    settings = load_settings()

    import os

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

    main_downloaded = Path(
        hf_hub_download(
            repo_id=main_repo,
            filename=main_filename,
            local_dir=str(settings.main_model_path.parent),
            local_dir_use_symlinks=False,
            token=hf_token,
        )
    )
    embed_downloaded = Path(
        hf_hub_download(
            repo_id=embed_repo,
            filename=embed_filename,
            local_dir=str(settings.embed_model_path.parent),
            local_dir_use_symlinks=False,
            token=hf_token,
        )
    )
    return main_downloaded, embed_downloaded


def main() -> None:
    main_model, embed_model = download_models()
    print(f"Main model: {main_model}")
    print(f"Embed model: {embed_model}")


if __name__ == "__main__":
    main()
