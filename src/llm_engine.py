from __future__ import annotations

import json
from dataclasses import dataclass, field

from src.config import Settings


SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions based on the user's Telegram history.\n"
    "Context is provided in format: [Date, Author]: Message.\n"
    "Answer STRICTLY based on context. If the answer is not found, state it.\n"
    "Be concise."
)


ROUTER_PROMPT = (
    "Extract structured search parameters from a user question.\n"
    "Return ONLY valid JSON with keys: search_query (string), filters (object).\n"
    "filters may contain: author (string), chat_name (string), date_range (object with from_ts and to_ts as unix ints).\n"
    "Do not include markdown."
)


@dataclass
class RouterOutput:
    search_query: str
    filters: dict = field(default_factory=dict)


class LlamaEngine:
    def __init__(self, settings: Settings):
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError("llama-cpp-python is not installed") from exc

        self._Llama = Llama
        self.settings = settings
        self._gen_model = self._Llama(
            model_path=str(settings.main_model_path),
            n_ctx=settings.llama_n_ctx,
            n_threads=settings.llama_n_threads,
            n_batch=settings.llama_n_batch,
            verbose=False,
        )
        self._embed_model = self._Llama(
            model_path=str(settings.embed_model_path),
            embedding=True,
            n_ctx=settings.llama_n_ctx,
            n_threads=settings.llama_n_threads,
            n_batch=settings.llama_n_batch,
            verbose=False,
        )

    def embed_text(self, text: str) -> list[float]:
        result = self._embed_model.embed(text)
        return list(result)

    def route_query(self, question: str) -> RouterOutput:
        response = self._gen_model.create_chat_completion(
            messages=[
                {"role": "system", "content": ROUTER_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        content = response["choices"][0]["message"]["content"].strip()
        parsed = self._safe_parse_json(content)
        return RouterOutput(
            search_query=str(parsed.get("search_query", question)).strip() or question,
            filters=parsed.get("filters", {}) if isinstance(parsed.get("filters", {}), dict) else {},
        )

    def generate_answer(self, question: str, expanded_context: str) -> str:
        user_prompt = f"Question:\n{question}\n\nContext:\n{expanded_context}"
        response = self._gen_model.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=512,
        )
        return response["choices"][0]["message"]["content"].strip()

    @staticmethod
    def _safe_parse_json(text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
