from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime

from src.config import Settings
from llama_cpp import Llama

SYSTEM_PROMPT = (
    "You are a helpful AI assistant.\n"
    "Sometimes, you will be provided with context from the user's Telegram history in the format: [Date, Author]: Message.\n"
    "If the user's question is about their history, friends, or a specific event, you MUST base your answer on the provided context.\n"
    "If the user asks a general question or requests a creative task (like writing a presentation, poem, or code), fulfill their request normally using your general knowledge. Ignore the lack of context in these cases.\n"
    "Be helpful, concise, and conversational."
)


ROUTER_PROMPT = (
    "Extract structured search parameters from a user question.\n"
    "Return ONLY valid JSON with keys: requires_history (boolean), search_query (string), filters (object).\n"
    "Set requires_history to false ONLY if the user is asking a general knowledge question or a creative task (like writing code, a poem, or an essay) that doesn't need personal messages.\n"
    "filters may contain: author (string), chat_name (string), date_range (object with from_ts and to_ts as unix ints).\n"
    "Do not include markdown."
)


@dataclass
class RouterOutput:
    search_query: str
    requires_history: bool = True
    filters: dict = field(default_factory=dict)


class LlamaEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._gen_model = Llama(
            model_path=str(settings.main_model_path),
            n_ctx=settings.llama_n_ctx,
            n_threads=settings.llama_n_threads,
            n_batch=settings.llama_n_batch,
            n_gpu_layers=settings.llama_n_gpu_layers,
            verbose=False,
        )
        self._embed_model = Llama(
            model_path=str(settings.embed_model_path),
            embedding=True,
            n_ctx=settings.llama_n_ctx,
            n_threads=settings.llama_n_threads,
            n_batch=settings.llama_n_batch,
            n_gpu_layers=settings.llama_n_gpu_layers,
            verbose=False,
        )

    def embed_text(self, text: str) -> list[float]:
        """Generate vector embeddings for the given text using the embedding model."""
        result = self._embed_model.embed(text)
        if result and isinstance(result[0], list):
            return result[0]
        return list(result)

    def route_query(self, question: str, chat_history: list[dict] | None = None) -> RouterOutput:
        """Extract structured search parameters and filters from a user question, considering chat history."""
        now = datetime.now()
        current_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        current_unix = int(now.timestamp())
        system_content = f"{ROUTER_PROMPT}\n\nIMPORTANT: The current date and time is {current_time_str} (UNIX timestamp: {current_unix}). Use this to calculate accurate UNIX timestamps for date_range."
        
        messages = [{"role": "system", "content": system_content}]
        
        if chat_history:
            for msg in chat_history[-4:]:
                messages.append({"role": msg["role"], "content": msg["content"]})
                
        messages.append({"role": "user", "content": question})
        
        response = self._gen_model.create_chat_completion(
            messages=messages,
            temperature=0.0,
            max_tokens=256,
        )
        content = response["choices"][0]["message"]["content"].strip()
        parsed = self._safe_parse_json(content)
        return RouterOutput(
            search_query=str(parsed.get("search_query", question)).strip() or question,
            requires_history=bool(parsed.get("requires_history", True)),
            filters=parsed.get("filters", {}) if isinstance(parsed.get("filters", {}), dict) else {},
        )

    def generate_answer_stream(self, question: str, expanded_context: str, chat_history: list[dict] | None = None) -> Any:
        """Generate a contextual answer to the user's question, yielding text chunks as they are generated."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        if chat_history:
            for msg in chat_history:
                messages.append({"role": msg["role"], "content": msg["content"]})
                
        user_prompt = f"Question:\n{question}\n\nContext:\n{expanded_context}"
        messages.append({"role": "user", "content": user_prompt})
        
        response = self._gen_model.create_chat_completion(
            messages=messages,
            temperature=0.1,
            max_tokens=512,
            stream=True,
        )
        for chunk in response:
            delta = chunk["choices"][0].get("delta", {})
            if "content" in delta:
                yield delta["content"]

    def generate_answer(self, question: str, expanded_context: str, chat_history: list[dict] | None = None) -> str:
        """Generate a contextual answer to the user's question based on retrieved Telegram history and chat history."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        if chat_history:
            for msg in chat_history:
                messages.append({"role": msg["role"], "content": msg["content"]})
                
        user_prompt = f"Question:\n{question}\n\nContext:\n{expanded_context}"
        messages.append({"role": "user", "content": user_prompt})
        
        response = self._gen_model.create_chat_completion(
            messages=messages,
            temperature=0.1,
            max_tokens=512,
        )
        return response["choices"][0]["message"]["content"].strip()

    @staticmethod
    def _safe_parse_json(text: str) -> dict:
        """Safely parse a JSON string, stripping markdown formatting if necessary."""
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
