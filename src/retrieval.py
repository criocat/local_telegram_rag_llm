from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
from qdrant_client.http import models
from sklearn.metrics.pairwise import cosine_similarity

from src.config import Settings, load_settings
from src.llm_engine import LlamaEngine, RouterOutput
from src.qdrant_setup import QdrantStore, create_store


@dataclass
class RetrievedChunk:
    point_id: str
    score: float
    payload: dict
    vector: list[float] = field(default_factory=list)


@dataclass
class RetrievalResult:
    router: RouterOutput
    top_chunks: list[RetrievedChunk]
    expanded_context: str


class RetrievalEngine:
    def __init__(self, settings: Settings, store: QdrantStore, llm: LlamaEngine):
        self.settings = settings
        self.store = store
        self.llm = llm

    def retrieve(self, question: str) -> RetrievalResult:
        router = self.llm.route_query(question)
        vector = self.llm.embed_text(router.search_query)
        query_filter = self._build_filter(router.filters)
        candidates = self.store.search(vector=vector, query_filter=query_filter, limit=30, with_payload=True)
        re_ranked = self._apply_mmr(query_vector=vector, candidates=candidates, lambda_param=0.6, top_k=5)
        expanded = self._expand_context(re_ranked)
        return RetrievalResult(router=router, top_chunks=re_ranked, expanded_context=expanded)

    def _build_filter(self, filters: dict) -> models.Filter | None:
        conditions: list[models.FieldCondition] = []
        author = filters.get("author")
        chat_name = filters.get("chat_name")
        date_range = filters.get("date_range", {})

        if author:
            normalized_author = str(author)
            if not normalized_author.startswith("@"):
                normalized_author = f"@{normalized_author}"
            conditions.append(
                models.FieldCondition(
                    key="author_name",
                    match=models.MatchValue(value=normalized_author),
                )
            )
        if chat_name:
            conditions.append(
                models.FieldCondition(
                    key="chat_name",
                    match=models.MatchText(text=str(chat_name)),
                )
            )
        from_ts = date_range.get("from_ts") if isinstance(date_range, dict) else None
        to_ts = date_range.get("to_ts") if isinstance(date_range, dict) else None
        if from_ts is not None or to_ts is not None:
            conditions.append(
                models.FieldCondition(
                    key="timestamp",
                    range=models.Range(gte=from_ts, lte=to_ts),
                )
            )
        if not conditions:
            return None
        return models.Filter(must=conditions)

    def _apply_mmr(
        self,
        query_vector: list[float],
        candidates: list[Any],
        lambda_param: float = 0.6,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []

        vectors = np.array([np.array(point.vector) for point in candidates], dtype=float)
        query = np.array(query_vector, dtype=float).reshape(1, -1)
        relevance = cosine_similarity(vectors, query).reshape(-1)

        selected_idxs: list[int] = []
        remaining = set(range(len(candidates)))
        while remaining and len(selected_idxs) < top_k:
            if not selected_idxs:
                idx = int(np.argmax(relevance))
                selected_idxs.append(idx)
                remaining.discard(idx)
                continue
            best_idx = None
            best_score = float("-inf")
            for idx in remaining:
                sim_to_query = relevance[idx]
                sim_to_selected = max(
                    cosine_similarity(vectors[idx].reshape(1, -1), vectors[s].reshape(1, -1)).item()
                    for s in selected_idxs
                )
                mmr_score = lambda_param * sim_to_query - (1 - lambda_param) * sim_to_selected
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            assert best_idx is not None
            selected_idxs.append(best_idx)
            remaining.discard(best_idx)

        return [
            RetrievedChunk(
                point_id=str(candidates[idx].id),
                score=float(getattr(candidates[idx], "score", 0.0)),
                payload=dict(candidates[idx].payload or {}),
                vector=list(candidates[idx].vector or []),
            )
            for idx in selected_idxs
        ]

    def _expand_context(self, selected: list[RetrievedChunk]) -> str:
        expanded_points: dict[str, dict] = {}
        for chunk in selected:
            payload = chunk.payload
            chat_id = int(payload.get("chat_id"))
            first_message_id = int(payload.get("first_message_id", 0))
            last_message_id = int(payload.get("last_message_id", 0))
            if chat_id == 0:
                continue
            qfilter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="chat_id",
                        match=models.MatchValue(value=chat_id),
                    ),
                    models.FieldCondition(
                        key="first_message_id",
                        range=models.Range(lte=last_message_id + 2),
                    ),
                    models.FieldCondition(
                        key="last_message_id",
                        range=models.Range(gte=max(0, first_message_id - 2)),
                    ),
                ]
            )
            neighbors, _ = self.store.scroll(query_filter=qfilter, limit=20, with_payload=True, with_vectors=False)
            for point in neighbors:
                expanded_points[str(point.id)] = dict(point.payload or {})

        sorted_payloads = sorted(expanded_points.values(), key=lambda p: (p.get("timestamp", 0), p.get("first_message_id", 0)))
        lines: list[str] = []
        for payload in sorted_payloads:
            ts = int(payload.get("timestamp", 0))
            dt = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            author = payload.get("author_name", "@unknown")
            text = payload.get("text", "")
            lines.append(f"[{dt}, {author}]: {text}")
        return "\n".join(lines)


def build_engine() -> RetrievalEngine:
    settings = load_settings()
    store = create_store(settings)
    llm = LlamaEngine(settings)
    return RetrievalEngine(settings=settings, store=store, llm=llm)
