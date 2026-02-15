from types import SimpleNamespace

from src.llm_engine import RouterOutput
from src.retrieval import RetrievalEngine


class FakeStore:
    def __init__(self, candidates, neighbors):
        self._candidates = candidates
        self._neighbors = neighbors

    def search(self, vector, query_filter=None, limit=30, with_payload=True):
        return self._candidates[:limit]

    def scroll(self, query_filter=None, limit=256, with_payload=True, with_vectors=False):
        return self._neighbors[:limit], None


class FakeLLM:
    def route_query(self, question):
        return RouterOutput(
            search_query="devops messages",
            filters={"author": "nikita", "chat_name": "DevOps", "date_range": {"from_ts": 100, "to_ts": 1000}},
        )

    def embed_text(self, text):
        return [1.0, 0.0]


def test_build_filter_contains_all_conditions():
    engine = RetrievalEngine(settings=SimpleNamespace(), store=FakeStore([], []), llm=FakeLLM())
    qfilter = engine._build_filter({"author": "nikita", "chat_name": "DevOps", "date_range": {"from_ts": 1, "to_ts": 2}})
    assert qfilter is not None
    assert len(qfilter.must) == 3


def test_retrieve_runs_mmr_and_expansion():
    candidates = [
        SimpleNamespace(
            id="1",
            vector=[1.0, 0.0],
            score=0.9,
            payload={"chat_id": 1, "first_message_id": 10, "last_message_id": 10, "timestamp": 500, "author_name": "@a", "text": "x"},
        ),
        SimpleNamespace(
            id="2",
            vector=[0.0, 1.0],
            score=0.8,
            payload={"chat_id": 1, "first_message_id": 11, "last_message_id": 11, "timestamp": 600, "author_name": "@b", "text": "y"},
        ),
    ]
    neighbors = [
        SimpleNamespace(
            id="n1",
            payload={"chat_id": 1, "first_message_id": 9, "timestamp": 400, "author_name": "@a", "text": "ctx1"},
        ),
        SimpleNamespace(
            id="n2",
            payload={"chat_id": 1, "first_message_id": 12, "timestamp": 700, "author_name": "@b", "text": "ctx2"},
        ),
    ]
    engine = RetrievalEngine(settings=SimpleNamespace(), store=FakeStore(candidates, neighbors), llm=FakeLLM())
    result = engine.retrieve("What happened?")
    assert result.router.search_query == "devops messages"
    assert len(result.top_chunks) >= 1
    assert "ctx1" in result.expanded_context
