from types import SimpleNamespace

from src.llm_engine import RouterOutput
from src.retrieval import RetrievalEngine


class IntegrationStore:
    def __init__(self):
        self.search_called = False
        self.scroll_called = False

    def search(self, vector, query_filter=None, limit=30, with_payload=True):
        self.search_called = True
        return [
            SimpleNamespace(
                id="chunk-1",
                vector=[1.0, 0.0],
                score=0.95,
                payload={
                    "chat_id": 777,
                    "first_message_id": 100,
                    "last_message_id": 100,
                    "timestamp": 1700000000,
                    "author_name": "@nikita",
                    "text": "Deploy at 19:00",
                },
            )
        ]

    def scroll(self, query_filter=None, limit=256, with_payload=True, with_vectors=False):
        self.scroll_called = True
        return [
            SimpleNamespace(
                id="ctx-1",
                payload={
                    "chat_id": 777,
                    "first_message_id": 99,
                    "timestamp": 1699999900,
                    "author_name": "@alice",
                    "text": "Let's deploy today",
                },
            ),
            SimpleNamespace(
                id="ctx-2",
                payload={
                    "chat_id": 777,
                    "first_message_id": 100,
                    "timestamp": 1700000000,
                    "author_name": "@nikita",
                    "text": "Deploy at 19:00",
                },
            ),
        ], None


class IntegrationLLM:
    def route_query(self, question):
        return RouterOutput(search_query=question, filters={})

    def embed_text(self, text):
        return [1.0, 0.0]


def test_retrieval_pipeline_end_to_end_with_fakes():
    store = IntegrationStore()
    llm = IntegrationLLM()
    engine = RetrievalEngine(settings=SimpleNamespace(), store=store, llm=llm)

    result = engine.retrieve("When was deploy?")
    assert store.search_called
    assert store.scroll_called
    assert len(result.top_chunks) == 1
    assert "@nikita" in result.expanded_context
