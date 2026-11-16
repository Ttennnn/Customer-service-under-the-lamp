"""CLI tests for the full reindex job.

The module-level service getters are replaced so these tests never resolve
settings or touch MySQL/Milvus; ``--limit 0`` short-circuits before any
session or embedder is built.
"""


class FakeStore:
    def __init__(self, recreate=None):
        self._recreate = recreate
        self.items = []

    def recreate_collection(self):
        if self._recreate is not None:
            self._recreate.append("recreate")

    def upsert(self, items):
        self.items.extend(items)


def test_recreate_requires_yes(monkeypatch, capsys):
    from app.jobs import reindex_hybrid

    assert reindex_hybrid.main(["--recreate"]) == 2
    assert "--yes" in capsys.readouterr().out


def test_recreate_calls_store_recreate(monkeypatch):
    from app.jobs import reindex_hybrid

    called = []
    monkeypatch.setattr(
        reindex_hybrid, "get_vector_store", lambda: FakeStore(recreate=called)
    )
    assert reindex_hybrid.main(["--recreate", "--yes", "--limit", "0"]) == 0
    assert called == ["recreate"]
