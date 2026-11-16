"""Focused tests for the ch03 legacy retrieval evaluation job."""

import json
from types import SimpleNamespace

from app.jobs import eval_retrieval


def _write_cases(path, cases):
    path.write_text(
        json.dumps({"version": 1, "cases": cases}, ensure_ascii=False),
        encoding="utf-8",
    )


def test_main_passes_explicit_threshold_to_retrieve(monkeypatch, tmp_path):
    cases_path = tmp_path / "cases.json"
    _write_cases(
        cases_path,
        [
            {
                "id": "N01",
                "query": "老板是谁",
                "expect_hit": False,
            }
        ],
    )
    calls = []

    def fake_retrieve(query, *, threshold=None):
        calls.append((query, threshold))
        return []

    monkeypatch.setattr(eval_retrieval, "retrieve", fake_retrieve)
    monkeypatch.setattr(
        eval_retrieval,
        "get_settings",
        lambda: SimpleNamespace(faq_top_k=3, faq_score_threshold=0.5),
    )

    code = eval_retrieval.main(["--cases", str(cases_path)])

    assert code == 0
    assert calls == [("老板是谁", 0.5)]


def test_banner_states_threshold_and_top_k(monkeypatch, tmp_path, capsys):
    cases_path = tmp_path / "cases.json"
    _write_cases(cases_path, [{"query": "q", "expect_hit": False}])

    monkeypatch.setattr(eval_retrieval, "retrieve", lambda query, **kwargs: [])
    monkeypatch.setattr(
        eval_retrieval,
        "get_settings",
        lambda: SimpleNamespace(faq_top_k=7, faq_score_threshold=0.42),
    )

    assert eval_retrieval.main(["--cases", str(cases_path)]) == 0

    out = capsys.readouterr().out
    assert "dense_top_k=7" in out
    assert "score_threshold=0.42" in out
