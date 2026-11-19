"""Offline tests for the ch09 intent token cost report.

Every Langfuse call is faked: importing/using the job must never build a real
SDK client or touch the network.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from langfuse.api.core.api_error import ApiError

from app.jobs import cost_by_intent as job


# --- fakes -----------------------------------------------------------------


class FakeResponse:
    def __init__(self, data, meta=None):
        self.data = data
        self.meta = meta


class FakeMetrics:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls: list[str] = []

    def metrics(self, *, query):
        self.calls.append(query)
        if self.error is not None:
            raise self.error
        return self.response


class FakeTrace:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    def list(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeObservations:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    def get_many(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeApi:
    def __init__(
        self,
        *,
        metrics_v_2=None,
        metrics=None,
        trace=None,
        observations=None,
    ):
        self.metrics_v_2 = metrics_v_2
        self.metrics = metrics
        self.trace = trace
        self.observations = observations


class FakeClient:
    def __init__(self, **api_parts):
        self.api = FakeApi(**api_parts)


def _settings(report_dir="data/ch09/reports"):
    return SimpleNamespace(
        langfuse_public_key="pk-lf-test",
        langfuse_secret_key="sk-lf-test",
        langfuse_base_url="http://localhost:3000",
        langfuse_environment="development",
        ch09_report_dir=str(report_dir),
    )


def _metric_row(tags, trace_id, tokens):
    return {
        "tags": tags,
        "traceId": trace_id,
        "sum_totalTokens": tokens,
        "sum_inputTokens": tokens - 10,
        "sum_outputTokens": 10,
    }


# --- pure aggregation ------------------------------------------------------


def test_aggregate_by_intent_sums_tokens_and_computes_share():
    rows = [
        {"tags": ["intent:商品咨询"], "traceId": "t1", "totalTokens": 100},
        {"tags": ["intent:商品咨询"], "traceId": "t1", "totalTokens": 50},
        {"tags": ["intent:退款退货"], "traceId": "t2", "totalTokens": 50},
    ]
    out = job.aggregate_by_intent(rows)
    by = {r["intent"]: r for r in out}
    assert by["商品咨询"]["total_tokens"] == 150
    assert by["商品咨询"]["requests"] == 1  # unique traceId
    assert by["商品咨询"]["share"] == 0.75
    assert by["退款退货"]["avg_tokens"] == 50


def test_build_report_has_rows_and_window():
    report = job.build_report(
        [
            {
                "intent": "闲聊",
                "requests": 1,
                "total_tokens": 10,
                "avg_tokens": 10,
                "share": 1.0,
            }
        ],
        7,
        generated_at=datetime(2026, 11, 19, 12, 0, tzinfo=timezone.utc),
    )
    assert report["window_days"] == 7
    assert report["rows"][0]["intent"] == "闲聊"
    assert report["generated_at"].startswith("2026-11-19T12:00")


def test_aggregate_by_intent_skips_untagged_rows():
    rows = [
        {"tags": ["other"], "traceId": "t1", "totalTokens": 99},
        {"tags": ["intent:闲聊"], "traceId": "t2", "totalTokens": 1},
    ]
    out = job.aggregate_by_intent(rows)
    assert [row["intent"] for row in out] == ["闲聊"]
    assert out[0]["total_tokens"] == 1


def test_aggregate_by_intent_reads_usage_details_and_counts_missing_trace():
    rows = [
        {"tags": "intent:物流", "usageDetails": {"input": 10, "output": 5}},
        {"tags": "intent:物流", "traceId": "t2", "usage": {"total": 7}},
    ]
    out = job.aggregate_by_intent(rows)
    assert out[0]["total_tokens"] == 22
    # No traceId on the first row: it still counts as one request.
    assert out[0]["requests"] == 2
    assert out[0]["avg_tokens"] == 11


def test_aggregate_by_intent_reads_comma_joined_tags():
    rows = [{"tags": "other,intent:物流", "traceId": "t1", "totalTokens": 5}]
    out = job.aggregate_by_intent(rows)
    assert [row["intent"] for row in out] == ["物流"]


def test_render_table_has_chinese_columns_and_percentage():
    rows = [
        {
            "intent": "商品咨询",
            "requests": 1,
            "total_tokens": 150,
            "avg_tokens": 150,
            "share": 1.0,
        }
    ]
    text = job.render_table(rows, 7, source="metrics_v1")
    assert "意图" in text and "请求数" in text and "总 tokens" in text
    assert "100.00%" in text
    assert "metrics_v1" in text


# --- source selection ------------------------------------------------------


def test_fetch_observations_prefers_metrics_v2():
    metrics_v2 = FakeMetrics(FakeResponse([_metric_row(["intent:商品咨询"], "t1", 100)]))
    client = FakeClient(metrics_v_2=metrics_v2)

    rows, source = job.fetch_observations(client, _settings(), 7)

    assert source == "metrics_v2"
    assert rows == [
        {
            "tags": ["intent:商品咨询"],
            "traceId": "t1",
            "sum_totalTokens": 100,
            "sum_inputTokens": 90,
            "sum_outputTokens": 10,
        }
    ]
    query = json.loads(metrics_v2.calls[0])
    assert query["view"] == "observations"
    assert {"field": "tags"} in query["dimensions"]
    assert {"field": "traceId"} in query["dimensions"]
    measures = {(m["measure"], m["aggregation"]) for m in query["metrics"]}
    assert ("totalTokens", "sum") in measures
    assert query["fromTimestamp"].endswith("Z")
    assert query["toTimestamp"].endswith("Z")
    # v2 needs a top-N orderBy to accept the traceId dimension.
    assert query["orderBy"] == [
        {"field": "sum_totalTokens", "direction": "desc"}
    ]


def test_fetch_observations_falls_back_to_legacy_metrics_on_400():
    metrics_v2 = FakeMetrics(error=ApiError(status_code=400, body="high cardinality"))
    metrics_v1 = FakeMetrics(FakeResponse([_metric_row(["intent:退款退货"], "t2", 50)]))
    client = FakeClient(metrics_v_2=metrics_v2, metrics=metrics_v1)

    rows, source = job.fetch_observations(client, _settings(), 3)

    assert source == "metrics_v1"
    assert len(metrics_v2.calls) == 1
    assert len(metrics_v1.calls) == 1
    assert rows[0]["traceId"] == "t2"


def test_fetch_observations_rejects_metric_rows_without_trace_id():
    metrics_v2 = FakeMetrics(
        FakeResponse([{"tags": ["intent:商品咨询"], "sum_totalTokens": 100}])
    )
    metrics_v1 = FakeMetrics(
        FakeResponse([_metric_row(["intent:商品咨询"], "t1", 100)])
    )
    client = FakeClient(metrics_v_2=metrics_v2, metrics=metrics_v1)

    rows, source = job.fetch_observations(client, _settings(), 7)

    assert source == "metrics_v1"
    assert rows[0]["traceId"] == "t1"


def test_fetch_observations_falls_back_to_observation_pull():
    metrics_v2 = FakeMetrics(error=ApiError(status_code=404, body="missing"))
    metrics_v1 = FakeMetrics(error=ApiError(status_code=400, body="bad query"))
    trace = FakeTrace(
        FakeResponse(
            [
                {"id": "t1", "tags": ["intent:退款退货"]},
                {"id": "t2", "tags": ["other"]},
            ]
        )
    )
    observations = FakeObservations(
        FakeResponse(
            [
                {"traceId": "t1", "usageDetails": {"input": 20, "output": 30}},
                {"traceId": "t2", "usageDetails": {"total": 999}},
            ]
        )
    )
    client = FakeClient(
        metrics_v_2=metrics_v2,
        metrics=metrics_v1,
        trace=trace,
        observations=observations,
    )

    rows, source = job.fetch_observations(client, _settings(), 7)

    assert source == "observations"
    assert rows == [
        {
            "traceId": "t1",
            "usageDetails": {"input": 20, "output": 30},
            "tags": ["intent:退款退货"],
        }
    ]
    assert "from_timestamp" in trace.calls[0]
    assert "from_start_time" in observations.calls[0]


def test_fetch_observations_raises_when_all_sources_fail():
    client = FakeClient(
        metrics_v_2=FakeMetrics(error=ApiError(status_code=404, body="missing")),
        metrics=FakeMetrics(error=ApiError(status_code=400, body="bad query")),
        trace=FakeTrace(error=ApiError(status_code=500, body="down")),
    )

    with pytest.raises(job.CostReportError) as excinfo:
        job.fetch_observations(client, _settings(), 7)

    message = str(excinfo.value)
    assert "metrics_v2" in message
    assert "metrics_v1" in message
    assert "observations" in message


def test_fetch_observations_returns_empty_rows_on_successful_empty_query():
    metrics_v2 = FakeMetrics(FakeResponse([]))
    client = FakeClient(metrics_v_2=metrics_v2)

    rows, source = job.fetch_observations(client, _settings(), 7)

    assert rows == []
    assert source == "metrics_v2"


# --- main ------------------------------------------------------------------


def test_main_writes_json_and_txt_offline(tmp_path, monkeypatch):
    client = FakeClient(
        metrics_v_2=FakeMetrics(
            FakeResponse(
                [
                    _metric_row(["intent:商品咨询"], "t1", 100),
                    _metric_row(["intent:商品咨询"], "t1", 20),
                    _metric_row(["intent:退款退货"], "t2", 40),
                ]
            )
        )
    )
    monkeypatch.setattr(job, "get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(job, "get_langfuse_client", lambda settings: client)

    assert job.main(["--days", "3"]) == 0

    json_path = tmp_path / "cost_by_intent.json"
    txt_path = tmp_path / "cost_by_intent.txt"
    assert json_path.exists() and txt_path.exists()

    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["window_days"] == 3
    assert report["rows"][0]["intent"] == "商品咨询"
    assert report["rows"][0]["total_tokens"] == 120
    assert report["rows"][0]["requests"] == 1
    assert report["rows"][0]["share"] == 0.75
    assert report["rows"][1]["avg_tokens"] == 40

    text = txt_path.read_text(encoding="utf-8")
    assert "商品咨询" in text and "75.00%" in text
    assert "metrics_v2" in text


def test_main_json_flag_prints_report(tmp_path, monkeypatch, capsys):
    client = FakeClient(metrics_v_2=FakeMetrics(FakeResponse([])))
    monkeypatch.setattr(job, "get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(job, "get_langfuse_client", lambda settings: client)

    assert job.main(["--days", "1", "--json"]) == 0

    printed = json.loads(capsys.readouterr().out)
    assert printed["window_days"] == 1
    assert printed["rows"] == []
    assert (tmp_path / "cost_by_intent.json").exists()


def test_main_refuses_without_credentials_and_does_not_create_client(
    tmp_path, monkeypatch, capsys
):
    settings = SimpleNamespace(
        langfuse_public_key="",
        langfuse_secret_key="",
        langfuse_base_url="",
        ch09_report_dir=str(tmp_path),
    )
    created = []

    def fake_client(settings):
        created.append(settings)
        return FakeClient()

    monkeypatch.setattr(job, "get_settings", lambda: settings)
    monkeypatch.setattr(job, "get_langfuse_client", fake_client)

    assert job.main([]) == 1
    assert created == []
    assert "credentials" in capsys.readouterr().out.lower()


def test_main_reports_fetch_failure(tmp_path, monkeypatch, capsys):
    client = FakeClient(
        metrics_v_2=FakeMetrics(error=ApiError(status_code=500, body="down")),
        metrics=FakeMetrics(error=ApiError(status_code=500, body="down")),
        trace=FakeTrace(error=ApiError(status_code=500, body="down")),
    )
    monkeypatch.setattr(job, "get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(job, "get_langfuse_client", lambda settings: client)

    assert job.main([]) == 2
    assert "failed to fetch" in capsys.readouterr().out
    assert not (tmp_path / "cost_by_intent.json").exists()
