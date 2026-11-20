"""按意图统计 Langfuse token 成本（ch09）。

数据源按自部署 Langfuse v3 的实际可用性依次降级（已用 Context7 + 安装包
``langfuse==3.15.0`` 核对方法名与响应形状）：

1. **Metrics API v2**：``client.api.metrics_v_2.metrics(query=json)``，
   GET ``/api/public/v2/metrics``，返回 ``MetricsV2Response.data``
   （``list[dict]``）。
2. **Metrics API v1（legacy）**：``client.api.metrics.metrics(query=json)``，
   GET ``/api/public/metrics``，返回 ``MetricsResponse.data``。
   v2 文档明确禁止把 ``traceId`` 作为分组维度（高基数维度会返回 400），
   所以 v2 实际通常会 400，再走这里按 ``tags + traceId`` 分组；同 trace
   的多条观测由脚本按 ``traceId`` 去重计请求数。
3. **本地拉取**：``client.api.trace.list(...)`` 取 trace 的 ``tags`` 建立
   ``traceId -> intent`` 映射，再用
   ``client.api.observations.get_many(...)`` 拉观测，按 ``traceId`` 关联。

注意：``langfuse==3.15.0`` **没有** ``api.legacy.metrics_v1`` 资源；legacy
metrics 的实际入口就是 ``api.metrics``。查询里的 measure 使用
``totalTokens`` / ``inputTokens`` / ``outputTokens``，响应聚合字段形如
``sum_totalTokens``。

所有统计（请求数、总量、均值、占比）都在本脚本算好并写入
``<ch09_report_dir>/cost_by_intent.{txt,json}``；页面只读 JSON。
导入本模块不会创建 Langfuse 客户端，也不会访问网络。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from app.config import get_settings
from app.observability import get_langfuse_client, tracing_enabled

logger = logging.getLogger(__name__)

INTENT_TAG_PREFIX = "intent:"
DEFAULT_WINDOW_DAYS = 7
REPORT_TXT = "cost_by_intent.txt"
REPORT_JSON = "cost_by_intent.json"

#: Metrics API 单次查询最多返回 1000 行。
_METRICS_ROW_LIMIT = 1000
#: 本地拉取分页大小与安全上限（避免坏 meta 造成死循环）。
_PAGE_LIMIT = 100
_MAX_PAGES = 100

_SOURCE_LABELS = {
    "metrics_v2": "Langfuse Metrics v2 (client.api.metrics_v_2.metrics)",
    "metrics_v1": "Langfuse Metrics v1 legacy (client.api.metrics.metrics)",
    "observations": (
        "observations pull (client.api.trace.list + "
        "client.api.observations.get_many)"
    ),
}


class CostReportError(RuntimeError):
    """Raised when every Langfuse data source fails."""


# --- small normalisers -----------------------------------------------------


def _as_mapping(item: Any) -> dict:
    """Return *item* as a plain dict (pydantic models, namespaces, dicts)."""
    if isinstance(item, dict):
        return item
    for kwargs in ({"by_alias": True, "exclude_none": True}, {"by_alias": True}, {}):
        method = getattr(item, "dict", None)
        if callable(method):
            try:
                value = method(**kwargs)
            except TypeError:
                continue
            if isinstance(value, dict):
                return value
    if hasattr(item, "__dict__"):
        return vars(item)
    return {}


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        # Some metrics responses join tags with commas; split only when an
        # intent tag is visible so a tag containing a comma is left alone.
        if "," in text:
            parts = [part.strip() for part in text.split(",") if part.strip()]
            if any(part.startswith(INTENT_TAG_PREFIX) for part in parts):
                return parts
        return [text]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _intent_from_tags(value: Any) -> str | None:
    for tag in _tags(value):
        text = tag.strip()
        if text.startswith(INTENT_TAG_PREFIX):
            intent = text[len(INTENT_TAG_PREFIX) :].strip()
            if intent:
                return intent
    return None


def _row_trace_id(row: dict) -> str | None:
    for key in ("traceId", "trace_id", "id"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _row_tokens(row: dict) -> int:
    """Total tokens from a metrics row or a raw observation row."""
    for key in (
        "totalTokens",
        "total_tokens",
        "sum_totalTokens",
        "sum_total_tokens",
    ):
        if row.get(key) is not None:
            return _as_int(row[key])

    for key in ("usageDetails", "usage_details", "usage"):
        usage = row.get(key)
        if isinstance(usage, dict):
            if usage.get("total") is not None:
                return _as_int(usage["total"])
            if usage:
                return sum(_as_int(value) for value in usage.values())

    input_tokens = row.get("inputTokens", row.get("input_tokens"))
    output_tokens = row.get("outputTokens", row.get("output_tokens"))
    if input_tokens is not None or output_tokens is not None:
        return _as_int(input_tokens) + _as_int(output_tokens)
    return 0


def _response_data(response: Any) -> list:
    data = getattr(response, "data", None)
    if data is None and isinstance(response, dict):
        data = response.get("data")
    return list(data or [])


# --- pure aggregation ------------------------------------------------------


def aggregate_by_intent(observations: Iterable[Any]) -> list[dict]:
    """Aggregate observation/metric rows by ``intent:<name>`` tag.

    ``requests`` is the number of unique trace ids per intent (rows without a
    trace id each count as one request). ``avg_tokens`` and ``share`` are
    computed here so the report/JSON is ready for the page to render.
    """
    buckets: dict[str, dict] = {}

    for row in observations or []:
        mapping = _as_mapping(row)
        intent = _intent_from_tags(mapping.get("tags"))
        if not intent:
            continue
        bucket = buckets.setdefault(
            intent,
            {"intent": intent, "total_tokens": 0, "trace_ids": set(), "anonymous": 0},
        )
        bucket["total_tokens"] += _row_tokens(mapping)
        trace_id = _row_trace_id(mapping)
        if trace_id is None:
            bucket["anonymous"] += 1
        else:
            bucket["trace_ids"].add(trace_id)

    grand_total = sum(bucket["total_tokens"] for bucket in buckets.values())
    rows: list[dict] = []
    for bucket in buckets.values():
        requests = len(bucket["trace_ids"]) + bucket["anonymous"]
        total_tokens = bucket["total_tokens"]
        rows.append(
            {
                "intent": bucket["intent"],
                "requests": requests,
                "total_tokens": total_tokens,
                "avg_tokens": round(total_tokens / requests) if requests else 0,
                "share": round(total_tokens / grand_total, 4)
                if grand_total
                else 0.0,
            }
        )
    rows.sort(key=lambda item: (-item["total_tokens"], item["intent"]))
    return rows


def build_report(
    rows: list[dict],
    window_days: int,
    *,
    generated_at: datetime | None = None,
) -> dict:
    """Build the JSON document consumed by ``/observability`` (spec §5)."""
    timestamp = generated_at or datetime.now(timezone.utc)
    return {
        "generated_at": timestamp.isoformat(timespec="seconds"),
        "window_days": int(window_days),
        "rows": [dict(row) for row in rows],
    }


# --- Langfuse source selection --------------------------------------------


def _window(days: int) -> tuple[datetime, datetime]:
    end = datetime.now(timezone.utc)
    return end - timedelta(days=days), end


def _timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _metrics_query(days: int) -> str:
    """Build a Metrics API (v2 and v1 share the query shape) JSON string.

    ``traceId`` is included so each row can be deduplicated into unique
    requests. v2 rejects that high-cardinality dimension with HTTP 400, which
    is the documented reason the legacy endpoint is the usual success path.
    """
    start, end = _window(days)
    query = {
        "view": "observations",
        "dimensions": [{"field": "tags"}, {"field": "traceId"}],
        "metrics": [
            {"measure": "totalTokens", "aggregation": "sum"},
            {"measure": "inputTokens", "aggregation": "sum"},
            {"measure": "outputTokens", "aggregation": "sum"},
        ],
        "filters": [],
        "fromTimestamp": _timestamp(start),
        "toTimestamp": _timestamp(end),
        # v2 only allows high-cardinality traceId as a top-N dimension: it
        # needs row_limit, a descending orderBy on a metric, and no
        # timeDimension.
        "orderBy": [{"field": "sum_totalTokens", "direction": "desc"}],
        "config": {"row_limit": _METRICS_ROW_LIMIT},
    }
    return json.dumps(query, ensure_ascii=False)


def _require_trace_ids(rows: list) -> list:
    """Reject metric rows without ``traceId``: requests could not be counted."""
    for row in rows:
        mapping = _as_mapping(row)
        if not (mapping.get("traceId") or mapping.get("trace_id")):
            raise CostReportError(
                "metrics response has no traceId; cannot count unique requests"
            )
    return rows


def _fetch_metrics_v2(client: Any, days: int) -> list:
    response = client.api.metrics_v_2.metrics(query=_metrics_query(days))
    return _require_trace_ids(_response_data(response))


def _fetch_metrics_v1(client: Any, days: int) -> list:
    response = client.api.metrics.metrics(query=_metrics_query(days))
    return _require_trace_ids(_response_data(response))


def _fetch_trace_intents(client: Any, days: int) -> dict[str, str]:
    """Map ``traceId -> intent`` from traces carrying ``intent:`` tags."""
    start, end = _window(days)
    intents: dict[str, str] = {}
    for page in range(1, _MAX_PAGES + 1):
        response = client.api.trace.list(
            page=page,
            limit=_PAGE_LIMIT,
            from_timestamp=start,
            to_timestamp=end,
        )
        data = _response_data(response)
        for item in data:
            mapping = _as_mapping(item)
            intent = _intent_from_tags(mapping.get("tags"))
            trace_id = mapping.get("id")
            if intent and trace_id:
                intents[str(trace_id)] = intent
        if len(data) < _PAGE_LIMIT:
            break
    return intents


def _fetch_observations_local(client: Any, days: int) -> list:
    """Join raw observations with trace tags into aggregation-ready rows."""
    trace_intents = _fetch_trace_intents(client, days)
    if not trace_intents:
        return []

    start, end = _window(days)
    rows: list[dict] = []
    for page in range(1, _MAX_PAGES + 1):
        response = client.api.observations.get_many(
            page=page,
            limit=_PAGE_LIMIT,
            from_start_time=start,
            to_start_time=end,
        )
        data = _response_data(response)
        for item in data:
            mapping = dict(_as_mapping(item))
            trace_id = _row_trace_id(mapping)
            intent = trace_intents.get(trace_id or "")
            if not intent:
                continue
            mapping["traceId"] = trace_id
            mapping["tags"] = [f"{INTENT_TAG_PREFIX}{intent}"]
            rows.append(mapping)
        if len(data) < _PAGE_LIMIT:
            break
    return rows


def fetch_observations(
    client: Any, settings: Any, days: int
) -> tuple[list, str]:
    """Fetch rows for the report, trying v2 Metrics, v1 Metrics, then a pull.

    Returns ``(rows, source)`` where *source* is one of ``metrics_v2``,
    ``metrics_v1`` or ``observations``. A source that succeeds but yields no
    ``intent:``-tagged rows is not authoritative: the next source is tried, so
    a partially-populated higher-priority endpoint cannot hide data the lower
    one has. If every source succeeds but none has intent tags, the last
    successful (empty) result is returned. Raises :class:`CostReportError`
    only when every source actually fails, with each failure summarised.

    *settings* is accepted for interface symmetry with the other ch09 jobs;
    the API key already scopes queries to one project, so no additional
    environment filter is applied.
    """
    failures: list[str] = []
    empty_result: tuple[list, str] | None = None
    sources = (
        ("metrics_v2", _fetch_metrics_v2),
        ("metrics_v1", _fetch_metrics_v1),
        ("observations", _fetch_observations_local),
    )
    for name, fetcher in sources:
        try:
            rows = fetcher(client, days)
        except Exception as exc:  # noqa: BLE001 - try the next source
            # A v2 400 for the traceId dimension is expected on most servers,
            # so this is informational rather than a warning with a traceback.
            logger.info("cost_by_intent: %s source failed: %s", name, exc)
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        if aggregate_by_intent(rows):
            return rows, name
        logger.info(
            "cost_by_intent: %s returned no intent-tagged rows; trying next "
            "data source",
            name,
        )
        empty_result = (rows, name)

    if empty_result is not None:
        return empty_result

    raise CostReportError(
        "all Langfuse data sources failed (" + "; ".join(failures) + ")"
    )


def _remove_reports(report_dir: Path) -> None:
    """Delete stale report files so the page cannot show old numbers."""
    for name in (REPORT_TXT, REPORT_JSON):
        try:
            (report_dir / name).unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:  # noqa: BLE001 - best effort on failure
            logger.warning("cost_by_intent: cannot remove %s: %s", name, exc)


# --- rendering -------------------------------------------------------------


def _display_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text)


def _pad(text: str, width: int, *, right: bool = False) -> str:
    padding = " " * max(0, width - _display_width(text))
    return padding + text if right else text + padding


def _source_label(source: str) -> str:
    """Human-readable source with the stable key kept for grep-ability."""
    if not source:
        return "unknown"
    label = _SOURCE_LABELS.get(source)
    return f"{source} - {label}" if label else source


def render_table(
    rows: list[dict],
    window_days: int,
    *,
    source: str = "",
    generated_at: datetime | None = None,
) -> str:
    """Render the txt report (Chinese table + data source note)."""
    timestamp = generated_at or datetime.now(timezone.utc)
    headers = ["意图", "请求数", "总 tokens", "平均 tokens", "占比"]
    body = [
        [
            str(row.get("intent", "")),
            str(row.get("requests", 0)),
            str(row.get("total_tokens", 0)),
            str(row.get("avg_tokens", 0)),
            f"{float(row.get('share', 0.0)):.2%}",
        ]
        for row in rows
    ]

    widths = [_display_width(header) for header in headers]
    for line in body:
        for index, cell in enumerate(line):
            widths[index] = max(widths[index], _display_width(cell))

    def format_line(cells: list[str]) -> str:
        parts = [
            _pad(cell, widths[index], right=index > 0)
            for index, cell in enumerate(cells)
        ]
        return "  ".join(parts).rstrip()

    lines = [
        f"# 意图 token 成本报告（最近 {window_days} 天）",
        "",
        f"生成时间: {timestamp.isoformat(timespec='seconds')}",
        f"数据源: {_source_label(source)}",
        f"窗口: 最近 {window_days} 天",
        "",
        format_line(headers),
        format_line(["-" * width for width in widths]),
    ]
    if body:
        lines.extend(format_line(line) for line in body)
    else:
        lines.append("（暂无数据：先聊几句再重跑）")
    lines.append("")
    return "\n".join(lines)


def _report_dir(settings: Any) -> Path:
    return Path(getattr(settings, "ch09_report_dir", "data/ch09/reports"))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: fetch Langfuse data and write the cost report."""
    parser = argparse.ArgumentParser(
        description="按 intent tag 统计 Langfuse token 成本"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_WINDOW_DAYS,
        help=f"统计最近多少天（默认 {DEFAULT_WINDOW_DAYS}）",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="把 JSON 报告打印到 stdout（仍会写文件）",
    )
    args = parser.parse_args(argv)
    if args.days <= 0:
        parser.error("--days must be a positive integer")

    settings = get_settings()
    report_dir = _report_dir(settings)
    if not tracing_enabled(settings):
        # No run happened, so any existing report is stale too.
        _remove_reports(report_dir)
        print(
            "cost_by_intent: Langfuse credentials missing "
            "(set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL); "
            "start Langfuse with `make langfuse-up` and chat a few times first."
        )
        return 1

    try:
        client = get_langfuse_client(settings)
        observations, source = fetch_observations(client, settings, args.days)
    except Exception as exc:  # noqa: BLE001 - CLI reports and exits non-zero
        # A failed run must not leave yesterday's numbers on /observability.
        _remove_reports(report_dir)
        print(f"cost_by_intent: failed to fetch Langfuse data: {exc}")
        return 2

    rows = aggregate_by_intent(observations)
    generated_at = datetime.now(timezone.utc)
    report = build_report(rows, args.days, generated_at=generated_at)
    table = render_table(
        rows, args.days, source=source, generated_at=generated_at
    )

    report_dir.mkdir(parents=True, exist_ok=True)
    txt_path = report_dir / REPORT_TXT
    json_path = report_dir / REPORT_JSON
    txt_path.write_text(table, encoding="utf-8")
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(table)

    # Keep stdout machine-parseable with --json; status goes to stderr.
    print(f"cost_by_intent: source={source}", file=sys.stderr)
    print(f"cost_by_intent: wrote {txt_path} and {json_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
