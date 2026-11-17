"""Ch05 native page tests (Task 12)."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


def test_ch05_page_has_required_contract_strings():
    html = (Path(__file__).parents[1] / "app" / "static" / "ch05.html").read_text(
        encoding="utf-8"
    )
    for text in [
        "转人工", "建工单", "已转接人工客服", "客服小猫",
        "/ch05/chat", "/ch05/tickets", "lamp-cs-ch05-session-id",
    ]:
        assert text in html
    assert "confirm(" in html


def test_ch05_page_has_no_data_timeout_contract():
    """Pin the spec §7.2 90-second no-data watchdog and retry path."""
    html = (Path(__file__).parents[1] / "app" / "static" / "ch05.html").read_text(
        encoding="utf-8"
    )
    for text in [
        "AbortController",
        "90000",
        "AbortError",
        "setTimeout",
        "clearTimeout",
        "startWatchdog",
        "重试",
    ]:
        assert text in html


def test_ch05_page_route_serves_html():
    with TestClient(app) as client:
        response = client.get("/ch05")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "转人工" in response.text
    assert "lamp-cs-ch05-session-id" in response.text


def test_root_redirects_to_ch05_page():
    with TestClient(app) as client:
        response = client.get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/ch05"
