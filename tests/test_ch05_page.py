"""Ch05/ch06 page tests (Task 16 compatibility + ch06 shell)."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


def test_ch05_page_serves_latest_ch06_page():
    with TestClient(app) as client:
        response = client.get("/ch05")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["cache-control"] == "no-store"
    assert "lamp-cs-ch06-session-id" in response.text
    assert "/ch06/chat" in response.text


def test_ch06_page_route_serves_html():
    with TestClient(app) as client:
        response = client.get("/ch06")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["cache-control"] == "no-store"
    assert "lamp-cs-ch06-session-id" in response.text


def test_root_serves_ch07_page_directly():
    # The root entry point moved to the ch07 multi-conversation page; /ch06
    # still serves the previous page.
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["cache-control"] == "no-store"
    assert "lamp-cs-ch07-user-id" in response.text
    assert "/ch07/chat" in response.text


def test_ch06_page_shell_has_required_contract_strings():
    html = (Path(__file__).parents[1] / "app" / "static" / "ch06.html").read_text(
        encoding="utf-8"
    )
    for text in [
        "ch06",
        "lamp-cs-ch06-session-id",
        "/ch06/chat",
        "<input",
        "发送",
    ]:
        assert text in html
