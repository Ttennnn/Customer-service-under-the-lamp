"""Ch05/ch06 page tests (Task 16 compatibility + ch06 shell)."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


def test_ch05_page_redirects_to_ch06():
    with TestClient(app) as client:
        response = client.get("/ch05", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/ch06"


def test_ch06_page_route_serves_html():
    with TestClient(app) as client:
        response = client.get("/ch06")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "lamp-cs-ch06-session-id" in response.text


def test_root_serves_ch06_page_directly():
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "lamp-cs-ch06-session-id" in response.text
    assert "/ch06/chat" in response.text


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
