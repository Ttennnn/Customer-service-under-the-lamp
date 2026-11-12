# tests/test_db_config.py
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from app.config import Settings
from app.db.base import Base
from app.db import session as db_session

def test_database_url_from_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "mysql+pymysql://u:p@localhost:3306/x")
    assert Settings().database_url == "mysql+pymysql://u:p@localhost:3306/x"

def test_engine_and_factory_lazy(monkeypatch):
    monkeypatch.setattr(db_session, "get_settings", lambda: Settings(
        openai_api_key="sk-test", database_url="sqlite://"
    ))
    db_session.get_engine.cache_clear() if hasattr(db_session.get_engine, "cache_clear") else None
    assert isinstance(db_session.get_engine(), Engine)
    assert isinstance(db_session.get_session_factory(), sessionmaker)
