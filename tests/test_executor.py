# tests/test_executor.py
import time
from langchain.tools import tool
from app.tools.executor import execute_tool

@tool
def fast_tool(x: int) -> str:
    """double the input"""
    return str(x * 2)

@tool
def slow_tool(x: int) -> str:
    """sleep forever"""
    time.sleep(10)
    return str(x)

@tool
def boom_tool(x: int) -> str:
    """always raise"""
    raise RuntimeError("boom")

def test_execute_success():
    assert execute_tool(fast_tool, {"x": 3}, timeout_s=5) == "6"

def test_execute_bad_args_returns_error():
    out = execute_tool(fast_tool, {"x": "not-an-int"}, timeout_s=5)
    assert "失败" in out or "错误" in out

def test_execute_timeout_returns_error():
    assert "失败" in execute_tool(slow_tool, {"x": 1}, timeout_s=0.2, retries=1)

def test_execute_exception_returns_error():
    assert "失败" in execute_tool(boom_tool, {"x": 1}, timeout_s=5)
