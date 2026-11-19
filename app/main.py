import json
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.sse import EventSourceResponse, ServerSentEvent
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from app.agent.tools import legacy_tools
from app.api.ch05 import api_router as ch05_api_router
from app.api.ch05 import page_router as ch05_page_router
from app.api.ch06 import api_router as ch06_api_router
from app.api.ch06 import close_ch06_graph
from app.api.ch06 import page_router as ch06_page_router
from app.api.ch07 import CH07_PAGE
from app.api.ch07 import api_router as ch07_api_router
from app.api.ch07 import page_router as ch07_page_router
from app.api.conversations import api_router as conversations_api_router
from app.api.kb import api_router as kb_api_router
from app.api.kb import page_router as kb_page_router
from app.api.rag_eval import api_router as rag_eval_api_router
from app.api.rag_eval import page_router as rag_eval_page_router
from app.config import Settings, build_llm, get_settings
from app.db.session import get_db
from app.logging_config import configure_file_logging
from app.prompts.customer_service import render_system_prompt
from app.schemas import ChatRequest, ExtractRequest
from app.services.chat import stream_chat
from app.services.context_budget import validate_budget
from app.services.extract import extract_after_sale
from app.services.summary_job import shutdown_summary_executor
from app.tools.registry import get_all_specs

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Wire logging, self-check the context budget, release on shutdown.

    The ch06 graph is still built lazily on the first request (and the ch07
    graph is never built here), so startup stays cheap and offline-safe.
    """
    configure_file_logging()
    validate_budget(get_settings())
    try:
        yield
    finally:
        close_ch06_graph(app)
        shutdown_summary_executor()


app = FastAPI(lifespan=lifespan)

# Populated on the first /ch06 request by ``get_ch06_graph_dep``.  State is
# initialised here so the lifespan and dependency share one lock without
# constructing any LLM or opening the SQLite checkpoint at import time.
app.state.ch06_graph = None
app.state.ch06_checkpoint_handle = None
app.state.ch06_graph_lock = threading.Lock()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(kb_page_router)
app.include_router(kb_api_router)
app.include_router(rag_eval_page_router)
app.include_router(rag_eval_api_router)
app.include_router(ch06_page_router)
app.include_router(ch06_api_router)
app.include_router(ch07_page_router)
app.include_router(ch07_api_router)
app.include_router(conversations_api_router)
app.include_router(ch05_page_router)
app.include_router(ch05_api_router)


def get_llm_dep(settings: Settings = Depends(get_settings)):
    """Build the LLM from settings (overridable in tests)."""
    return build_llm(settings)


def get_self_check_llm_dep(settings: Settings = Depends(get_settings)):
    """Build the temperature-0 LLM used for the evidence self-check."""
    return build_llm(settings, temperature=0.0)


@app.post("/chat", response_class=EventSourceResponse)
def chat(
    req: ChatRequest,
    settings: Settings = Depends(get_settings),
    llm=Depends(get_llm_dep),
    self_check_llm=Depends(get_self_check_llm_dep),
    session: Session = Depends(get_db),
):
    """Stream an assistant reply as Server-Sent Events.

    Headers (HTTP 200) are already sent before the generator runs, so an
    upstream failure mid-stream cannot be surfaced as a 500. Instead we yield
    an error frame and still terminate with ``[DONE]`` so the client never
    hangs waiting for the stream to close.
    """
    try:
        # Legacy page: exactly the three read tools, resolved from the
        # registry. It intentionally has no write/control/MCP tools (no
        # confirmation UI here; logistics is MCP-only in ch07).
        specs = get_all_specs(settings)
        tools = legacy_tools(specs)
        for token in stream_chat(
            llm,
            session,
            tools,
            render_system_prompt(),
            req.session_id,
            req.message,
            settings.max_context_tokens,
            self_check_llm=self_check_llm,
            specs=specs,
            settings=settings,
        ):
            yield ServerSentEvent(raw_data=token)
    except Exception:  # noqa: BLE001 - stream already in flight, must end cleanly
        logger.exception("chat stream failed")
        yield ServerSentEvent(
            raw_data=json.dumps(
                {"error": "客服服务暂时不可用,请稍后重试"}, ensure_ascii=False
            )
        )
    yield ServerSentEvent(raw_data="[DONE]")


@app.post("/extract")
def extract(req: ExtractRequest, llm=Depends(get_llm_dep)):
    """Extract structured after-sale fields from free-form text."""
    try:
        result = extract_after_sale(llm, req.text)
    except Exception:  # noqa: BLE001 - upstream/parse failure surfaced as 502
        logger.exception("extract failed")
        return JSONResponse(
            status_code=502, content={"detail": "结构化提取失败,请稍后重试"}
        )
    return result.model_dump()


# The root URL is the chat entry point. Chapter 7 owns the chat experience
# (multi-conversation sidebar on top of the ch06 order picker/refund form), so
# serve its native page directly at ``/``; ``/ch06`` keeps the old page.
@app.get("/", include_in_schema=False)
def root_chat_page() -> FileResponse:
    return FileResponse(
        CH07_PAGE,
        media_type="text/html",
        headers={"Cache-Control": "no-store"},
    )


# The legacy Vite/React widget is still mounted for any static assets when a
# production build exists. The ``/`` route above takes precedence for the page
# itself; ``POST /chat`` remains available for API compatibility.
_FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"

if _FRONTEND_DIST.is_dir():
    app.mount(
        "/",
        StaticFiles(directory=_FRONTEND_DIST, html=True),
        name="frontend",
    )
