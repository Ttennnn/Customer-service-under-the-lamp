import json
import logging
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.sse import EventSourceResponse, ServerSentEvent
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from app.api.kb import api_router as kb_api_router
from app.api.kb import page_router as kb_page_router
from app.api.rag_eval import api_router as rag_eval_api_router
from app.api.rag_eval import page_router as rag_eval_page_router
from app.config import Settings, build_llm, get_settings
from app.db.session import get_db
from app.prompts.customer_service import render_system_prompt
from app.schemas import ChatRequest, ExtractRequest
from app.services.chat import stream_chat
from app.services.extract import extract_after_sale
from app.tools.registry import ALL_TOOLS

logger = logging.getLogger(__name__)

app = FastAPI()

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
        for token in stream_chat(
            llm,
            session,
            ALL_TOOLS,
            render_system_prompt(),
            req.session_id,
            req.message,
            settings.max_context_tokens,
            self_check_llm=self_check_llm,
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


# The chat widget is a Vite/React app. In local runs we serve its production
# build from FastAPI so the chat page shares port 8000 with the API. The dev
# server on 5173 still works for frontend iteration.
_FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"

if _FRONTEND_DIST.is_dir():
    app.mount(
        "/",
        StaticFiles(directory=_FRONTEND_DIST, html=True),
        name="frontend",
    )
else:
    @app.get("/", response_class=HTMLResponse)
    def frontend_build_missing():
        return HTMLResponse(
            """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>聊天页尚未构建</title>
<body style="font-family:sans-serif;background:#171310;color:#f2e9de;padding:40px;line-height:1.8">
<h1>聊天页尚未构建</h1>
<p>先在 <code>frontend/</code> 下执行 <code>npm run build</code>,然后重启后端。</p>
<p>也可以继续用 Vite 开发服务器:<code>cd frontend &amp;&amp; npm run dev</code>,地址
<a style="color:#f0b76a" href="http://localhost:5173/">http://localhost:5173/</a>。</p>
<p><a style="color:#f0b76a" href="/kb">知识录入页</a> ·
<a style="color:#f0b76a" href="/rag-eval">RAG 评估</a></p>
</body></html>""",
            status_code=503,
        )
