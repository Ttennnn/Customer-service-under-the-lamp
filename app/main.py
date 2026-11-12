import json
import logging

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.sse import EventSourceResponse, ServerSentEvent
from sqlalchemy.orm import Session

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
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_llm_dep(settings: Settings = Depends(get_settings)):
    """Build the LLM from settings (overridable in tests)."""
    return build_llm(settings)


@app.post("/chat", response_class=EventSourceResponse)
def chat(
    req: ChatRequest,
    settings: Settings = Depends(get_settings),
    llm=Depends(get_llm_dep),
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
