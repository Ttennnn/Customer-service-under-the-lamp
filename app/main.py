import json

from fastapi import Depends, FastAPI
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app.config import Settings, build_llm, get_settings
from app.prompts.customer_service import render_system_prompt
from app.schemas import ChatRequest, ExtractRequest
from app.services.chat import stream_chat
from app.services.extract import extract_after_sale
from app.services.session import SessionStore

app = FastAPI()

_store = SessionStore()


def get_store_dep() -> SessionStore:
    """Return the module-level session store singleton."""
    return _store


def get_llm_dep(settings: Settings = Depends(get_settings)):
    """Build the LLM from settings (overridable in tests)."""
    return build_llm(settings)


@app.post("/chat", response_class=EventSourceResponse)
def chat(
    req: ChatRequest,
    settings: Settings = Depends(get_settings),
    llm=Depends(get_llm_dep),
    store=Depends(get_store_dep),
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
            store,
            render_system_prompt(),
            req.session_id,
            req.message,
            settings.max_context_tokens,
        ):
            yield ServerSentEvent(raw_data=token)
    except Exception as exc:  # noqa: BLE001 - stream already in flight, must end cleanly
        yield ServerSentEvent(
            raw_data=json.dumps({"error": str(exc)}, ensure_ascii=False)
        )
    yield ServerSentEvent(raw_data="[DONE]")


@app.post("/extract")
def extract(req: ExtractRequest, llm=Depends(get_llm_dep)):
    """Extract structured after-sale fields from free-form text."""
    result = extract_after_sale(llm, req.text)
    return result.model_dump()
