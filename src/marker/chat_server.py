"""Mimir-Protocol chat server — Modal ASGI endpoint."""

from __future__ import annotations

import threading
import uuid
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

_model = None
_tokenizer = None
_axiom_mlps: list = []
_sessions: dict[str, Any] = {}
_ready = False
_loading = False
_load_error: str | None = None


def _load_model_and_axioms(model_name: str, axiom_dir: str) -> None:
    global _model, _tokenizer, _axiom_mlps, _ready, _load_error
    from pathlib import Path

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from marker.axiom_store import load_axiom

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model {model_name}...")
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = (
            AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(device).eval()
        )
        for p in _model.parameters():
            p.requires_grad_(False)

        axiom_path = Path(axiom_dir)
        for pt_file in sorted(axiom_path.glob("*.pt")):
            try:
                axiom = load_axiom(pt_file, _model, _tokenizer)
                _axiom_mlps.append(axiom)
                print(f"  loaded {axiom.term}")
            except Exception as e:
                print(f"  failed {pt_file.name}: {e}")

        _ready = True
        print(f"Ready — {len(_axiom_mlps)} axioms loaded")
    except Exception as e:
        _load_error = str(e)
        print(f"Load failed: {e}")


def create_app(
    model_name: str = "Qwen/Qwen2.5-32B",
    axiom_dir: str = "/axioms",
) -> FastAPI:
    web_app = FastAPI(title="Mimir-Protocol")
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @web_app.on_event("startup")
    async def startup() -> None:
        # Load in background so the server accepts requests immediately.
        # /health returns loading status; /chat returns 503 until ready.
        global _loading
        _loading = True
        t = threading.Thread(
            target=_load_model_and_axioms, args=(model_name, axiom_dir), daemon=True
        )
        t.start()

    @web_app.get("/", response_class=HTMLResponse)
    async def root() -> str:
        status = "loading..." if _loading and not _ready else ("ready" if _ready else "starting")
        axioms = [a.term for a in _axiom_mlps]
        return f"""<html><body style="font-family:monospace;padding:2rem;background:#0d1117;color:#c9d1d9">
<h2>Mimir-Protocol Chat API</h2>
<p>Status: <strong>{status}</strong></p>
<p>Axioms: {axioms or "loading..."}</p>
<p>Chat UI: <a href="https://mattyv.github.io/mimir-protocol/" style="color:#58a6ff">mattyv.github.io/mimir-protocol</a></p>
<hr style="border-color:#30363d">
<code>POST /chat  GET /axioms  GET /health</code>
</body></html>"""

    @web_app.get("/health")
    async def health() -> dict:
        return {
            "status": "ready" if _ready else "loading",
            "axioms_loaded": len(_axiom_mlps),
            "model": model_name,
            "error": _load_error,
        }

    class ChatRequest(BaseModel):
        message: str
        session_id: str | None = None

        model_config = {"extra": "ignore"}

    class ChatResponse(BaseModel):
        response: str
        session_id: str
        active_axioms: list[str]

    @web_app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest) -> ChatResponse:
        if not _ready:
            raise HTTPException(
                status_code=503,
                detail="Model still loading, please retry in a moment",
            )
        from marker.run_axiom_mlp_demo import AxiomSession

        sid = req.session_id or str(uuid.uuid4())
        if sid not in _sessions:
            _sessions[sid] = AxiomSession(_model, _axiom_mlps)

        session: AxiomSession = _sessions[sid]
        response = session.chat(_model, _tokenizer, req.message, max_new=200)
        return ChatResponse(
            response=response,
            session_id=sid,
            active_axioms=sorted(session.active),
        )

    @web_app.delete("/session/{session_id}")
    async def reset_session(session_id: str) -> dict:
        if session_id in _sessions:
            _sessions[session_id].reset()
            del _sessions[session_id]
        return {"status": "reset"}

    @web_app.get("/axioms")
    async def list_axioms() -> dict:
        return {
            "axioms": [
                {"term": a.term, "skill": a.skill_mode, "dependencies": a.dependencies}
                for a in _axiom_mlps
            ],
            "ready": _ready,
        }

    return web_app
