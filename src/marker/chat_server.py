"""Mimir-Protocol chat server — Modal ASGI endpoint."""

from __future__ import annotations

import uuid
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

_model = None
_tokenizer = None
_axiom_mlps: list = []
_sessions: dict[str, Any] = {}


def preload(model_name: str, axiom_dir: str) -> None:
    """Load model and axioms synchronously. Called before FastAPI starts."""
    global _model, _tokenizer, _axiom_mlps
    from pathlib import Path

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from marker.axiom_store import load_axiom

    print(f"Loading {model_name}...")
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = (
        AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        .to("cuda")
        .eval()
    )
    for p in _model.parameters():
        p.requires_grad_(False)

    for pt_file in sorted(Path(axiom_dir).glob("*.pt")):
        try:
            axiom = load_axiom(pt_file, _model, _tokenizer)
            _axiom_mlps.append(axiom)
            print(f"  loaded {axiom.term}")
        except Exception as e:
            print(f"  failed {pt_file.name}: {e}")

    print(f"Ready — {len(_axiom_mlps)} axioms")


def create_app() -> FastAPI:
    web_app = FastAPI(title="Mimir-Protocol")
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @web_app.get("/", response_class=HTMLResponse)
    async def root() -> str:
        axioms = [a.term for a in _axiom_mlps]
        return f"""<html><body style="font-family:monospace;padding:2rem;background:#0d1117;color:#c9d1d9">
<h2>Mimir-Protocol Chat API</h2>
<p>Status: <strong>ready</strong></p>
<p>Axioms: {axioms}</p>
<p>Chat UI: <a href="https://mattyv.github.io/mimir-protocol/" style="color:#58a6ff">mattyv.github.io/mimir-protocol</a></p>
<hr style="border-color:#30363d">
<code>POST /chat  GET /axioms  GET /health</code>
</body></html>"""

    @web_app.get("/health")
    async def health() -> dict:
        return {"status": "ready", "axioms_loaded": len(_axiom_mlps)}

    @web_app.post("/chat")
    async def chat(request: Request) -> JSONResponse:
        body = await request.json()
        message = body.get("message", "")
        session_id = body.get("session_id")
        if not message:
            raise HTTPException(status_code=400, detail="message required")

        from marker.run_axiom_mlp_demo import AxiomSession

        sid = session_id or str(uuid.uuid4())
        if sid not in _sessions:
            _sessions[sid] = AxiomSession(_model, _axiom_mlps)

        session: AxiomSession = _sessions[sid]
        response = session.chat(_model, _tokenizer, message, max_new=200)
        return JSONResponse(
            {
                "response": response,
                "session_id": sid,
                "active_axioms": sorted(session.active),
            }
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
            "ready": True,
        }

    return web_app
