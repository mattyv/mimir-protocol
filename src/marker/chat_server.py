"""Mimir-Protocol chat server — Modal ASGI endpoint.

Loads pre-trained axioms from the Modal Volume at startup, then serves
a /chat endpoint that runs AxiomSession inference.

Deploy:  modal deploy modal_blends.py::deploy_chat
URL:     https://<your-workspace>--mimir-chat.modal.run
"""

from __future__ import annotations

import uuid
from typing import Any

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

_model = None
_tokenizer = None
_axiom_mlps: list = []
_sessions: dict[str, Any] = {}  # session_id → AxiomSession


def _load_model_and_axioms(model_name: str, axiom_dir: str) -> None:
    global _model, _tokenizer, _axiom_mlps
    from pathlib import Path

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from marker.axiom_store import load_axiom

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model {model_name} on {device}...")
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
            print(f"  loaded {axiom.term} ({'skill' if axiom.skill_mode else 'fact'})")
        except Exception as e:
            print(f"  failed to load {pt_file.name}: {e}")

    print(f"Ready — {len(_axiom_mlps)} axioms: {[a.term for a in _axiom_mlps]}")


def create_app(
    model_name: str = "Qwen/Qwen2.5-32B",
    axiom_dir: str = "/axioms",
) -> FastAPI:
    web_app = FastAPI(title="Mimir-Protocol Chat")
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @web_app.on_event("startup")
    async def startup() -> None:
        _load_model_and_axioms(model_name, axiom_dir)

    class ChatRequest(BaseModel):
        message: str
        session_id: str | None = None

    class ChatResponse(BaseModel):
        response: str
        session_id: str
        active_axioms: list[str]

    @web_app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest) -> ChatResponse:
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
        return {"status": "reset"}

    @web_app.get("/axioms")
    async def list_axioms() -> dict:
        return {
            "axioms": [
                {"term": a.term, "skill": a.skill_mode, "dependencies": a.dependencies}
                for a in _axiom_mlps
            ]
        }

    @web_app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "axioms_loaded": len(_axiom_mlps), "model": model_name}

    return web_app
