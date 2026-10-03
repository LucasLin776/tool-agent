from __future__ import annotations

import os
from pathlib import Path

from tool_agent.serving.recommender import RemoteApiRecommender

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Install the optional service dependencies: pip install -e '.[service]'") from exc


class RecommendRequest(BaseModel):
    task: str
    top_k: int = 8


class ChatRequest(BaseModel):
    query: str


app = FastAPI(title="API Recommendation Service")
_agent: RemoteApiRecommender | None = None


def get_agent() -> RemoteApiRecommender:
    global _agent
    if _agent is None:
        _agent = RemoteApiRecommender(
            base_model=os.getenv("TOOL_AGENT_BASE_MODEL", "Qwen/Qwen3.5-4B"),
            checkpoint=os.getenv("TOOL_AGENT_CHECKPOINT"),
            catalog_path=Path(os.getenv("TOOL_AGENT_API_CATALOG", "data/processed/toolbench_api_catalog.jsonl")),
            default_top_k=int(os.getenv("TOOL_AGENT_TOP_K", "8")),
        )
    return _agent


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/recommend")
def recommend(request: RecommendRequest) -> dict:
    return get_agent().recommend(request.task, request.top_k)


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    """Alias for the API recommendation endpoint."""
    return get_agent().recommend(request.query)
