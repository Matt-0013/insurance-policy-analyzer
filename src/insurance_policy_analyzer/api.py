"""FastAPI service: typed endpoints over the RAG core and the comparison matrix.

The retrieval stack (torch, chromadb, the embedding model) is imported inside
the endpoint instead of at module import, so /health answers immediately and the
OpenAPI schema loads without downloading a model. The startup hook then pays the
import and model load once, at boot, so the first question is not penalised.
"""
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import registry
from .compare import build_cell, load_policies
from .config import MODES, TOP_K
from .render import to_markdown
from .schema import FIELDS
from .status import index_report

log = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Warm the retrieval stack, then serve.

    A failure is logged and swallowed: without an index or without network the
    API still starts, and /health is what reports that it cannot answer. Failing
    startup instead would turn a missing volume into a crash loop that says less
    than the health check does.
    """
    if os.getenv("WARM_ON_STARTUP", "1") != "0":
        started = time.perf_counter()
        try:
            from .search import warmup

            info = warmup()
            log.info(
                "warm-up done in %d ms: %d chunks, %d vectors, %d dims",
                info["ms"], info["chunks"], info["vectors"], info["dimensions"],
            )
        except Exception as exc:  # noqa: BLE001 - startup must not die on this
            log.warning(
                "warm-up skipped after %.1fs: %s", time.perf_counter() - started, exc
            )
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Insurance Policy Analyzer",
    version="0.2.0",
    description=(
        "Grounded answers with page citations, and evidence-based comparisons, "
        "over Indian health insurance policy documents."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class Turn(BaseModel):
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=TOP_K, ge=1, le=20)
    policy: str | None = Field(default=None, description="Restrict retrieval to one policy id.")
    mode: str = Field(default="hybrid", description=f"One of: {', '.join(MODES)}")
    history: list[Turn] = Field(default_factory=list)
    use_llm_rewrite: bool = True


class Source(BaseModel):
    n: int
    policy: str
    policy_label: str | None = None
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    text: str | None = None
    score: float | None = None
    retrieval: dict = Field(default_factory=dict)


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    grounding: dict
    retrieval: dict
    usage: dict
    timings_ms: dict


class CompareRequest(BaseModel):
    policies: list[str] | None = Field(default=None, description="Subset to compare; default all.")


class CellOut(BaseModel):
    field: str
    label: str
    policy: str
    state: str
    quote: str
    pages: list[int]
    aliases_tried: int


class CompareResponse(BaseModel):
    policies: list[str]
    cells: list[CellOut]
    markdown: str


def ask_pipeline(question: str, **kwargs) -> dict:
    """Import seam: the heavy retrieval stack loads on the first request.

    The question is a separate parameter on purpose. Declaring the seam as
    **kwargs let the endpoint's positional call raise TypeError only inside the
    container, where a unit test that replaces this function cannot see it.
    """
    from .pipeline import ask

    return ask(question, **kwargs)


@app.get("/health")
def health(response: Response) -> dict:
    """Liveness plus whether the index is really there.

    503 when it is not: a probe that says 200 on an empty store keeps a broken
    container in rotation, and the failure surfaces later as a 500 from /ask.
    """
    index = index_report()
    if not index["present"]:
        response.status_code = 503
    return {
        "status": "ok" if index["present"] else "degraded",
        "policies": len(registry.policy_ids()),
        "modes": list(MODES),
        "index": index,
    }


@app.get("/documents")
def documents() -> list[dict]:
    result = []
    for policy_id in registry.policy_ids():
        meta = registry.info(policy_id)
        result.append(
            {
                "id": policy_id,
                "insurer": meta["insurer"],
                "name": meta["name"],
                "doc_type": meta["doc_type"],
                "aliases": meta["aliases"],
            }
        )
    return result


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest) -> dict:
    if request.mode not in MODES:
        raise HTTPException(
            status_code=422, detail=f"mode must be one of: {', '.join(MODES)}"
        )
    if request.policy and request.policy not in registry.policy_ids():
        raise HTTPException(status_code=404, detail=f"unknown policy: {request.policy}")

    try:
        return ask_pipeline(
            request.question,
            k=request.k,
            policy=request.policy,
            history=[turn.model_dump() for turn in request.history],
            mode=request.mode,
            use_llm_rewrite=request.use_llm_rewrite,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/compare", response_model=CompareResponse)
def compare_endpoint(request: CompareRequest) -> dict:
    by_policy = load_policies()
    if not by_policy:
        raise HTTPException(status_code=503, detail="no chunks indexed; run ingestion first")

    unknown = [policy for policy in request.policies or [] if policy not in by_policy]
    if unknown:
        raise HTTPException(status_code=404, detail=f"unknown policy: {', '.join(unknown)}")

    chosen = sorted(request.policies or by_policy)
    cells = []
    for spec in FIELDS:
        for policy in chosen:
            cell = build_cell(spec, policy, by_policy[policy])[0]
            cells.append(
                CellOut(
                    field=spec.key,
                    label=spec.label,
                    policy=policy,
                    state=cell.state,
                    quote=cell.quote,
                    pages=list(cell.pages),
                    aliases_tried=cell.aliases_tried,
                ).model_dump()
            )

    return {"policies": chosen, "cells": cells, "markdown": to_markdown(chosen)}
