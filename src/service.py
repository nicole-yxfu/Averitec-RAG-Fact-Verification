"""
service.py — a FastAPI service that wraps the open-web fact-verification
pipeline behind a single HTTP endpoint.

    POST /verify   {"claim": "..."}  -> verdict + evidence + latency + cost
    GET  /health                     -> liveness + which model/pipeline is wired
    GET  /docs                       -> auto-generated Swagger UI (try it live)

The service does NOT reimplement the pipeline — it calls the exact retrieval,
prompt-building, LLM and label-normalisation functions used in the offline
experiments (`open_web_baseline`, `reprompt_eval`, `bm25_baseline`). So the
deployed behaviour is faithful to the evaluated *open-web* system, the
project's best non-oracle configuration (Macro-F1 0.4495 on the official dev
set). One claim in -> retrieve open-web evidence -> prompt the LLM -> return a
grounded verdict, timed and costed per request.

Run locally:
    uvicorn src.service:app --reload --port 8000
    # or:  python src/service.py
Then open http://localhost:8000/docs

Requires the same Azure OpenAI credentials as the experiments (.env):
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT
"""
import os
import sys
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Reuse the research pipeline verbatim — the service is a thin wrapper.
from open_web_baseline import ddg_search_cached, hits_to_passages
from reprompt_eval import PLAIN_PROMPT
from bm25_baseline import (
    build_langchain_client, call_langchain,
    build_user_input, normalize_llm_output,
)

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────
# gpt-4.1-mini list price (USD per 1M tokens); override via env for other
# models / regional rates. Used only to report an estimated per-request cost.
COST_PER_1M_INPUT = float(os.environ.get("AZURE_COST_PER_1M_INPUT", "0.40"))
COST_PER_1M_OUTPUT = float(os.environ.get("AZURE_COST_PER_1M_OUTPUT", "1.60"))
DDG_CACHE_DIR = os.environ.get("DDG_CACHE_DIR", os.path.join(
    os.path.dirname(SCRIPT_DIR), "results", "duckduckgo_cache"))
MODEL_NAME = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")

VERDICT_LABELS = {
    0: "Supported",
    1: "Refuted",
    2: "Not enough evidence",
    3: "Conflicting / cherry-picking",
}
CANONICAL = {
    0: "supported", 1: "refuted",
    2: "not enough evidence", 3: "conflicting evidence/cherrypicking",
}


# ── Schemas ──────────────────────────────────────────────────────────────────
class VerifyRequest(BaseModel):
    claim: str = Field(
        ..., min_length=3, max_length=2000,
        description="The factual claim to verify.",
        examples=["The Eiffel Tower is taller than the Great Pyramid of Giza."],
    )
    top_k: int = Field(
        5, ge=1, le=10,
        description="How many open-web passages to retrieve as evidence.",
    )


class EvidenceItem(BaseModel):
    url: str
    title: str | None = None
    snippet: str


class Latency(BaseModel):
    retrieval_ms: float = Field(..., description="Open-web retrieval wall time.")
    llm_ms: float = Field(..., description="LLM verification wall time.")
    total_ms: float


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float = Field(..., description="Estimated cost of this request.")


class VerifyResponse(BaseModel):
    claim: str
    verdict: str = Field(..., description="Canonical AVeriTeC label (lowercase).")
    verdict_label: str = Field(..., description="Human-readable verdict.")
    verdict_id: int | None = Field(..., description="0=Sup 1=Ref 2=NEE 3=Conf; null if unparseable.")
    evidence: list[EvidenceItem]
    n_evidence: int
    latency: Latency
    usage: Usage
    model: str
    raw_output: str = Field(..., description="The LLM's raw text output, pre-normalisation.")


class HealthResponse(BaseModel):
    status: str
    model: str
    evidence_source: str
    llm_configured: bool


# ── Pipeline (one claim) ──────────────────────────────────────────────────────
def _split_passage(passage: str) -> tuple[str | None, str]:
    """hits_to_passages stores 'Title: X\\nSnippet: Y'; pull the two apart."""
    if passage.startswith("Title:"):
        head, _, tail = passage.partition("\nSnippet:")
        return head[len("Title:"):].strip() or None, tail.strip()
    return None, passage


def verify_claim(client, claim: str, top_k: int = 5) -> VerifyResponse:
    """Retrieve open-web evidence, prompt the LLM, return a timed/costed verdict.

    Reuses ddg_search_cached + build_user_input + call_langchain +
    normalize_llm_output so the result matches the offline open-web system.
    """
    # 1. Open-web retrieval (disk-cached per query, so repeats are instant).
    t0 = time.perf_counter()
    hits = ddg_search_cached(claim.strip(), DDG_CACHE_DIR, top_k=top_k)
    passages = hits_to_passages(hits)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    # 2. Build the SAME prompt the experiments use (no speaker/source metadata
    #    for a bare API claim, so use_metadata=False).
    claim_obj = SimpleNamespace(
        original_claim=claim, speaker=None, reporting_source=None)
    user_input = build_user_input(claim_obj, passages, use_metadata=False)

    # 3. LLM verification.
    t1 = time.perf_counter()
    try:
        raw, prompt_tokens, completion_tokens = call_langchain(
            client, PLAIN_PROMPT, user_input)
    except Exception as e:
        err = str(e)
        if "content_filter" in err or "ResponsibleAI" in err:
            raise HTTPException(
                status_code=422,
                detail="The claim or its retrieved evidence was blocked by the "
                       "model's content filter.")
        raise HTTPException(status_code=502, detail=f"LLM backend error: {err[:200]}")
    llm_ms = (time.perf_counter() - t1) * 1000

    verdict_id = normalize_llm_output(raw)
    cost = (prompt_tokens / 1e6) * COST_PER_1M_INPUT + \
           (completion_tokens / 1e6) * COST_PER_1M_OUTPUT

    evidence = []
    for _score, url, passage in passages:
        title, snippet = _split_passage(passage)
        evidence.append(EvidenceItem(url=url, title=title, snippet=snippet))

    return VerifyResponse(
        claim=claim,
        verdict=CANONICAL.get(verdict_id, "unparseable"),
        verdict_label=VERDICT_LABELS.get(verdict_id, "Unparseable"),
        verdict_id=verdict_id,
        evidence=evidence,
        n_evidence=len(evidence),
        latency=Latency(
            retrieval_ms=round(retrieval_ms, 1),
            llm_ms=round(llm_ms, 1),
            total_ms=round(retrieval_ms + llm_ms, 1)),
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=round(cost, 6)),
        model=MODEL_NAME,
        raw_output=raw,
    )


# ── App ──────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the Azure OpenAI client once at startup (not per request)."""
    try:
        app.state.client = build_langchain_client()
        app.state.llm_configured = True
    except Exception as e:
        # Start anyway so /health and /docs work and report the misconfig,
        # instead of crashing the whole server on a missing key.
        print(f"[startup] LLM client not configured: {e}")
        app.state.client = None
        app.state.llm_configured = False
    yield


app = FastAPI(
    title="AVeriTeC Open-Web Fact Verifier",
    description="Retrieval-augmented fact verification: a claim in, a grounded "
                "verdict out, with per-request latency and cost. Wraps the "
                "project's best non-oracle configuration (open-web evidence, "
                "Macro-F1 0.4495).",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    return HealthResponse(
        status="ok",
        model=MODEL_NAME,
        evidence_source="open-web (DuckDuckGo, top-k)",
        llm_configured=getattr(app.state, "llm_configured", False),
    )


@app.post("/verify", response_model=VerifyResponse, tags=["verify"])
def verify(req: VerifyRequest):
    client = getattr(app.state, "client", None)
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="LLM client not configured; set AZURE_OPENAI_* in .env.")
    return verify_claim(client, req.claim, req.top_k)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("service:app", host="127.0.0.1",
                port=int(os.environ.get("PORT", "8000")), reload=False)
