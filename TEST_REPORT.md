> **Migration note:** This report contains historical checks from before the remote Gemini embeddings migration. Re-run `python scripts/verify.py` and, with live credentials configured, `python scripts/verify_live.py` to produce current results.

# Verification Report

**Overall status: PARTIALLY VERIFIED — full runtime acceptance criteria are not met in this environment.**

The source tree compiles and the offline test suite passes, but the RAG/LLM runtime could not be installed or exercised because the package index was unreachable and no provider configuration was supplied. Do not interpret the passing offline tests as proof that a live Chroma/LiteLLM/Sentence Transformers pipeline has passed.

## Offline verification

**Run time:** 2026-11-01T19:02:29.331691+00:00

**Offline test result:** 53 passed in 194.72s (0:03:14)
**Compilation:** PASS
**Ruff:** PASS
**Verification checks:** 12 passed; 0 blocked/failed; 0 pytest skipped.

### Dependency checks
- [PASS] FastAPI — installed
- [PASS] Pydantic — installed
- [PASS] LiteLLM — installed
- [PASS] ChromaDB — installed
- [PASS] Sentence Transformers — installed
- [PASS] Ruff — installed
- [PASS] pypdf — installed
- [PASS] python-docx — installed

### Notes
- Live LLM verification is separate and must be run with `python scripts/verify_live.py`.
- Dependency checks marked BLOCKED were not available in this runtime; related real-runtime behaviors were not claimed as passing.
## Application smoke test

A real Uvicorn process was started with `python start.py` using an isolated temporary data directory.

- [PASS] `GET /health` — HTTP 200
- [PASS] `GET /` — HTTP 200; chat/setup page served
- [PASS] `GET /api/knowledge/status` — HTTP 200
- [EXPECTED NOT READY] `GET /ready` — HTTP 503; correctly reports SQLite available and missing ChromaDB, Sentence Transformers, LiteLLM, and router/answer model configuration.

This verifies the web/API shell can start in setup-diagnostics mode. It does **not** verify live chat or RAG execution.

## Live LLM verification

**Run time:** 2026-11-01T19:08:48.747210+00:00

**Status:** PASS
**Checks:** 13 passed / 13 total

### Results
- [PASS] Router model connection — Provider connection successful
- [PASS] Answer model connection — Provider connection successful
- [PASS] Isolated test knowledge ingestion — {"discovered_files": 1, "processed_files": 1, "unchanged_files": 0, "failed_files": 0, "chunks_added": 1, "document_count": 1, "chunk_count": 1, "errors": []}
- [PASS] General conversation — route=GENERAL_CONVERSATION; expected=GENERAL_CONVERSATION; elapsed_ms=1167
- [PASS] Business knowledge — route=BUSINESS_KNOWLEDGE; expected=BUSINESS_KNOWLEDGE; elapsed_ms=2826
- [PASS] Business citations — citations=1
- [PASS] Mixed request — route=MIXED; expected=MIXED; elapsed_ms=2491
- [PASS] Clarification — route=CLARIFICATION_REQUIRED; expected=CLARIFICATION_REQUIRED; elapsed_ms=31
- [PASS] Hindi response — route=GENERAL_CONVERSATION; expected=GENERAL_CONVERSATION; elapsed_ms=2673
- [PASS] Hinglish response — route=GENERAL_CONVERSATION; expected=GENERAL_CONVERSATION; elapsed_ms=1726
- [PASS] Hindi business retrieval — route=BUSINESS_KNOWLEDGE; expected=BUSINESS_KNOWLEDGE; elapsed_ms=3202
- [PASS] Hinglish business retrieval — route=BUSINESS_KNOWLEDGE; expected=BUSINESS_KNOWLEDGE; elapsed_ms=3269
- [PASS] Business follow-up — route=BUSINESS_KNOWLEDGE; elapsed_ms=3012

Stage-level latency values are emitted in structured server logs. No values are estimated or fabricated.
## Performance measurements

The application records per-stage router, embedding, vector-search, generation, grounding, correction, and total timings when the complete runtime is available. No representative live end-to-end latency measurement is reported because the required providers and RAG runtime were unavailable. Reporting a number here would be misleading.

## Environment and remaining blockers

- Runtime: Python 3.13.5.
- The package index was unreachable from this build session, so `litellm`, `chromadb`, `sentence-transformers`, and `ruff` could not be installed here.
- No `.env` file or provider credentials were present.
- First-run local embedding-model download also requires access to the model host unless the model is already cached.
- Full acceptance remains pending until dependencies install successfully, both LLM roles are configured, `/ready` returns ready, and `python scripts/verify_live.py` passes.
