# Roadmap: prototype → production-grade RAG application

This is the plan of record for turning the current working prototype into an
end-to-end portfolio project. Each stage is independently shippable: it leaves
the tests green, the README honest, and a number you can quote.

---

## Where the project stands

**Working today**

| Area | State |
| --- | --- |
| PDF extraction | PyMuPDF4LLM with classic → layout → plain fallback, gated on 80% alphanumeric coverage |
| Cleaning | letter-spaced text repair (`SPACED_OUT` + wordninja) |
| Chunking | section-aware, numbered/markdown headings, 1500-char cap, table headers repeated on split |
| Index | Chroma (cosine) + in-process BM25 over the same `section + text` |
| Retrieval | vector, keyword, hybrid (RRF, 50 candidates), cross-encoder rerank |
| Query handling | deterministic abbreviation expansion + policy detection + small-model query rewriting |
| Generation | Groq `openai/gpt-oss-120b`, strict grounding prompt, numbered sources |
| Grounding | citation validation + figure verification (`citations.py`), refusal on empty retrieval |
| Memory | bounded conversation transcript for follow-ups |
| Retrieval eval | 12 questions incl. 2 unanswerable, hit@k + MRR, results in `eval/results/` |
| Tests | 73 tests: extraction, cleaning, chunking, fusion, citations, rewriting, context, memory, comparison, the scoring table, the API |

**Now also working:**

| Area | State |
| --- | --- |
| Comparison | 12 evidence-backed fields x 4 policies, four cell states, page per cell (`compare.py`, `schema.py`, `render.py`); every scoring weight and evidence window is a documented row in `docs/SCORING.md`, generated from the code and checked by a test |
| API | FastAPI `/ask`, `/compare`, `/documents`, `/health` with typed contracts and CORS (`api.py`); `/health` reports stored vectors against the chunk count and answers 503 when nothing is indexed |
| Startup | the lifespan hook loads the embedding model, the collection, the chunks and the BM25 index, and imports the pipeline, so the first question costs what every question costs (~18-33 s of boot) |
| UI | Streamlit: ask with citations, compare table, documents, evaluation tabs (`app.py`) |
| Answer eval | `eval/eval_ragas.py` and a committed `eval/report.md`: faithfulness 0.84, answer relevancy 0.96, context precision 0.98, rule-based grounding 0.80, refusal 1.00 on 10 questions, judge `openai/gpt-oss-20b` |
| Packaging | multi-stage non-root Dockerfile with CPU-only torch (2.56 GB, down from 7.1 GB), the embedding model baked in, and a healthcheck with a 60 s start period; docker-compose (api + ui) gates the UI on `service_healthy` and restarts a crashed API; Makefile; GitHub Actions CI |
| Tests | 73 tests covering extraction, cleaning, chunking, fusion, citations, rewriting, context, memory, comparison, the scoring table and the API |

**Still missing:** Stage 2 metadata (UIN and effective-date extraction, insurer and
doc type in chunk metadata so retrieval can filter with a `where` clause, duplicate
upload detection); `POST /documents` and `DELETE /documents/{id}`; a ~30-question
golden set with reference answers so context recall can be reported; `ruff`/`pre-commit` and an
image build in CI; rate limiting and upload validation; `docs/DECISIONS.md`,
screenshots and a demo GIF (Stages 6 and 8).

---

## How we work on every stage

1. **Write the acceptance criteria first.** If you cannot state how you will know
   it works, the stage is not defined yet.
2. **Build the smallest version that can pass them.** No speculative abstraction.
3. **Measure before and after.** Retrieval and answer quality are numbers, not
   opinions. Save the raw results under `eval/results/` with the config in the name.
4. **Test the tricky logic, not every line.** Parsers, thresholds and fallbacks
   get tests; glue code is exercised by the eval and the API tests.
5. **Commit with the why.** `feat: rerank fused candidates with a cross-encoder`
   plus one line on the measured effect beats `update code`.
6. **Update this file and the README.** A recruiter reads the README, the commit
   log, and maybe one source file. Optimise for those three.

---

## Stage 1 — Retrieval and answer quality ✅ done

**Goal:** stop returning plausible-but-wrong chunks, and never present an
unsupported figure as fact.

Built:
- `rewrite.py` — abbreviation expansion (`PED` → pre-existing disease), policy
  detection from aliases, optional LLM rewrite into standalone follow-up queries.
- `search.search_multi` — every rewritten query is retrieved, rankings fused with RRF.
- `rerank.py` — cross-encoder over the 50 fused candidates, injectable scorer,
  graceful fallback to retrieval order if scoring fails.
- `citations.py` — citation numbers validated against the sources, and every
  amount/percentage/duration in the answer must exist in a source. Indian
  shorthand is normalised (`5 lakh` == `5,00,000`).
- `memory.py` — bounded transcript for follow-ups.
- `pipeline.ask` — one call returns answer, numbered sources with page ranges,
  rewritten queries, grounding verdict, token usage and timings.

Measured (10 answerable questions, k=10):

| mode | hit@1 | hit@3 | hit@5 | hit@10 | MRR |
| --- | --- | --- | --- | --- | --- |
| vector | 0.50 | 0.90 | 1.00 | 1.00 | 0.71 |
| keyword | 0.70 | 0.90 | 0.90 | 1.00 | 0.80 |
| hybrid | **0.80** | 0.90 | 1.00 | 1.00 | **0.86** |
| hybrid + rerank | 0.70 | **1.00** | 1.00 | 1.00 | 0.83 |

**The negative result is part of the deliverable.** Two rerankers were measured
on the fused candidates (`eval/experiments/rerank_compare.py`):

| reranker | hit@1 | hit@3 | hit@5 | MRR |
| --- | --- | --- | --- | --- |
| none (hybrid) | **0.80** | 0.90 | 1.00 | **0.86** |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 0.70 | **1.00** | 1.00 | 0.83 |
| `BAAI/bge-reranker-base` | 0.70 | 0.80 | 0.90 | 0.77 |

Truncation was ruled out: 800/1200/2000/3000-char pairs all score hit@1 0.70 for
MiniLM. The likely explanation is that hybrid already ranks the target chunk first
for 8 of 10 questions, and the phrase-based gold labels only accept a chunk that
contains the expected wording — a reranker that prefers a *different* but equally
valid clause is scored as a miss. That is a limitation of rule-based relevance
labels, which Stage 4's answer-level metrics address.

So reranking stays available (`mode="rerank"`) but is **not the default**, and the
next step is an insurance-tuned reranker plus better labels, not a bigger model.

---

## Stage 2 — Document intelligence and metadata

**Goal:** make "which policy is this and where exactly did it come from" a
first-class, queryable fact.

Work:
- `data/policies.json` already holds insurer, product, doc type and aliases.
  Extend it by extracting UIN, policy name and effective date from the first
  pages instead of hand-maintaining it.
- Store `insurer`, `product`, `doc_type` in chunk metadata so Chroma can filter
  (`where={"insurer": ...}`) instead of relying on prompt instructions.
- Reject duplicate uploads (same file hash) and surface near-duplicate chunks.
- Every source must resolve to `document → section → page`.

Acceptance: filter retrieval by insurer and by doc type; a duplicate PDF is
reported, not indexed twice; tests cover the UIN/date extractor.

Why it matters: metadata filters are the cheapest precision win in RAG. Models
ignore prompt instructions like "only use Star Health"; `where` clauses do not.

---

## Stage 3 — Multi-policy comparison

**Goal:** the feature users actually want: "which of these four policies is
better for maternity?" answered as a table, not prose.

Work:
- Define a Pydantic schema: sum insured, room rent limit, co-pay, PED waiting
  period, initial waiting period, maternity wait and limit, newborn cover,
  daycare, ambulance, sub-limits, exclusions.
- For each policy: filter retrieval to that policy, retrieve the field-specific
  queries, then extract into the schema with an LLM, **with the source page per
  field**.
- `"not stated"` is a valid value and must be used instead of guessing.
- Render as a table plus a short narrative of the differences that matter.

Acceptance: 4 policies × N fields, every cell has a page citation, the grounding
check passes for every filled cell, and unfilled cells say "not stated".
Test with two policies whose wording genuinely differs (Star brochure vs ICICI
wording) to catch a model that copies one policy into the other's column.

---

## Stage 4 — Evaluation (RAGAS + retrieval)

**Goal:** replace "it looks good" with the five numbers the project claims.

Work:
- Grow the golden set to ~30 questions: answerable, multi-policy, comparison,
  and unanswerable, each with `must_contain` phrases and a reference answer.
- `eval/eval_ragas.py`: faithfulness, answer relevancy, context precision,
  context recall over the golden set, run per configuration
  (`hybrid` vs `rerank`, `k=3` vs `k=5`).
- Store `eval/results/ragas_<config>.json` and a generated `eval/report.md`
  table so the README numbers are reproducible, not remembered.

Acceptance: one command regenerates every number in the README; the report says
which judge model and how many samples were used.

Watch out: RAGAS uses an LLM as judge, so it costs tokens and is noisy on small
sets. Run it on a fixed sample, keep the judge model fixed, and never tune on the
same questions you report.

---

## Stage 5 — FastAPI backend

**Goal:** the RAG core becomes a service, not a script.

Work:
- `POST /ask` (question, history, policy filter, k, mode) → answer, sources,
  grounding, timings.
- `POST /compare` → comparison table.
- `GET /documents`, `POST /documents` (upload + ingest validation),
  `DELETE /documents/{id}`, `GET /health`.
- Pydantic request/response models; sessions for conversation memory.
- Timeouts, clear 4xx/5xx, OpenAPI examples, CORS for Streamlit.

Acceptance: `pytest` covers the endpoints with `TestClient` (LLM faked), the
server runs with `uvicorn`, and README has copy-pasteable curl examples.

Why it matters: interviewers ask "how do you serve this". A clean API boundary
with typed contracts is the answer.

---

## Stage 6 — Streamlit frontend

**Goal:** a demo a non-technical recruiter can use in 30 seconds.

Work: multi-PDF upload with ingest status, document library, chat with page-level
citations, comparison view with evidence per cell, a grounding badge
(verified / unsupported figures listed / refused), settings for k and mode,
and the retrieval-eval table on an "Evaluation" tab.

Acceptance: a recorded demo (GIF in the README) showing upload → question →
citation → comparison.

Design rule: show uncertainty. A visible "this figure is not in the sources"
state is more impressive than a confident wrong answer.

---

## Stage 7 — Production hardening and packaging

**Goal:** someone else can run it in one command, and CI proves it.

Work:
- `pydantic-settings` for config, structured logging, retry/backoff around Groq,
  explicit timeouts.
- `Dockerfile` (multi-stage, non-root) + `docker-compose.yml` (api + ui),
  `.dockerignore`, persistent volume for `data/`.
- `Makefile` (`make ingest`, `make api`, `make ui`, `make eval`, `make test`)
  and `ruff` + `pre-commit`.
- GitHub Actions: lint + tests on push; cache the HF models.
- Input safety: file type/size checks, path traversal protection, no secrets in
  images, rate limiting on the public demo.

Acceptance: fresh clone → `.env` → `docker compose up` works; CI is green;
`data/` survives container restarts.

---

## Stage 8 — Portfolio polish

- README: problem, architecture diagram, metrics table, quickstart, screenshots,
  limitations, roadmap link.
- `docs/DECISIONS.md`: short ADRs — why hybrid, why RRF, why the reranker is not
  default, why section-aware chunking, why refuse instead of answer.
- A short write-up (blog/LinkedIn) built from Stage 1's before/after numbers.

---

## Interview mapping

| If they ask… | Point at |
| --- | --- |
| "How do you know retrieval works?" | `eval_retrieval.py`, `eval/results/*.json`, the mode comparison table |
| "Why hybrid search?" | vector misses exact figures, BM25 misses paraphrase; RRF needs no score calibration |
| "How do you stop hallucinations?" | grounding prompt + `citations.py` figure/citation verification + refusal path + RAGAS faithfulness |
| "How is it served?" | FastAPI stage, typed contracts, sessions, Docker |
| "What would you do with more time?" | the negative reranker result, an insurance-tuned reranker, incremental indexing, auth |

---

## Definition of done (every stage)

- [ ] acceptance criteria met and written down
- [ ] tests added for the tricky logic, full suite green
- [ ] before/after numbers recorded when quality is involved
- [ ] README and this roadmap updated
- [ ] one commit whose message explains the why
