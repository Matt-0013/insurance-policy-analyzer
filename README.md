# Insurance Policy Analyzer — advanced RAG over Indian health policies

Ask questions across multiple Indian health insurance policy PDFs and get answers
with document- and page-level citations, or a refusal when the documents do not
contain the answer.

Built as an end-to-end GenAI project: PyMuPDF extraction, section-aware chunking,
local embeddings, hybrid retrieval, query rewriting, optional cross-encoder
reranking, grounded generation with Groq, and a deterministic hallucination check
on every answer.

**Status:** retrieval and answer quality are done and measured; comparison,
FastAPI, Streamlit, RAGAS and Docker are next — see [docs/ROADMAP.md](docs/ROADMAP.md).

---

## Results

Retrieval on the 10 answerable golden questions, k=10. A hit means the retrieved
chunk comes from the right policy *and* contains the expected wording.

| mode | hit@1 | hit@3 | hit@5 | hit@10 | MRR |
| --- | --- | --- | --- | --- | --- |
| vector (granite embeddings) | 0.50 | 0.90 | 1.00 | 1.00 | 0.71 |
| keyword (BM25) | 0.70 | 0.90 | 0.90 | 1.00 | 0.80 |
| **hybrid (RRF)** | **0.80** | 0.90 | 1.00 | 1.00 | **0.86** |
| hybrid + cross-encoder rerank | 0.70 | 1.00 | 1.00 | 1.00 | 0.83 |

Reproduce with `uv run python -m insurance_policy_analyzer.eval_retrieval hybrid`;
raw rows land in `eval/results/`.

Reranking is implemented and available (`mode="rerank"`) but **not the default**:
two independent rerankers (`ms-marco-MiniLM-L-6-v2` and `BAAI/bge-reranker-base`)
both lowered hit@1 on this corpus. The measurements and the reasoning are in
[the roadmap](docs/ROADMAP.md#stage-1--retrieval-and-answer-quality--done) and
`eval/experiments/rerank_compare.py`.

---

## How it works

```
PDF ─► extract ─► clean ─► chunk ─┬─► embed (granite, local) ─► Chroma
                                  └─► BM25 index
                                          │
question ─► rewrite ─► hybrid retrieve (RRF) ─► rerank (optional) ─► context
                                          │
                                     Groq gpt-oss-120b ─► answer + citations
                                          │
                             citation + figure verification (grounding)
```

- **Extraction** — PyMuPDF4LLM with classic → layout → plain fallbacks, gated on
  80% alphanumeric coverage so a bad extraction is retried instead of indexed.
- **Cleaning** — repairs letter-spaced PDF text (`T h i s` → `This`).
- **Chunking** — grouped under detected section headings, 1500-char cap, page
  ranges kept, table headers repeated when a table is split.
- **Embeddings** — `ibm-granite/granite-embedding-small-english-r2`, local CPU,
  normalised, stored in a persistent Chroma collection (cosine).
- **Hybrid retrieval** — vector + BM25 over the same `section + text`, fused with
  Reciprocal Rank Fusion, so exact figures (`2,50,000`) and paraphrases both work.
- **Query rewriting** — expands insurance shorthand (`PED` → pre-existing
  disease), detects a named policy from aliases, and optionally uses a small model
  to turn a follow-up into standalone queries. Rewrites are *added* to the
  original question, never replace it, and the original is weighted higher in the
  fusion.
- **Metadata filtering** — a question that names one policy is filtered to that
  policy in Chroma/BM25. The policy name is then stripped from the search text:
  every chunk carries its policy name in the embedded prefix, so leaving it in a
  short query drowns out the discriminative terms.
- **Grounding** — after generation, every citation number is checked against the
  sources, and every amount/percentage/duration in the answer must appear in a
  source. Indian shorthand is normalised, so `5 lakh` matches `5,00,000`. The
  verdict is returned alongside the answer (`grounding.grounded`,
  `unsupported_numbers`).
- **Conversation memory** — bounded transcript (last 3 turns) used for follow-up
  rewriting and generation; the policy filter is carried forward deterministically.
- **Refusal** — if retrieval returns nothing, the answer is the fixed refusal
  string, with no LLM call at all.

---

## Quickstart

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync

# 1. API key (free): https://console.groq.com/keys
cp .env.example .env      # then put the key in GROQ_API_KEY

# 2. Add policy PDFs — sources and download links in data/SOURCES.md
#    data/raw/*.pdf

# 3. Build the index (extract → clean → chunk → embed)
uv run python -m insurance_policy_analyzer.extract
uv run python -m insurance_policy_analyzer.chunk
uv run python -m insurance_policy_analyzer.index

# 4. Ask
uv run python -m insurance_policy_analyzer.answer "What is the co-payment in Niva Bupa Senior First?"
```

Example output:

```
**Co-payment in Niva Bupa Senior First**
- The default co-payment is 50 % of the admissible claim amount, and it can be
  changed at the start of the first policy to 40 %, 30 %, or 20 % [1].

Sources:
  [1] Niva Bupa - Senior First (policy wording) | 4. Claim Cost Sharing | p.9
Tokens: 1284 prompt + 178 completion = 1462
Timing: 3362 ms retrieval, 4385 ms total
```

Other entry points:

```bash
uv run python -m insurance_policy_analyzer.search hybrid "ambulance charges by road"
uv run python -m insurance_policy_analyzer.eval_retrieval rerank
uv run pytest
```

---

## Project layout

```
src/insurance_policy_analyzer/
  config.py          paths, models, tunables
  extract.py         PDF → per-page text with method + coverage
  clean.py           repair letter-spaced text
  chunk.py           section-aware chunking
  index.py           embed chunks into Chroma
  search.py          vector / keyword / hybrid / rerank, RRF, multi-query
  rerank.py          cross-encoder over fused candidates
  rewrite.py         abbreviation expansion, policy detection, LLM rewriting
  registry.py        insurer / product / aliases, policy filters
  prompts.py         grounding, answer and rewrite prompts
  citations.py       citation parsing + figure verification
  memory.py          bounded conversation transcript
  llm.py             shared Groq client
  pipeline.py        ask(): the end-to-end path used by API and UI
  answer.py          CLI over pipeline.ask
  eval_retrieval.py  hit@k / MRR over eval/questions.jsonl
eval/
  questions.jsonl    12 golden questions (10 answerable, 2 unanswerable)
  results/           one JSON per configuration
  experiments/       reranker comparison
docs/ROADMAP.md      staged plan to production
```

---

## Evaluation setup

`eval/questions.jsonl` holds 12 questions across the four policies: sum insured,
room rent, co-pay, PED and specific-disease waiting periods, maternity, ambulance,
named ailments, and two questions deliberately not answered by the corpus. Each
row carries `must_contain` phrases that must appear in a retrieved chunk, so the
label is checkable rather than a judgement call.

Known limitation: phrase-based labels only credit a chunk that contains the
expected wording. A chunk that answers the question differently is scored as a
miss — part of why the rerankers look worse than they may be. Answer-level
metrics (RAGAS) are on the roadmap.

---

## Limitations

- The index is rebuilt from scratch on ingest; no incremental add/delete yet.
- One corpus per deployment; no per-user isolation or auth.
- Groq free tier limits request rate and tokens per minute.
- The reranker is available but off by default (measured, not assumed).
- The API and UI do not exist yet; this is a library + CLI today.

---

## Roadmap

[docs/ROADMAP.md](docs/ROADMAP.md) — multi-policy comparison, RAGAS evaluation,
FastAPI, Streamlit, Docker/CI, and the portfolio write-up, with acceptance
criteria and the reasoning for each stage.

---

## Comparison engine

`python -m insurance_policy_analyzer.render` prints a 12 field x 4 policy table
where every cell carries a page number and a state.

| state | meaning |
| --- | --- |
| `stated` | the clause is in the document; the cell quotes it |
| `excluded` | the document carves the item out (often with an IRDAI `Excl` code) |
| `scheduled` | the wording defers the number to the Policy Schedule, which is a separate document |
| `not stated` | every alias was tried against a complete document and the concept is absent |

**Why `scheduled` exists.** Policy *wordings* describe mechanisms and route every
quantity to the schedule issued with the individual policy; *brochures* print the
numbers because they are selling. In this corpus ICICI's wording runs to 205,113
characters and contains zero occurrences of "per day" or "% of Sum Insured",
while Star Health's brochure carries the actual slab table
(`|5,00,000/-|Upto 1% of Sum Insured per day|`). Writing "not stated" for those
cells would be false; writing a number would be invented.

**How a cell is decided.** A field is mentioned many times per document: in a
definition, in a benefit clause, in an exclusion list, in a consumables price
list. Candidates are therefore scored (clean +2, definition +2, schedule
reference +1, quantity +1, comma list -1) and the state comes from the winner.
Polarity is read narrowest-first: IRDAI code within +/-150 characters, then the
owning sentence, then the same-chunk preamble, then only strong markers
("shall not be liable") in the previous chunk, then an `Exclusions` heading.

Every weight and every evidence window, with the corpus clause that justifies
it, is generated into [docs/SCORING.md](docs/SCORING.md). The table is produced
from the code by `python -m insurance_policy_analyzer.compare --table`, and a
test fails if the document and the code disagree.

**Known limitations, stated rather than hidden**

- ICICI and Niva Bupa `maternity` are `excluded` through `Code-Excl18` in the
  section heading; Star Health is `stated` because a benefit sentence grants two
  deliveries. That is the rule: a code in the heading beats an incidental mention,
  an explicit cover grant beats the code, and a coded heading that still states a
  waiting period ("Code-Excl01 ... until 36 months of continuous coverage") is a
  waiting period, not a carve-out.
- Cells quote the clause instead of extracting normalised values ("36 months",
  "1% of Sum Insured") — value extraction is the next step.
- Care Health `ambulance` is `not stated`: the only mentions are a consumables
  price list, not a benefit clause.

## API

```bash
uv run uvicorn insurance_policy_analyzer.api:app --reload --port 8000
```

| endpoint | purpose |
| --- | --- |
| `GET /health` | liveness, policy count, available retrieval modes |
| `GET /documents` | indexed policies with insurer, product, doc type, aliases |
| `POST /ask` | question + optional history/policy/mode/k -> answer, sources, grounding, timings |
| `POST /compare` | optional policy subset -> per-field cells with state, quote, page, plus markdown |

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/ask -H "content-type: application/json" \
  -d '{"question":"What is the waiting period for pre-existing diseases?","mode":"hybrid"}'
curl -s -X POST localhost:8000/compare -H "content-type: application/json" \
  -d '{"policies":["niva_bupa_senior_first_wording","star_health_women_care_brochure"]}'
```

The retrieval stack (torch, chromadb, the embedding model) is imported inside the
endpoint, so `/health` and the OpenAPI schema answer without loading a model.

## UI

```bash
uv run streamlit run app.py
```

Ask tab shows the answer, a grounding badge and expandable sources with page
numbers; Compare tab renders the evidence table; Documents tab lists what is
indexed; Evaluation tab shows whatever is in `eval/results/`.

## Packaging

```bash
export GROQ_API_KEY=...      # PowerShell: $env:GROQ_API_KEY="..."
docker compose up --build    # api on :8000, ui on :8501, data/ mounted from the host
```

Multi-stage build (uv resolves dependencies in the builder), non-root runtime,
`.dockerignore` keeps `.git`, `.venv`, `eval/` and `.env` out of the image.
`make ingest|api|ui|test|eval|compare` wraps the common commands.

## Answer quality (RAGAS)

```bash
uv run python -m insurance_policy_analyzer.eval_ragas --limit 5 --k 3
# -> eval/results/ragas_hybrid.json and eval/report.md
```

Run on 5 answerable questions, judge `openai/gpt-oss-20b` via Groq's
OpenAI-compatible endpoint (no OpenAI key):

| metric | score | what it measures |
| --- | --- | --- |
| faithfulness | 0.53 | are the answer's claims supported by the retrieved clauses |
| answer relevancy | 0.93 | does the answer address the question asked |
| context precision | 0.97 | did retrieval bring clauses that matter |
| refusal on unanswerable | 1.00 | 2 of 2 refused instead of guessing |
| grounded, rule-based (`citations.py`) | 1.00 (post-fix) | every figure and citation appears in the sources |

Read that table honestly:

- **n = 5, with one judge failure** (recorded as null, excluded from the mean).
  This is a smoke-scale measurement, not a benchmark.
- **The judge and the rule-based checker disagree**, and they measure different
  things: `citations.py` checks that every figure appears verbatim in a source,
  RAGAS faithfulness checks whether each claim is entailed by the context. An
  answer can pass one and fail the other. Both numbers are published for that
  reason; a single "accuracy" number would hide the disagreement.
- **The free tier caps at 8k tokens/minute**, so the run uses one worker, `k=3`
  and a 20b judge. The 120b judge returns 429s; a paid tier would let the sample
  grow and the judge be stronger.
- **Context recall is not reported**: it needs a reference answer per question,
- **The 0.40 above this line was a bug, not a hallucination rate.** gpt-oss emits
  fullwidth citation brackets (`【1】`) instead of `[1]`, so `parse_citations`
  found no citations and marked correct answers ungrounded. After normalising the
  brackets, all 5 answers verify: rule-based grounding is **1.00**. RAGAS faithfulness
  was never affected, because the judge reads the text, not the bracket syntax.
  and writing references after seeing the output would measure the references.
