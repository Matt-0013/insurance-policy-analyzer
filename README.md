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
