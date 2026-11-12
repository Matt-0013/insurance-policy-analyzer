"""Streamlit demo: ask grounded questions and compare policies side by side.

Run with:  streamlit run app.py
The src/ path is added here so the app works without setting PYTHONPATH.
"""
import json
import sys
from pathlib import Path

import streamlit as st

SRC = Path(__file__).parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from insurance_policy_analyzer import registry  # noqa: E402
from insurance_policy_analyzer.config import CHUNKS_DIR, MODES  # noqa: E402
from insurance_policy_analyzer.pipeline import ask  # noqa: E402
from insurance_policy_analyzer.render import to_markdown  # noqa: E402

st.set_page_config(page_title="Insurance Policy Analyzer", page_icon="\U0001f6e1", layout="wide")

POLICY_IDS = registry.policy_ids()


def grounding_badge(result: dict) -> None:
    grounding = result["grounding"]
    if grounding.get("refused"):
        st.info("Refused: no relevant clause was retrieved, so nothing was guessed.")
        return
    if grounding["grounded"]:
        st.success("Grounded: every citation and figure was found in the retrieved clauses.")
        return
    st.warning("Not fully grounded — check these before trusting the answer:")
    for number in grounding["unsupported_numbers"]:
        st.write(f"- figure not in any source: **{number['raw']}** ({number['kind']})")
    for number in grounding["invalid_citations"]:
        st.write(f"- citation points at a missing source: **[{number}]**")
    if not grounding["citations"]:
        st.write("- the answer cites no source at all")


def render_sources(result: dict) -> None:
    with st.expander(f"Sources ({len(result['sources'])})", expanded=False):
        for source in result["sources"]:
            st.markdown(
                f"**[{source['n']}]** {source['policy_label']} — {source['section']} "
                f"(p.{source['page_start']})"
            )
            st.caption((source["text"] or "")[:600])


with st.sidebar:
    st.header("Settings")
    mode = st.selectbox("Retrieval mode", MODES, index=MODES.index("hybrid"))
    k = st.slider("Clauses to retrieve", 3, 10, 5)
    choice = st.selectbox(
        "Policy filter",
        ["All policies", *POLICY_IDS],
        format_func=lambda value: value if value == "All policies" else registry.display_name(value),
    )
    st.caption(
        "Hybrid = vector + BM25 fused with RRF. Reranking is available but measured "
        "worse on hit@1, so it is not the default."
    )

policy = None if choice == "All policies" else choice

st.title("Insurance Policy Analyzer")
st.caption("Grounded answers with page citations, and evidence-based multi-policy comparison.")

ask_tab, compare_tab, documents_tab, evaluation_tab = st.tabs(
    ["Ask", "Compare", "Documents", "Evaluation"]
)

with ask_tab:
    if "chat" not in st.session_state:
        st.session_state.chat = []
        st.session_state.history = []

    for message in st.session_state.chat:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("e.g. What is the waiting period for pre-existing diseases?")
    if prompt:
        st.session_state.chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving clauses and generating a grounded answer..."):
                try:
                    result = ask(
                        prompt,
                        k=k,
                        policy=policy,
                        history=st.session_state.history,
                        mode=mode,
                    )
                except Exception as exc:  # surface the failure instead of a blank page
                    st.error(f"Request failed: {exc}")
                    result = None
            if result:
                st.markdown(result["answer"])
                grounding_badge(result)
                render_sources(result)
                timings = result["timings_ms"]
                st.caption(
                    f"{timings.get('retrieval', 0)} ms retrieval · "
                    f"{timings.get('total', 0)} ms total · "
                    f"queries: {', '.join(result['retrieval']['queries'])}"
                )
                st.session_state.chat.append({"role": "assistant", "content": result["answer"]})
                st.session_state.history.append({"question": prompt, "answer": result["answer"]})

with compare_tab:
    st.subheader("Field-by-field comparison")
    st.caption(
        "Every cell carries its page. `set by the policy schedule` means the wording "
        "defers the number to a schedule that is not in this corpus — it is not the "
        "same as 'no limit'."
    )
    chosen = st.multiselect(
        "Policies to compare",
        POLICY_IDS,
        default=POLICY_IDS[:2],
        format_func=registry.display_name,
    )
    if st.button("Build comparison", type="primary") and chosen:
        try:
            st.markdown(to_markdown(chosen))
        except KeyError as exc:
            st.error(f"Unknown policy: {exc}")

with documents_tab:
    st.subheader("Indexed documents")
    rows = []
    for policy_id in POLICY_IDS:
        meta = registry.info(policy_id)
        path = CHUNKS_DIR / f"{policy_id}.json"
        chunks = len(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else 0
        rows.append(
            {
                "policy": registry.display_name(policy_id),
                "document type": meta["doc_type"],
                "chunks": chunks,
                "aliases": ", ".join(meta["aliases"]),
            }
        )
    st.table(rows)

with evaluation_tab:
    st.subheader("Measured, not claimed")
    results_dir = Path(__file__).parent / "eval" / "results"
    files = sorted(results_dir.glob("*.json")) if results_dir.exists() else []
    if not files:
        st.info("No evaluation results yet. Run the retrieval eval to populate eval/results/.")
    for path in files:
        st.markdown(f"**{path.name}**")
        st.json(json.loads(path.read_text(encoding="utf-8")))
