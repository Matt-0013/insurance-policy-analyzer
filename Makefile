.PHONY: install ingest api ui test eval compare

install:
	uv sync

ingest:
	uv run python -m insurance_policy_analyzer.ingest

api:
	uv run uvicorn insurance_policy_analyzer.api:app --reload --port 8000

ui:
	uv run streamlit run app.py

test:
	uv run pytest -q

eval:
	uv run python -m insurance_policy_analyzer.eval_retrieval

compare:
	uv run python -m insurance_policy_analyzer.render
