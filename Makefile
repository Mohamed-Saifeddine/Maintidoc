.PHONY: setup run index test lint qdrant

setup:
	uv sync --extra dev

run:
	uv run streamlit run app.py

index:
	uv run maintidoc-index data/source/unicef_bulgaria_annual_report_2024.pdf

test:
	uv run pytest

lint:
	uv run ruff check .

qdrant:
	docker compose up -d qdrant

