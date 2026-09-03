.PHONY: setup download prepare audit baseline source-probe test test-data lint format verify

setup:
	uv sync --all-groups

download:
	uv run claim-download

prepare: download
	uv run claim-prepare

audit: prepare
	uv run claim-audit

baseline: prepare
	uv run claim-baseline

source-probe: prepare
	uv run claim-source-probe

test:
	uv run pytest --cov=claim_detector --cov-report=term-missing

test-data: prepare
	uv run pytest -m real_data

lint:
	uv run ruff check .
	uv run mypy src

format:
	uv run ruff format .
	uv run ruff check --fix .

verify: lint test
