.PHONY: setup download prepare audit baseline train-bert train-bert-heldout train-bert-control calibrate serve benchmark docker-build docker-up docker-down source-probe test test-data lint format verify

setup:
	uv sync --all-extras --all-groups

download:
	uv run claim-download

prepare: download
	uv run claim-prepare

audit: prepare
	uv run claim-audit

baseline: prepare
	uv run claim-baseline

train-bert: prepare
	uv run claim-train-bert

train-bert-heldout: prepare
	uv run claim-train-bert-heldout

train-bert-control: prepare
	uv run claim-train-bert-control

calibrate:
	uv run claim-calibrate

serve:
	uv run claim-api

benchmark:
	uv run python scripts/benchmark_api.py --requests 100 --concurrency 1 --warmup 10 --deployment docker-cpu --output reports/generated/api_benchmark_c1.json
	uv run python scripts/benchmark_api.py --requests 500 --concurrency 4 --warmup 10 --deployment docker-cpu --output reports/generated/api_benchmark_c4.json

docker-build:
	PATH="/Applications/Docker.app/Contents/Resources/bin:$${PATH}" docker build --tag claim-detector:local .

docker-up:
	PATH="/Applications/Docker.app/Contents/Resources/bin:$${PATH}" docker compose up --build

docker-down:
	PATH="/Applications/Docker.app/Contents/Resources/bin:$${PATH}" docker compose down

source-probe: prepare
	uv run claim-source-probe

test:
	uv run pytest --cov=claim_detector --cov-report=term-missing

test-data: prepare
	uv run pytest -m real_data

lint:
	uv run ruff check .
	uv run mypy src scripts

format:
	uv run ruff format .
	uv run ruff check --fix .

verify: lint test
