.PHONY: install dev test lint format typecheck clean docker-test demo api e2e

install:
	python -m pip install -e .

dev:
	python -m pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

lint:
	python -m black --check src/ tests/ main.py scripts/
	python -m isort --check-only src/ tests/ main.py scripts/

format:
	python -m black src/ tests/ main.py scripts/
	python -m isort src/ tests/ main.py scripts/

typecheck:
	python -m mypy --config-file pyproject.toml

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml dist

demo:
	python main.py

api:
	python scripts/run_api.py

docker-test:
	docker compose --profile test run --rm test

e2e:
	pytest e2e/ -v --tb=short
