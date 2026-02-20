.PHONY: install dev test lint format typecheck clean docker-test demo api

install:
	python -m pip install -e .

dev:
	python -m pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

lint:
	python -m black --check src/ tests/ main.py
	python -m isort --check-only src/ tests/ main.py

format:
	python -m black src/ tests/ main.py
	python -m isort src/ tests/ main.py

typecheck:
	python -m mypy src/ultrasound --ignore-missing-imports

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml

demo:
	python main.py

docker-test:
	docker compose run --rm test

api:
	python scripts/run_api.py
