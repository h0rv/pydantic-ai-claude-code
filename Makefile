.PHONY: install format lint typecheck test testcov all
install:
	uv sync --frozen
	uv run pre-commit install
format:
	uv run ruff format
	uv run ruff check --fix --fix-only
lint:
	uv run ruff format --check
	uv run ruff check
typecheck:
	PYRIGHT_PYTHON_IGNORE_WARNINGS=1 uv run pyright
test:
	uv run pytest -n auto --dist=loadgroup --durations=20
testcov:
	uv run coverage run -m pytest
	uv run coverage report
all: lint typecheck testcov
