test:
    uv run pytest

format:
    uv run ruff format

lint:
    uv run pre-commit run --all-files

dev:
    DEBUG=1 uv run easysort/sorting/pipeline.py
