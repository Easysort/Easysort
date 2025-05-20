test:
    uv run pytest

lint:
    uv run pre-commit run --all-files

dev:
    DEBUG=1 uv run easysort/sorting/pipeline.py