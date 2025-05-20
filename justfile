test:
    uv run pytest

lint:
    uv run pre-commit run --all-files

dev:
    uv run easysort/sorting/pipeline.py