alias t := test
alias f := format

test *PATHS:
    uv run pytest {{PATHS}}

format:
    uv run ruff format

lint:
    uv run pre-commit run --all-files

install-hooks:
    uv run pre-commit install

dev:
    DEBUG=1 uv run easysort/sorting/pipeline.py
