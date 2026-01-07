

test:
    TESTING=1 uv run pytest .

lint:
    uv run ruff check easysort/

pr-ready:
    just test
    just lint