

test:
    TESTING=1 uv run pytest .

lint:
    uv run ruff check easysort/

healthy:
    uv run easyprod/check_health.py

pull-all:
    git pull
    cd easyprod
    git pull
    cd ..

pr-ready:
    just test
    just lint