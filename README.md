<div align="center">

<h1>EasySort</h1>

EasySort: An open-source waste sorting system. Maintained by [EasySort](https://github.com/Easysort).
</div>

---

![Lint](https://github.com/easysort/easysort/workflows/Lint/badge.svg)

To clone: ```git clone --recurse-submodules https://github.com/EasySort/easysort```

# What is EasySort
EasySort makes you go from a sorting problem to a solution in just a couple of hours. Collect data, annotate it, train a model and you are ready to go.
Our annotation tool we use at EasySort is also open-source and called [EasyLabeler](https://github.com/EasySort/easylabeler).

# How to setup EasySort

The recommended way to setup EasySort is to install it in a virtual environment. The easiest way to do that is using [uv](https://github.com/astral-sh/uv?tab=readme-ov-file#installation):

```bash
# This will install all the dependencies and setup the virtual environment
uv sync
# Run the tests to make sure everything is working
uv run pytest
# Install git hooks
uv run pre-commit install
# Activate the virtual environment (or just prefix your commands with uv run)
source .venv/bin/activate
```

# Why open-source?
I believe this is a positive-sum game. The market is so large that we can all win. This technology should be so cheap that it exists everywhere on earth helping everyone better sort items and improve the average quality of products running through their system.

You can use EasySort as you like, copy it, modify it, sell it, do whatever you like with it. The project is in active development, so issues and pull requests are welcome.
