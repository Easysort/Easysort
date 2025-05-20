<div align="center">

![EasySort](docs/assets/easysort-logo.png)

Easysort: An open-source waste sorting system. Maintained by [EasySort](https://github.com/Easysort).

[Website](https://easysort.org) • [Documentation](https://docs.easysort.org) • [Discord](https://discord.com/invite/2Bh3SBFbdP)
</div>

---

![Lint](https://github.com/easysort/easysort/workflows/Lint/badge.svg)

To clone: ```git clone --recurse-submodules https://github.com/EasySort/easysort```

# What is EasySort
EasySort makes you go from a sorting problem to a solution in just a couple of hours. Collect data, annotate it, train a model and you are ready to go.
Our annotation tool we use at EasySort is also open-source and called [EasyLabeler](https://github.com/EasySort/easylabeler).

# How to setup EasySort

We used the following tools. Make sure you have all of them installed.

- [uv](https://github.com/astral-sh/uv)
- [just](https://github.com/casey/just)

Clone our repo:

```bash
git clone --recurse-submodules https://github.com/EasySort/easysort
```

Setup uv and run our tests:

```bash
# This will install all the dependencies and setup the virtual environment
uv sync
# Run the tests to make sure everything is working
just test
```

Notice: If you wish to run any specific file, use `uv run <file>` as opposed to `python <file>`.

You can start the UI and API locally by running:

```bash
just dev
```

# Why open-source?
I believe this is a positive-sum game. The market is so large that we can all win. This technology should be so cheap that it exists everywhere on earth helping everyone better sort items and improve the average quality of products running through their system.

You can use EasySort as you like, copy it, modify it, sell it, do whatever you like with it. The project is in active development, so issues and pull requests are welcome.
