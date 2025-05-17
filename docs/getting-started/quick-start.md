# Quick Start

We have spent and still spend a lot of time on making Easysort easy to use. In just 10 minutes you should be able to make simple changes to our code and run it on your own machine.

## 1. Setup your development environment

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

## 2. Run our models

You can start the UI and API locally by running:

```bash
just dev
```

## 3. Make changes to the code

Search for 

```bash
git grep "drawText" selfdrive/ui/qt/onroad/hud.cc
```

You will find this code inside `selfdrive/ui/qt/onroad/hud.cc``:

```cpp
void OnroadHud::drawText(QPainter &p, int x, int y, const QString &text, const QColor &color) {
  if (text.isEmpty()) {
    return;
  }
}
```

Make the following change:

```cpp
void OnroadHud::drawText(QPainter &p, int x, int y, const QString &text, const QColor &color) {
  if (text.isEmpty()) {
    return;
  }
}
```

This change will ...

## 4. View your changes

Again, start UI and API locally:

```bash
just dev
```