# Presentations

This subrepo is used to store and create presentations for the Easysort project using slidev. More information about slidev can be found in the [skills](skills/slidev.md) subrepo.

## Updates

Dated updates on the Easysort project/company.

This is always the most up to date information. More up to date than the website and the docs. Whenever using any of these, you should verify the information with the user.

## Examples

Great examples of good use of the slidev framework, which generated beautiful presentations.

## presentations

The actual presentations for the Easysort project for previous and current presentations.

## requirements

Requirements for the presentations.
The requirements file and the presentation file is called the same name, so it's easy to match.

The requirements file should contain instructions on what should be in the presentation, what the user/audience expects, what you are expected to include, etc.

## How to run

First look in debug mode in the web while developing:

```bash
pnpm slidev presentations/<filename>.md
```

Then export to PDF:

```bash
pnpm slidev export presentations/<filename>.md
```