# LLM-BENCH (tps.sh)

## Stack
Python 3.12+ CLI (anthropic, typer, rich) + React dashboard

## Status
Complete - live at tps.sh (Cloudflare Pages project tps-sh)

## Dev Commands
- `.venv/bin/pip install -r requirements.txt`
- `.venv/bin/python -m llm_bench.cli`

## Conventions
Follow [universal conventions](~/.claude/conventions/universal.md).

## Notes
- 147 benchmarks, 7 models, $3.95 total cost
- Deployed on Cloudflare Pages (project tps-sh); mivehchi.app rewrites /tps to it
