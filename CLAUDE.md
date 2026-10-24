# LLM-BENCH (tps.sh) - local LLM tokens-per-second benchmark harness + dashboard

## What it is
Python 3.12+ CLI (anthropic, typer, rich) in `llm_bench/` plus a React dashboard in `dashboard/` and static pages in `site/` and `docs/`. Public product name tps.sh. GitHub: parsamivehchi/tps.sh (public).

## Current state (as of 2026-10-22)
- Status: Complete.
- Live at tps.sh (Cloudflare Pages project tps-sh); mivehchi.app rewrites /tps to it. Canonical URL per site/tps/index.html (rel=canonical) is https://mivehchi.app/tps.
- Corpus: 147 benchmarks, 7 models, $3.95 total cost (date of these figures unknown, unverified 2026-10-22).
- Deploy target conflict in repo history: commit 59d19f6 (2026-08-30) says Cloudflare Pages; README.md line 188 still says `docs/` is a "GitHub Pages site". No wrangler config or ship.json exists to settle it (unverified 2026-10-22).

## Run, gate, deploy
- Python: `.venv/bin/pip install -r requirements.txt`, then `.venv/bin/python -m llm_bench.cli`.
- Other entry points at repo root: `overnight_run.py`, `generate_comparison_pptx.py`.
- Dashboard: `dashboard/package.json` scripts `dev`, `build` (`tsc -b && vite build`), `lint`; `bun.lock` present. Named URL `https://llmbench.localhost` (portless name `llmbench` in `dashboard/portless.json`, registered in `~/.claude/reference/localhost-registry.json`).
- Deploy command and probe: none recorded in the repo (unverified 2026-10-22).

## Architecture map
- `llm_bench/`: cli, config, models, runner, judge, analysis, hardware, export, reports, prompts.
- `dashboard/`: React + Vite + recharts dashboard; data in `dashboard/public/data/dashboard_data.json`.
- `site/tps/`: static product page. `docs/`: comparison pages and screenshots.

## Invariants
- `.pii-scanignore` exempts the benchmark corpus (`dashboard/public/data/dashboard_data.json`, `llm_bench/prompts/bank/`) because raw LLM outputs contain example emails and IPs used to test regex validators (commit 9e059b3). Do not re-litigate in a public-demo scan; the denylist still applies.

## Project conventions
Follow [universal conventions](~/.claude/conventions/universal.md).

## Gotchas
- README.md line 188 still labels `docs/` a "GitHub Pages site"; that label predates the Cloudflare Pages move in commit 59d19f6, so do not read it as the deploy target.

## Where history lives
`git log` (no CHANGELOG or PROGRESS file), README.md.
