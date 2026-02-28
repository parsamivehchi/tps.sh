"""Typer CLI for LLM-Bench."""

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from llm_bench.config import MODELS, CATEGORIES, CATEGORY_LABELS, RESULTS_DIR, SCORED_DIR, EXPORTS_DIR

app = typer.Typer(name="llm-bench", help="LLM Benchmarking Toolkit — Local vs Cloud")
console = Console()


def _get_run_dir(run_id: str) -> Path:
    run_dir = RESULTS_DIR / run_id
    if not run_dir.exists():
        console.print(f"[red]Run not found: {run_id}[/]")
        raise typer.Exit(1)
    return run_dir


@app.command()
def run(
    model: Optional[list[str]] = typer.Option(None, "--model", "-m", help="Model name(s) to test"),
    category: Optional[list[str]] = typer.Option(None, "--category", "-c", help="Category(ies) to test"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Skip cache for cloud results"),
    hardware_metrics: bool = typer.Option(False, "--hardware-metrics", help="Capture GPU/thermal/power metrics during runs"),
):
    """Run benchmarks against selected models and categories."""
    from dotenv import load_dotenv
    load_dotenv()

    from llm_bench.runner.executor import run_benchmark
    asyncio.run(run_benchmark(
        model_names=model or None,
        categories=category or None,
        use_cache=not no_cache,
        hardware_metrics=hardware_metrics,
    ))


@app.command()
def judge(run_id: str = typer.Argument(..., help="Run ID to judge")):
    """Score all outputs using Claude Sonnet as judge."""
    from dotenv import load_dotenv
    load_dotenv()

    from llm_bench.judge.scorer import judge_run
    asyncio.run(judge_run(run_id))


@app.command()
def analyze(run_id: str = typer.Argument(..., help="Run ID to analyze")):
    """Generate analysis and rankings from scored results."""
    from llm_bench.analysis.analyzer import analyze_run
    analyze_run(run_id)


@app.command()
def report(run_id: str = typer.Argument(..., help="Run ID to generate reports for")):
    """Generate Word doc and PowerPoint reports."""
    from llm_bench.reports.word_report import generate_word_report
    from llm_bench.reports.pptx_report import generate_pptx_report
    generate_word_report(run_id)
    generate_pptx_report(run_id)


@app.command()
def export(run_id: str = typer.Argument(..., help="Run ID to export")):
    """Export dashboard-ready JSON."""
    from llm_bench.export.json_export import export_dashboard_json
    export_dashboard_json(run_id)


@app.command(name="hardware-report")
def hardware_report():
    """Generate hardware analysis Word doc and PowerPoint."""
    from llm_bench.reports.hardware_report import generate_hardware_report
    from llm_bench.reports.hardware_pptx import generate_hardware_pptx
    generate_hardware_report()
    generate_hardware_pptx()


@app.command(name="cost-estimate")
def cost_estimate():
    """Estimate cost for a full benchmark run."""
    from llm_bench.runner.metrics import estimate_run_cost
    from llm_bench.prompts.loader import load_all_prompts

    prompts = load_all_prompts()
    cloud_models = [m for m in MODELS if m.provider == "anthropic"]
    costs = estimate_run_cost(cloud_models, len(prompts))

    table = Table(title="Estimated Cost (Full Run)")
    table.add_column("Component", style="cyan")
    table.add_column("Cost", justify="right", style="green")

    for name, cost in costs.items():
        if name != "total":
            table.add_row(f"{name} ({len(prompts)} prompts)", f"${cost:.4f}")
    table.add_row("", "")
    table.add_row("[bold]Total benchmark cost[/]", f"[bold]${costs['total']:.4f}[/]")

    # Add judging estimate
    total_outputs = len(MODELS) * len(prompts)
    judge_cost = (total_outputs * 1150 * 3.0 / 1_000_000) + (total_outputs * 150 * 15.0 / 1_000_000)
    table.add_row(f"Quality judging ({total_outputs} outputs via Sonnet)", f"${judge_cost:.4f}")
    table.add_row("[bold]Grand total[/]", f"[bold]${costs['total'] + judge_cost:.4f}[/]")

    console.print(table)


@app.command(name="list-runs")
def list_runs():
    """List all benchmark runs."""
    if not RESULTS_DIR.exists():
        console.print("[yellow]No runs found.[/]")
        return

    table = Table(title="Benchmark Runs")
    table.add_column("Run ID", style="cyan")
    table.add_column("Models", justify="right")
    table.add_column("Tests", justify="right")
    table.add_column("Status", style="green")
    table.add_column("Cost", justify="right")

    for run_dir in sorted(RESULTS_DIR.iterdir(), reverse=True):
        meta_file = run_dir / "meta.json"
        if not meta_file.exists():
            continue
        meta = json.loads(meta_file.read_text())
        status = "complete" if "completed_at" in meta else "in-progress"
        results_file = run_dir / "results.json"
        cost = ""
        if results_file.exists():
            results = json.loads(results_file.read_text())
            total_cost = sum(r.get("cost_usd", 0) for r in results)
            cost = f"${total_cost:.4f}" if total_cost > 0 else "free"
        table.add_row(
            meta["run_id"],
            str(len(meta.get("models", []))),
            str(meta.get("total_tests", "?")),
            status,
            cost,
        )

    console.print(table)
