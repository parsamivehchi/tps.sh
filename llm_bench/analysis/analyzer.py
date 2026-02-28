"""Pandas aggregation and comparative summary."""

import json
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from llm_bench.config import SCORED_DIR, RESULTS_DIR, CATEGORY_LABELS

console = Console()


def _load_scored(run_id: str) -> pd.DataFrame:
    """Load scored results into a DataFrame."""
    scored_file = SCORED_DIR / run_id / "scored_results.json"
    if not scored_file.exists():
        # Fall back to raw results
        results_file = RESULTS_DIR / run_id / "results.json"
        if not results_file.exists():
            raise FileNotFoundError(f"No results found for run {run_id}")
        data = json.loads(results_file.read_text())
    else:
        data = json.loads(scored_file.read_text())

    df = pd.DataFrame(data)

    # Expand scores dict into columns
    if "scores" in df.columns:
        scores_df = df["scores"].apply(
            lambda x: pd.Series(x) if isinstance(x, dict) else pd.Series()
        )
        scores_df.columns = [f"score_{c}" for c in scores_df.columns]
        df = pd.concat([df.drop(columns=["scores"]), scores_df], axis=1)

    return df


def analyze_run(run_id: str) -> dict:
    """Generate analysis and print summary. Returns analysis dict."""
    df = _load_scored(run_id)

    has_scores = "score_weighted" in df.columns and df["score_weighted"].notna().any()

    # === Overall model rankings ===
    agg_cols = {
        "ttft_ms": "mean",
        "tokens_per_sec": "mean",
        "total_time_ms": "mean",
        "output_tokens": "mean",
        "cost_usd": "sum",
    }
    if has_scores:
        agg_cols["score_weighted"] = "mean"
        agg_cols["score_correctness"] = "mean"
        agg_cols["score_completeness"] = "mean"
        agg_cols["score_clarity"] = "mean"

    model_summary = df.groupby("model_name").agg(agg_cols).round(2)

    # === Per-category breakdown ===
    cat_agg = {"tokens_per_sec": "mean", "total_time_ms": "mean"}
    if has_scores:
        cat_agg["score_weighted"] = "mean"
    category_summary = df.groupby(["category", "model_name"]).agg(cat_agg).round(2)

    # === Print tables ===
    console.print(f"\n[bold cyan]Analysis for run {run_id}[/]\n")

    # Model ranking table
    table = Table(title="Overall Model Rankings")
    table.add_column("Model", style="cyan")
    table.add_column("Avg TTFT (ms)", justify="right")
    table.add_column("Avg TPS", justify="right")
    table.add_column("Avg Time (ms)", justify="right")
    table.add_column("Total Cost", justify="right")
    if has_scores:
        table.add_column("Quality", justify="right", style="green")

    # Sort by quality if available, else by TPS
    sort_col = "score_weighted" if has_scores else "tokens_per_sec"
    sorted_models = model_summary.sort_values(sort_col, ascending=(not has_scores))

    for model_name, row in sorted_models.iterrows():
        cols = [
            str(model_name),
            f"{row['ttft_ms']:.0f}",
            f"{row['tokens_per_sec']:.1f}",
            f"{row['total_time_ms']:.0f}",
            f"${row['cost_usd']:.4f}" if row['cost_usd'] > 0 else "free",
        ]
        if has_scores:
            cols.append(f"{row['score_weighted']:.2f}/10")
        table.add_row(*cols)

    console.print(table)

    # Per-category table
    if has_scores:
        cat_table = Table(title="\nPer-Category Quality Scores")
        cat_table.add_column("Category", style="cyan")
        cat_table.add_column("Model", style="white")
        cat_table.add_column("Quality", justify="right", style="green")
        cat_table.add_column("TPS", justify="right")

        for (cat, model), row in category_summary.iterrows():
            label = CATEGORY_LABELS.get(cat, cat)
            cat_table.add_row(label, str(model), f"{row['score_weighted']:.2f}", f"{row['tokens_per_sec']:.1f}")

        console.print(cat_table)

    # Build analysis dict for export
    analysis = {
        "run_id": run_id,
        "model_rankings": model_summary.reset_index().to_dict(orient="records"),
        "category_breakdown": category_summary.reset_index().to_dict(orient="records"),
        "has_quality_scores": has_scores,
    }

    # Save analysis
    analysis_dir = SCORED_DIR / run_id
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "analysis.json").write_text(json.dumps(analysis, indent=2, default=str))
    console.print(f"\n[green]Analysis saved to {analysis_dir / 'analysis.json'}[/]")

    return analysis
