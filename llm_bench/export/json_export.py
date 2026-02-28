"""Export dashboard-ready JSON from scored results + analysis."""

import json
from pathlib import Path

from rich.console import Console

from llm_bench.config import SCORED_DIR, RESULTS_DIR, EXPORTS_DIR, DASHBOARD_DATA_DIR, MODELS, CATEGORY_LABELS
from llm_bench.analysis.rankings import compute_rankings

console = Console()


def export_dashboard_json(run_id: str) -> Path:
    """Generate dashboard_data.json for the React frontend."""
    # Load scored results
    scored_file = SCORED_DIR / run_id / "scored_results.json"
    if scored_file.exists():
        results = json.loads(scored_file.read_text())
    else:
        results_file = RESULTS_DIR / run_id / "results.json"
        results = json.loads(results_file.read_text())

    # Load metadata
    meta_file = RESULTS_DIR / run_id / "meta.json"
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}

    # Compute rankings
    rankings = compute_rankings(run_id)

    # Build model info
    models_info = {}
    for m in MODELS:
        models_info[m.name] = {
            "name": m.name,
            "provider": m.provider,
            "model_type": m.model_type,
            "cost_input": m.cost_input,
            "cost_output": m.cost_output,
        }

    # Build dashboard data
    dashboard_data = {
        "meta": {
            "run_id": run_id,
            "started_at": meta.get("started_at"),
            "completed_at": meta.get("completed_at"),
            "total_tests": meta.get("total_tests"),
            "model_count": len(meta.get("models", [])),
        },
        "models": models_info,
        "categories": CATEGORY_LABELS,
        "results": results,
        "rankings": rankings,
    }

    # Include hardware profile if available
    if meta.get("hardware_profile"):
        dashboard_data["hardware_profile"] = meta["hardware_profile"]

    # Save to exports dir
    export_file = EXPORTS_DIR / f"dashboard_data_{run_id}.json"
    export_file.write_text(json.dumps(dashboard_data, indent=2, default=str))

    # Also copy to dashboard public dir
    dashboard_file = DASHBOARD_DATA_DIR / "dashboard_data.json"
    dashboard_file.write_text(json.dumps(dashboard_data, indent=2, default=str))

    console.print(f"[green]Dashboard data exported:[/]")
    console.print(f"  {export_file}")
    console.print(f"  {dashboard_file}")

    return export_file
