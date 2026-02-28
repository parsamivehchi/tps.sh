"""Weighted rankings and per-category bests."""

import json
from llm_bench.config import SCORED_DIR, CATEGORY_LABELS


def compute_rankings(run_id: str) -> dict:
    """Compute rankings from analysis data."""
    analysis_file = SCORED_DIR / run_id / "analysis.json"
    if not analysis_file.exists():
        return {}

    analysis = json.loads(analysis_file.read_text())
    rankings = analysis.get("model_rankings", [])

    if not rankings:
        return {}

    has_scores = analysis.get("has_quality_scores", False)

    result = {
        "fastest_ttft": min(rankings, key=lambda x: x.get("ttft_ms", float("inf"))),
        "highest_tps": max(rankings, key=lambda x: x.get("tokens_per_sec", 0)),
        "lowest_cost": None,
        "best_quality": None,
        "best_value": None,
    }

    # Best cost (among models that have cost)
    paid = [r for r in rankings if r.get("cost_usd", 0) > 0]
    if paid:
        result["lowest_cost"] = min(paid, key=lambda x: x["cost_usd"])

    if has_scores:
        result["best_quality"] = max(rankings, key=lambda x: x.get("score_weighted", 0))

        # Best value = quality / cost (for paid) or quality * tps (for free)
        for r in rankings:
            if r.get("cost_usd", 0) > 0:
                r["_value"] = r.get("score_weighted", 0) / r["cost_usd"]
            else:
                r["_value"] = r.get("score_weighted", 0) * r.get("tokens_per_sec", 1)
        result["best_value"] = max(rankings, key=lambda x: x.get("_value", 0))

    # Per-category bests
    cat_breakdown = analysis.get("category_breakdown", [])
    category_bests = {}
    for cat_id, label in CATEGORY_LABELS.items():
        cat_results = [r for r in cat_breakdown if r.get("category") == cat_id]
        if cat_results and has_scores:
            best = max(cat_results, key=lambda x: x.get("score_weighted", 0))
            category_bests[cat_id] = {
                "label": label,
                "best_model": best.get("model_name"),
                "score": best.get("score_weighted"),
            }

    result["category_bests"] = category_bests

    # Save
    rankings_file = SCORED_DIR / run_id / "rankings.json"
    rankings_file.write_text(json.dumps(result, indent=2, default=str))

    return result
