"""Polished matplotlib charts for embedding in Word/PPTX reports."""

import json
from pathlib import Path
from io import BytesIO
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from llm_bench.config import SCORED_DIR, RESULTS_DIR, CATEGORY_LABELS

# ── Consistent color palette ──
PALETTE = {
    "local":    ["#10b981", "#06b6d4", "#f59e0b", "#8b5cf6"],
    "cloud":    ["#3b82f6", "#6366f1", "#ec4899"],
    "dims":     ["#3b82f6", "#22c55e", "#f59e0b"],
    "accent":   "#0f172a",
    "grid":     "#e2e8f0",
    "bg":       "#ffffff",
    "text":     "#334155",
    "muted":    "#94a3b8",
}

def _model_color(name: str, idx: int) -> str:
    if "Claude" in name:
        return PALETTE["cloud"][idx % len(PALETTE["cloud"])]
    return PALETTE["local"][idx % len(PALETTE["local"])]


def _style_ax(ax, title: str, xlabel: str = "", ylabel: str = ""):
    ax.set_title(title, fontsize=13, fontweight="bold", color=PALETTE["accent"], pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color=PALETTE["text"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color=PALETTE["text"])
    ax.tick_params(colors=PALETTE["text"], labelsize=9)
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(PALETTE["grid"])
    ax.spines["bottom"].set_color(PALETTE["grid"])


def _save(fig) -> BytesIO:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight",
                facecolor=PALETTE["bg"], edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf


def _load_data(run_id: str) -> list[dict]:
    scored = SCORED_DIR / run_id / "scored_results.json"
    if scored.exists():
        return json.loads(scored.read_text())
    return json.loads((RESULTS_DIR / run_id / "results.json").read_text())


def _model_agg(data: list[dict], field: str) -> dict[str, float]:
    sums: dict[str, list[float]] = defaultdict(list)
    for r in data:
        val = r.get(field)
        if val is None and isinstance(r.get("scores"), dict):
            val = r["scores"].get(field.replace("score_", ""))
        if val is not None:
            sums[r["model_name"]].append(float(val))
    return {k: float(np.mean(v)) for k, v in sums.items()}


# ─────────────────── Charts ───────────────────


def speed_chart(run_id: str) -> BytesIO:
    """Horizontal bar: avg tokens/sec by model, sorted descending."""
    data = _load_data(run_id)
    agg = _model_agg(data, "tokens_per_sec")
    if not agg:
        return BytesIO()

    items = sorted(agg.items(), key=lambda x: x[1])
    models, values = zip(*items)

    fig, ax = plt.subplots(figsize=(9, max(3.5, len(models) * 0.7)))
    colors = [_model_color(m, i) for i, m in enumerate(models)]
    bars = ax.barh(range(len(models)), values, color=colors, height=0.6, zorder=3)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=10)
    _style_ax(ax, "Average Generation Speed", xlabel="Tokens / second")
    ax.bar_label(bars, fmt="%.1f tok/s", padding=6, fontsize=9, color=PALETTE["text"])
    ax.set_xlim(0, max(values) * 1.25)
    fig.tight_layout()
    return _save(fig)


def latency_chart(run_id: str) -> BytesIO:
    """Horizontal bar: avg TTFT by model, sorted ascending (fastest first)."""
    data = _load_data(run_id)
    agg = _model_agg(data, "ttft_ms")
    if not agg:
        return BytesIO()

    items = sorted(agg.items(), key=lambda x: -x[1])  # slowest at top
    models, values = zip(*items)

    fig, ax = plt.subplots(figsize=(9, max(3.5, len(models) * 0.7)))
    colors = [_model_color(m, i) for i, m in enumerate(models)]
    bars = ax.barh(range(len(models)), values, color=colors, height=0.6, zorder=3)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=10)
    _style_ax(ax, "Average Time to First Token (Latency)", xlabel="TTFT (ms)")
    ax.bar_label(bars, fmt="%.0f ms", padding=6, fontsize=9, color=PALETTE["text"])
    ax.set_xlim(0, max(values) * 1.2)
    fig.tight_layout()
    return _save(fig)


def quality_chart(run_id: str) -> BytesIO | None:
    """Grouped vertical bars: correctness/completeness/clarity per model."""
    data = _load_data(run_id)
    model_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in data:
        if isinstance(r.get("scores"), dict):
            s = r["scores"]
            for dim in ["correctness", "completeness", "clarity"]:
                if s.get(dim) is not None:
                    model_scores[r["model_name"]][dim].append(s[dim])
    if not model_scores:
        return None

    models = sorted(model_scores.keys())
    dims = ["correctness", "completeness", "clarity"]
    x = np.arange(len(models))
    width = 0.22

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.5), 5))
    for i, dim in enumerate(dims):
        vals = [float(np.mean(model_scores[m].get(dim, [0]))) for m in models]
        bars = ax.bar(x + i * width, vals, width, label=dim.title(),
                      color=PALETTE["dims"][i], zorder=3, edgecolor="white", linewidth=0.5)
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2, color=PALETTE["text"])

    _style_ax(ax, "Quality Sub-Scores by Model", ylabel="Score (1–10)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 10.8)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.6)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return _save(fig)


def cost_chart(run_id: str) -> BytesIO | None:
    """Horizontal bar: total API cost per cloud model."""
    data = _load_data(run_id)
    costs: dict[str, float] = defaultdict(float)
    for r in data:
        costs[r["model_name"]] += r.get("cost_usd", 0)
    paid = {k: v for k, v in costs.items() if v > 0}
    if not paid:
        return None

    items = sorted(paid.items(), key=lambda x: x[1])
    models, values = zip(*items)

    fig, ax = plt.subplots(figsize=(8, max(2.5, len(models) * 0.8)))
    colors = [PALETTE["cloud"][i % len(PALETTE["cloud"])] for i in range(len(models))]
    bars = ax.barh(range(len(models)), values, color=colors, height=0.55, zorder=3)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=10)
    _style_ax(ax, "Total API Cost per Cloud Model", xlabel="Cost (USD)")
    ax.bar_label(bars, fmt="$%.4f", padding=6, fontsize=9, color=PALETTE["text"])
    ax.set_xlim(0, max(values) * 1.35)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("$%.3f"))
    fig.tight_layout()
    return _save(fig)


def category_heatmap(run_id: str) -> BytesIO | None:
    """Heatmap grid: category x model, colored by weighted quality score."""
    data = _load_data(run_id)
    scored = [r for r in data if isinstance(r.get("scores"), dict) and r["scores"].get("weighted")]
    if not scored:
        return None

    models = sorted({r["model_name"] for r in scored})
    cats = [c for c in CATEGORY_LABELS if any(r["category"] == c for r in scored)]
    if not cats:
        return None

    grid = np.zeros((len(cats), len(models)))
    for ci, cat in enumerate(cats):
        for mi, model in enumerate(models):
            vals = [r["scores"]["weighted"] for r in scored if r["model_name"] == model and r["category"] == cat]
            grid[ci, mi] = float(np.mean(vals)) if vals else 0

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.3), max(4, len(cats) * 0.8)))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=1, vmax=10, aspect="auto")

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels([CATEGORY_LABELS[c] for c in cats], fontsize=9)

    # Annotate cells
    for ci in range(len(cats)):
        for mi in range(len(models)):
            val = grid[ci, mi]
            color = "white" if val < 4 or val > 7 else PALETTE["accent"]
            ax.text(mi, ci, f"{val:.1f}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)

    ax.set_title("Quality Heatmap: Category x Model", fontsize=13,
                 fontweight="bold", color=PALETTE["accent"], pad=12)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Weighted Score (1–10)", fontsize=9, color=PALETTE["text"])
    fig.tight_layout()
    return _save(fig)


def combined_speed_quality(run_id: str) -> BytesIO | None:
    """Scatter: TPS (x) vs Quality (y) per model, bubble size = output tokens."""
    data = _load_data(run_id)
    scored = [r for r in data if isinstance(r.get("scores"), dict) and r["scores"].get("weighted")]
    if not scored:
        return None

    models = sorted({r["model_name"] for r in scored})
    points = []
    for m in models:
        mr = [r for r in scored if r["model_name"] == m]
        points.append({
            "name": m,
            "tps": float(np.mean([r["tokens_per_sec"] for r in mr])),
            "quality": float(np.mean([r["scores"]["weighted"] for r in mr])),
            "tokens": float(np.mean([r["output_tokens"] for r in mr])),
        })

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, p in enumerate(points):
        ax.scatter(p["tps"], p["quality"], s=max(80, p["tokens"] / 3),
                   color=_model_color(p["name"], i), alpha=0.8, edgecolors="white",
                   linewidth=1.5, zorder=3)
        ax.annotate(p["name"], (p["tps"], p["quality"]),
                    xytext=(8, 6), textcoords="offset points",
                    fontsize=8, color=PALETTE["text"])

    _style_ax(ax, "Speed vs Quality Trade-off", xlabel="Avg Tokens/sec", ylabel="Avg Quality Score (1–10)")
    ax.set_ylim(0, 10.5)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.6)
    fig.tight_layout()
    return _save(fig)


def model_comparison_radar(run_id: str) -> BytesIO | None:
    """Radar chart comparing models across all dimensions."""
    data = _load_data(run_id)
    scored = [r for r in data if isinstance(r.get("scores"), dict) and r["scores"].get("weighted")]
    if not scored:
        return None

    models = sorted({r["model_name"] for r in scored})
    dims = ["correctness", "completeness", "clarity"]
    dim_labels = ["Correctness", "Completeness", "Clarity"]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]

    for i, m in enumerate(models):
        mr = [r for r in scored if r["model_name"] == m]
        values = [float(np.mean([r["scores"][d] for r in mr])) for d in dims]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=m,
                color=_model_color(m, i), markersize=5)
        ax.fill(angles, values, alpha=0.08, color=_model_color(m, i))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, fontsize=10, color=PALETTE["text"])
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=8, color=PALETTE["muted"])
    ax.set_title("Quality Radar: All Models", fontsize=13, fontweight="bold",
                 color=PALETTE["accent"], pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig)
