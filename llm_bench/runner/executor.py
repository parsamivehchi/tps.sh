"""Main benchmark loop: model x prompt -> result."""

import asyncio
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from llm_bench.config import (
    MODELS, MODEL_BY_NAME, CATEGORIES, CATEGORY_LABELS,
    RESULTS_DIR, DEFAULT_MAX_TOKENS, ModelDef,
)
from llm_bench.models.base import ModelResponse
from llm_bench.models.registry import create_adapter
from llm_bench.prompts.loader import load_all_prompts, load_prompts_by_category, Prompt
from llm_bench.runner.cache import get_cached_result, save_to_cache

console = Console()


def _create_run_dir() -> Path:
    """Create a timestamped run directory."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


async def run_single(
    model: ModelDef,
    prompt: Prompt,
    run_dir: Path,
    use_cache: bool = True,
    hardware_monitor=None,
) -> ModelResponse:
    """Run a single model+prompt test."""
    # Check cache first (for cloud models)
    if use_cache and model.provider == "anthropic":
        cached = get_cached_result(run_dir, model.name, prompt.id)
        if cached is not None:
            return cached

    adapter = create_adapter(model)
    try:
        # Start hardware monitoring if enabled
        if hardware_monitor is not None:
            await hardware_monitor.start()

        response = await adapter.generate(
            prompt=prompt.prompt,
            system=prompt.system,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        response.prompt_id = prompt.id
        response.category = prompt.category

        # Stop monitoring and attach metrics
        if hardware_monitor is not None:
            from llm_bench.hardware.monitor import hardware_metrics_to_dict
            metrics = await hardware_monitor.stop()
            response.gpu_utilization_pct = metrics.avg_gpu_utilization_pct
            response.peak_thermal_pressure = metrics.peak_thermal_pressure
            response.hardware_metrics = hardware_metrics_to_dict(metrics)

        # Cache cloud results
        if model.provider == "anthropic":
            save_to_cache(run_dir, response)

        return response
    finally:
        await adapter.close()


async def run_benchmark(
    model_names: list[str] | None = None,
    categories: list[str] | None = None,
    use_cache: bool = True,
    hardware_metrics: bool = False,
) -> str:
    """Run the full benchmark. Returns the run_id (directory name)."""
    run_dir = _create_run_dir()
    run_id = run_dir.name

    # Set up hardware monitor if requested
    hw_monitor = None
    hw_profile_dict = None
    if hardware_metrics:
        from llm_bench.hardware.monitor import HardwareMonitor, get_hardware_profile, hardware_profile_to_dict
        hw_monitor = HardwareMonitor(sample_interval_ms=2000)
        hw_profile = get_hardware_profile()
        hw_profile_dict = hardware_profile_to_dict(hw_profile)
        console.print(f"  [cyan]Hardware monitoring enabled[/]")
        console.print(f"  Chip: {hw_profile.chip} | RAM: {hw_profile.memory_gb}GB | "
                      f"BW: {hw_profile.memory_bandwidth_gbs} GB/s | GPU: {hw_profile.gpu_cores} cores")

    # Determine which models to test
    if model_names:
        models = [MODEL_BY_NAME[n] for n in model_names if n in MODEL_BY_NAME]
    else:
        models = list(MODELS)

    # Load prompts
    if categories:
        prompts: list[Prompt] = []
        for cat in categories:
            prompts.extend(load_prompts_by_category(cat))
    else:
        prompts = load_all_prompts()

    total_tests = len(models) * len(prompts)
    results: list[ModelResponse] = []

    console.print(f"\n[bold cyan]tps.sh Run: {run_id}[/]")
    console.print(f"  Models: {len(models)} | Prompts: {len(prompts)} | Total tests: {total_tests}\n")

    # Save run metadata
    meta = {
        "run_id": run_id,
        "started_at": datetime.now().isoformat(),
        "models": [m.name for m in models],
        "categories": categories or CATEGORIES,
        "total_tests": total_tests,
    }
    if hw_profile_dict:
        meta["hardware_profile"] = hw_profile_dict
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        overall_task = progress.add_task("Overall", total=total_tests)

        # Run one model at a time (respects OLLAMA_MAX_LOADED_MODELS=1)
        for model in models:
            model_task = progress.add_task(f"  {model.name}", total=len(prompts))

            # Warmup
            progress.update(model_task, description=f"  {model.name} [warming up]")
            adapter = create_adapter(model)
            try:
                await adapter.warmup()
            except Exception as e:
                console.print(f"  [yellow]Warmup failed for {model.name}: {e}[/]")
            finally:
                await adapter.close()

            progress.update(model_task, description=f"  {model.name}")

            for prompt in prompts:
                response = await run_single(model, prompt, run_dir, use_cache,
                                            hardware_monitor=hw_monitor if model.provider == "ollama" else None)
                results.append(response)

                # Status indicator
                status = "[green]\u2713[/]" if not response.error else "[red]\u2717[/]"
                progress.update(model_task, advance=1)
                progress.update(overall_task, advance=1)

    # Save all results
    results_data = [asdict(r) for r in results]
    (run_dir / "results.json").write_text(json.dumps(results_data, indent=2))

    # Update metadata
    meta["completed_at"] = datetime.now().isoformat()
    meta["success_count"] = sum(1 for r in results if not r.error)
    meta["error_count"] = sum(1 for r in results if r.error)
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # Print summary
    console.print(f"\n[bold green]Benchmark complete![/]")
    console.print(f"  Run ID: {run_id}")
    console.print(f"  Results: {meta['success_count']} success, {meta['error_count']} errors")
    console.print(f"  Saved to: {run_dir}")

    total_cost = sum(r.cost_usd for r in results)
    if total_cost > 0:
        console.print(f"  Total cloud cost: ${total_cost:.4f}")

    return run_id
