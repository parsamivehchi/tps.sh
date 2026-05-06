#!/usr/bin/env python3
"""Overnight runner — keeps a single local model hot in GPU for back-to-back runs.

Usage:
    .venv/bin/python overnight_run.py                          # 8h, qwen3-coder
    .venv/bin/python overnight_run.py --hours 4                # 4h
    .venv/bin/python overnight_run.py --model gemma4:26b       # different model
    .venv/bin/python overnight_run.py --hardware-metrics       # capture GPU/thermal
    .venv/bin/python overnight_run.py --no-optimize            # skip system optimization
"""

import argparse
import asyncio
import json
import os
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

from rich.console import Console
from rich.table import Table

from llm_bench.config import (
    MODEL_BY_NAME, CATEGORIES, RESULTS_DIR, DEFAULT_MAX_TOKENS, OLLAMA_BASE_URL,
    OLLAMA_NUM_CTX, OLLAMA_NUM_BATCH,
)
from llm_bench.models.registry import create_adapter
from llm_bench.prompts.loader import load_all_prompts

console = Console()

# Background processes safe to kill during overnight runs.
# These are Apple daemons and apps that consume memory/bandwidth but aren't needed.
KILLABLE_PROCESSES = [
    "mediaanalysisd",           # Apple media analysis (~1.3 GB)
    "photoanalysisd",           # Photo library analysis (~0.5 GB)
    "photolibraryd",            # Photo library daemon (~0.4 GB)
    "com.apple.quicklook.ThumbnailsAgent",  # QuickLook thumbnails (~0.6 GB)
    "spotlightknowledged",      # Spotlight knowledge (~0.8 GB)
    "Notes",                    # Apple Notes app
    "Bitwarden",                # Password manager
    "Helium",                   # Browser
]


def optimize_system() -> dict:
    """Kill memory-hogging background processes and check system state.
    Returns a dict of what was done."""
    report = {"killed": [], "warnings": [], "memory_freed_mb": 0}

    console.print("\n[bold cyan]Pre-flight: System Optimization[/]")

    # Kill background processes
    for proc_name in KILLABLE_PROCESSES:
        try:
            result = subprocess.run(
                ["pkill", "-f", proc_name],
                capture_output=True, timeout=5,
            )
            if result.returncode == 0:
                report["killed"].append(proc_name)
                console.print(f"  [green]Killed[/] {proc_name}")
        except Exception:
            pass

    if not report["killed"]:
        console.print("  [dim]No killable processes found[/]")

    # Disable Spotlight indexing (requires sudo — skip if not available)
    try:
        result = subprocess.run(
            ["sudo", "-n", "mdutil", "-a", "-i", "off"],
            capture_output=True, timeout=5,
        )
        if result.returncode == 0:
            console.print("  [green]Disabled[/] Spotlight indexing")
            report["spotlight_disabled"] = True
        else:
            console.print("  [yellow]Spotlight[/] — run `sudo mdutil -a -i off` manually for best results")
    except Exception:
        console.print("  [yellow]Spotlight[/] — run `sudo mdutil -a -i off` manually for best results")

    # Check swap
    try:
        result = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True, timeout=5)
        if "used = " in result.stdout:
            used_str = result.stdout.split("used = ")[1].split("M")[0].strip()
            swap_mb = float(used_str)
            if swap_mb > 500:
                report["warnings"].append(f"Swap usage: {swap_mb:.0f} MB — close more apps to eliminate swap")
                console.print(f"  [red]WARNING[/] Swap: {swap_mb:.0f} MB in use — this hurts TPS")
            else:
                console.print(f"  [green]Swap:[/] {swap_mb:.0f} MB (OK)")
    except Exception:
        pass

    # Check available GPU memory
    try:
        result = subprocess.run(
            ["curl", "-s", f"{OLLAMA_BASE_URL}/api/ps"],
            capture_output=True, text=True, timeout=5,
        )
        data = json.loads(result.stdout)
        for m in data.get("models", []):
            vram_gb = m.get("size_vram", 0) / (1024**3)
            ctx = m.get("context_length", 0)
            console.print(f"  [cyan]Loaded:[/] {m['name']} — {vram_gb:.1f} GB VRAM, ctx={ctx}")
            if ctx != OLLAMA_NUM_CTX:
                report["warnings"].append(
                    f"Model loaded at ctx={ctx}, benchmark uses {OLLAMA_NUM_CTX}. "
                    f"Will reload at correct context on first request."
                )
                console.print(
                    f"  [yellow]Context mismatch[/] — loaded at {ctx}, "
                    f"will reload at {OLLAMA_NUM_CTX} on first request"
                )
    except Exception:
        pass

    console.print()
    return report


async def preload_model(model_id: str) -> None:
    """Force-load the model into GPU at the correct context size with keep_alive=-1."""
    import aiohttp
    console.print(f"[cyan]Pre-loading {model_id} into GPU...[/]")
    console.print(f"  keep_alive=-1 | num_ctx={OLLAMA_NUM_CTX} | num_batch={OLLAMA_NUM_BATCH} | kv_cache=q8_0")
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": False,
        "keep_alive": -1,
        "options": {
            "num_predict": 1,
            "num_ctx": OLLAMA_NUM_CTX,
            "num_batch": OLLAMA_NUM_BATCH,
        },
    }
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload) as resp:
            await resp.read()

    # Verify it loaded correctly
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(f"{OLLAMA_BASE_URL}/api/ps") as resp:
            data = await resp.json()
            for m in data.get("models", []):
                if model_id in m.get("name", ""):
                    vram_gb = m.get("size_vram", 0) / (1024**3)
                    ctx = m.get("context_length", 0)
                    expires = m.get("expires_at", "never")
                    console.print(f"  [green]Loaded:[/] {vram_gb:.1f} GB VRAM | ctx={ctx} | expires={expires}")

    console.print(f"[green]Model pinned in VRAM — no unloads, no dips.[/]")


async def run_overnight(
    model_name: str,
    hours: float,
    hardware_metrics: bool,
    skip_optimize: bool,
) -> None:
    model = MODEL_BY_NAME.get(model_name)
    if model is None:
        console.print(f"[red]Unknown model: {model_name}[/]")
        console.print(f"Available local models: {[m for m in MODEL_BY_NAME if MODEL_BY_NAME[m].provider == 'ollama']}")
        return

    if model.provider != "ollama":
        console.print(f"[red]{model_name} is a cloud model — overnight runs are for local models only.[/]")
        return

    # System optimization
    if not skip_optimize:
        optimize_system()

    # Pre-load the model into GPU at correct context size
    await preload_model(model.model_id)

    prompts = load_all_prompts()
    end_time = time.time() + (hours * 3600)
    duration_str = f"{hours}h"

    # Create run directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"overnight_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Set up hardware monitor
    hw_monitor = None
    if hardware_metrics:
        from llm_bench.hardware.monitor import HardwareMonitor
        hw_monitor = HardwareMonitor(sample_interval_ms=2000)
        console.print("[cyan]Hardware monitoring enabled[/]")

    console.print(f"\n[bold cyan]Overnight Run: {model_name}[/]")
    console.print(f"  Duration: {duration_str} (until {datetime.fromtimestamp(end_time).strftime('%H:%M')})")
    console.print(f"  Prompts per cycle: {len(prompts)}")
    console.print(f"  Config: ctx={OLLAMA_NUM_CTX} | batch={OLLAMA_NUM_BATCH} | kv=q8_0 | flash_attn=on")
    console.print(f"  Run dir: {run_dir}")
    console.print()

    meta = {
        "run_id": run_dir.name,
        "type": "overnight",
        "model": model_name,
        "started_at": datetime.now().isoformat(),
        "planned_hours": hours,
        "prompts_per_cycle": len(prompts),
        "config": {
            "num_ctx": OLLAMA_NUM_CTX,
            "num_batch": OLLAMA_NUM_BATCH,
            "kv_cache_type": "q8_0",
            "flash_attention": True,
            "keep_alive": -1,
        },
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    cycle = 0
    total_prompts = 0
    all_tps: list[float] = []

    try:
        while time.time() < end_time:
            cycle += 1
            remaining = (end_time - time.time()) / 3600
            console.print(f"\n[bold]--- Cycle {cycle} --- ({remaining:.1f}h remaining)[/]")

            cycle_tps: list[float] = []
            for i, prompt in enumerate(prompts, 1):
                if time.time() >= end_time:
                    console.print("[yellow]Time's up — stopping mid-cycle.[/]")
                    break

                adapter = create_adapter(model)
                try:
                    if hw_monitor:
                        await hw_monitor.start()

                    response = await adapter.generate(
                        prompt=prompt.prompt,
                        system=prompt.system,
                        max_tokens=DEFAULT_MAX_TOKENS,
                    )
                    response.prompt_id = prompt.id
                    response.category = prompt.category

                    if hw_monitor:
                        from llm_bench.hardware.monitor import hardware_metrics_to_dict
                        metrics = await hw_monitor.stop()
                        response.gpu_utilization_pct = metrics.avg_gpu_utilization_pct
                        response.hardware_metrics = hardware_metrics_to_dict(metrics)

                    total_prompts += 1
                    tps = response.tokens_per_sec
                    cycle_tps.append(tps)
                    all_tps.append(tps)

                    status = "[green]OK[/]" if not response.error else f"[red]ERR: {response.error[:50]}[/]"
                    console.print(
                        f"  [{i}/{len(prompts)}] {prompt.id:<30} "
                        f"TPS: {tps:>6.1f}  TTFT: {response.ttft_ms/1000:>5.1f}s  "
                        f"Tokens: {response.output_tokens:>5}  {status}"
                    )

                    # Append to progress log
                    log_entry = {
                        "ts": datetime.now().isoformat(),
                        "cycle": cycle,
                        "model_name": model_name,
                        "prompt_id": prompt.id,
                        "category": prompt.category,
                        "tps": tps,
                        "ttft_ms": response.ttft_ms,
                        "total_time_ms": response.total_time_ms,
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "error": response.error if hasattr(response, "error") else None,
                    }
                    with (run_dir / "progress.jsonl").open("a") as f:
                        f.write(json.dumps(log_entry) + "\n")

                finally:
                    await adapter.close()

            # Cycle summary
            if cycle_tps:
                avg = sum(cycle_tps) / len(cycle_tps)
                console.print(f"\n  [cyan]Cycle {cycle} avg TPS: {avg:.1f} ({len(cycle_tps)} prompts)[/]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user — saving results...[/]")

    # Re-enable Spotlight if we disabled it
    try:
        subprocess.run(["sudo", "-n", "mdutil", "-a", "-i", "on"], capture_output=True, timeout=5)
    except Exception:
        pass

    # Final summary
    elapsed_h = (time.time() - (end_time - hours * 3600)) / 3600
    meta["completed_at"] = datetime.now().isoformat()
    meta["actual_hours"] = round(elapsed_h, 2)
    meta["total_cycles"] = cycle
    meta["total_prompts"] = total_prompts
    if all_tps:
        meta["avg_tps"] = round(sum(all_tps) / len(all_tps), 2)
        meta["min_tps"] = round(min(all_tps), 2)
        meta["max_tps"] = round(max(all_tps), 2)
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    console.print(f"\n[bold green]Overnight run complete![/]")
    console.print(f"  Model: {model_name}")
    console.print(f"  Duration: {elapsed_h:.1f}h ({cycle} cycles, {total_prompts} prompts)")
    if all_tps:
        console.print(f"  TPS — avg: {meta['avg_tps']}, min: {meta['min_tps']}, max: {meta['max_tps']}")
    console.print(f"  Results: {run_dir}")


def main():
    parser = argparse.ArgumentParser(description="Overnight local LLM benchmark runner")
    parser.add_argument("--model", "-m", default="qwen3-coder", help="Local model name (default: qwen3-coder)")
    parser.add_argument("--hours", type=float, default=8.0, help="Hours to run (default: 8)")
    parser.add_argument("--hardware-metrics", action="store_true", help="Capture GPU/thermal metrics")
    parser.add_argument("--no-optimize", action="store_true", help="Skip system optimization (killing background processes)")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    asyncio.run(run_overnight(args.model, args.hours, args.hardware_metrics, args.no_optimize))


if __name__ == "__main__":
    main()
