"""Send outputs to Claude Sonnet for quality scoring."""

import json
import re
from pathlib import Path

import anthropic
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from llm_bench.config import (
    JUDGE_MODEL, JUDGE_MAX_TOKENS, QUALITY_WEIGHTS,
    RESULTS_DIR, SCORED_DIR,
)
from llm_bench.judge.templates import JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE

console = Console()


def _parse_judge_response(text: str) -> dict:
    """Parse judge response JSON, handling possible markdown wrapping."""
    text = text.strip()
    # Strip markdown code fences if present
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        text = match.group(1)
    # Try to find JSON object
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


def _compute_weighted_score(scores: dict) -> float:
    """Compute weighted quality score from sub-scores."""
    return round(
        scores["correctness"] * QUALITY_WEIGHTS["correctness"]
        + scores["completeness"] * QUALITY_WEIGHTS["completeness"]
        + scores["clarity"] * QUALITY_WEIGHTS["clarity"],
        2,
    )


async def judge_run(run_id: str) -> None:
    """Score all outputs in a run using Claude Sonnet as judge."""
    run_dir = RESULTS_DIR / run_id
    results_file = run_dir / "results.json"
    if not results_file.exists():
        console.print(f"[red]Results not found: {results_file}[/]")
        return

    results = json.loads(results_file.read_text())

    # Filter out errored results
    valid_results = [r for r in results if not r.get("error")]
    console.print(f"\n[bold cyan]Judging {len(valid_results)} outputs from run {run_id}[/]")

    client = anthropic.Anthropic()
    scored_results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Judging", total=len(valid_results))

        for result in valid_results:
            model_name = result["model_name"]
            prompt_id = result["prompt_id"]
            output = result["output"]

            # Find the original prompt text
            from llm_bench.prompts.loader import load_all_prompts
            prompts = {p.id: p for p in load_all_prompts()}
            original_prompt = prompts.get(prompt_id)
            prompt_text = original_prompt.prompt if original_prompt else "(prompt not found)"

            # Call the judge
            user_msg = JUDGE_USER_TEMPLATE.format(
                prompt=prompt_text,
                response=output[:8000],  # Truncate very long outputs
            )

            try:
                response = client.messages.create(
                    model=JUDGE_MODEL,
                    max_tokens=JUDGE_MAX_TOKENS,
                    system=JUDGE_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_msg}],
                )
                judge_text = response.content[0].text
                scores = _parse_judge_response(judge_text)

                # Add bias flag for Claude-judging-Claude
                is_claude_model = "Claude" in model_name
                weighted = _compute_weighted_score(scores)

                scored_entry = {
                    **result,
                    "scores": {
                        "correctness": scores["correctness"],
                        "completeness": scores["completeness"],
                        "clarity": scores["clarity"],
                        "weighted": weighted,
                        "reasoning": scores.get("reasoning", ""),
                    },
                    "judge_model": JUDGE_MODEL,
                    "self_bias_flag": is_claude_model,
                    "judge_input_tokens": response.usage.input_tokens,
                    "judge_output_tokens": response.usage.output_tokens,
                }
                scored_results.append(scored_entry)

            except Exception as e:
                console.print(f"  [red]Judge error for {model_name}/{prompt_id}: {e}[/]")
                scored_results.append({
                    **result,
                    "scores": None,
                    "judge_error": str(e),
                })

            progress.update(task, advance=1)

    # Save scored results
    scored_dir = SCORED_DIR / run_id
    scored_dir.mkdir(parents=True, exist_ok=True)
    (scored_dir / "scored_results.json").write_text(json.dumps(scored_results, indent=2))

    # Summary
    valid_scores = [s for s in scored_results if s.get("scores")]
    if valid_scores:
        avg_weighted = sum(s["scores"]["weighted"] for s in valid_scores) / len(valid_scores)
        total_judge_cost = sum(
            (s.get("judge_input_tokens", 0) * 3.0 / 1_000_000)
            + (s.get("judge_output_tokens", 0) * 15.0 / 1_000_000)
            for s in valid_scores
        )
        console.print(f"\n[bold green]Judging complete![/]")
        console.print(f"  Scored: {len(valid_scores)}/{len(valid_results)} outputs")
        console.print(f"  Average weighted score: {avg_weighted:.2f}/10")
        console.print(f"  Judging cost: ${total_judge_cost:.4f}")
        console.print(f"  Saved to: {scored_dir}")

        # Flag bias
        biased = [s for s in valid_scores if s.get("self_bias_flag")]
        if biased:
            console.print(f"  [yellow]Note: {len(biased)} scores flagged for Claude-judging-Claude bias[/]")
