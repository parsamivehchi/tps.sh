"""Cost calculation and metrics utilities."""

from llm_bench.config import ModelDef


def calculate_cost(model: ModelDef, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD for a model invocation."""
    return (input_tokens * model.cost_input / 1_000_000) + (
        output_tokens * model.cost_output / 1_000_000
    )


def estimate_run_cost(
    models: list[ModelDef], num_prompts: int, avg_input_tokens: int = 350, avg_output_tokens: int = 600
) -> dict[str, float]:
    """Estimate total cost for a benchmark run."""
    costs: dict[str, float] = {}
    for m in models:
        cost = calculate_cost(m, avg_input_tokens * num_prompts, avg_output_tokens * num_prompts)
        costs[m.name] = round(cost, 4)
    costs["total"] = round(sum(costs.values()), 4)
    return costs
