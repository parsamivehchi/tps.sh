"""Model name -> adapter factory."""

from llm_bench.config import ModelDef, MODELS, MODEL_BY_NAME
from llm_bench.models.base import ModelAdapter
from llm_bench.models.ollama_adapter import OllamaAdapter
from llm_bench.models.anthropic_adapter import AnthropicAdapter


def create_adapter(model: ModelDef) -> ModelAdapter:
    """Create the appropriate adapter for a model definition."""
    if model.provider == "ollama":
        return OllamaAdapter(model_id=model.model_id, model_name=model.name)
    elif model.provider == "anthropic":
        return AnthropicAdapter(
            model_id=model.model_id,
            model_name=model.name,
            cost_input=model.cost_input,
            cost_output=model.cost_output,
        )
    else:
        raise ValueError(f"Unknown provider: {model.provider}")


def get_adapter(model_name: str) -> ModelAdapter:
    """Get an adapter by model display name."""
    model = MODEL_BY_NAME.get(model_name)
    if not model:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_BY_NAME.keys())}")
    return create_adapter(model)


def get_all_adapters() -> list[tuple[ModelDef, ModelAdapter]]:
    """Get adapters for all configured models."""
    return [(m, create_adapter(m)) for m in MODELS]
