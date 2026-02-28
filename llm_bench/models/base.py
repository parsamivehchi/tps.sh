"""Abstract ModelAdapter and ModelResponse dataclass."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelResponse:
    """Captures all metrics from a single model invocation."""
    model_name: str
    prompt_id: str
    category: str
    output: str
    ttft_ms: float                  # time to first token
    total_time_ms: float            # end-to-end wall time
    tokens_per_sec: float           # output generation speed
    input_tokens: int
    output_tokens: int
    cost_usd: float                 # $0 for local models
    memory_mb: Optional[float] = None  # VRAM for Ollama models
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    # Hardware metrics (Phase 2) — populated when --hardware-metrics is used
    gpu_utilization_pct: Optional[float] = None
    peak_thermal_pressure: Optional[str] = None
    hardware_metrics: Optional[dict] = None


class ModelAdapter(ABC):
    """Interface every model backend must implement."""

    @abstractmethod
    async def generate(self, prompt: str, system: str = "", max_tokens: int = 4096) -> ModelResponse:
        ...

    @abstractmethod
    async def warmup(self) -> None:
        """Send a trivial request to load the model into memory."""
        ...

    @abstractmethod
    async def close(self) -> None:
        ...
