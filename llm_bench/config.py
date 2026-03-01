"""Model definitions, categories, and constants for tps.sh."""

from pathlib import Path
from dataclasses import dataclass

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_DIR / "results"
SCORED_DIR = DATA_DIR / "scored"
EXPORTS_DIR = DATA_DIR / "exports"
REPORTS_DIR = PROJECT_ROOT / "reports"
PROMPTS_DIR = Path(__file__).parent / "prompts" / "bank"
DASHBOARD_DATA_DIR = PROJECT_ROOT / "dashboard" / "public" / "data"

# Ensure directories exist
for d in [RESULTS_DIR, SCORED_DIR, EXPORTS_DIR, REPORTS_DIR, DASHBOARD_DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelDef:
    name: str            # display name
    model_id: str        # API model ID or Ollama model name
    provider: str        # "ollama" or "anthropic"
    model_type: str      # description like "30B MoE"
    cost_input: float    # $ per 1M input tokens (0 for local)
    cost_output: float   # $ per 1M output tokens (0 for local)


MODELS: list[ModelDef] = [
    ModelDef("qwen3-coder", "qwen3-coder:latest", "ollama", "30B MoE", 0, 0),
    ModelDef("qwen2.5-coder:14b", "qwen2.5-coder:14b", "ollama", "14B dense", 0, 0),
    ModelDef("deepseek-r1:14b", "deepseek-r1:14b", "ollama", "14B dense", 0, 0),
    ModelDef("glm-4.7-flash", "glm-4.7-flash:latest", "ollama", "~9B dense", 0, 0),
    ModelDef("Claude Haiku 4.5", "claude-haiku-4-5", "anthropic", "Cloud", 1.0, 5.0),
    ModelDef("Claude Sonnet 4.6", "claude-sonnet-4-6", "anthropic", "Cloud", 3.0, 15.0),
    ModelDef("Claude Opus 4.6", "claude-opus-4-6", "anthropic", "Cloud", 5.0, 25.0),
]

MODEL_BY_NAME: dict[str, ModelDef] = {m.name: m for m in MODELS}

CATEGORIES = [
    "code_generation",
    "debugging_reasoning",
    "short_quick",
    "long_complex",
    "refactoring",
    "explanation_teaching",
    "tool_calling",
]

CATEGORY_LABELS = {
    "code_generation": "Code Generation",
    "debugging_reasoning": "Debugging & Reasoning",
    "short_quick": "Short Quick Tasks",
    "long_complex": "Long Complex Research",
    "refactoring": "Refactoring",
    "explanation_teaching": "Explanation & Teaching",
    "tool_calling": "Tool Calling / Agentic",
}

# Judge config
JUDGE_MODEL = "claude-sonnet-4-6"
JUDGE_MAX_TOKENS = 2048

# Quality scoring weights
QUALITY_WEIGHTS = {
    "correctness": 0.40,
    "completeness": 0.35,
    "clarity": 0.25,
}

# Ollama config
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MAX_LOADED_MODELS = 1

# Benchmark defaults
DEFAULT_MAX_TOKENS = 4096
WARMUP_PROMPT = "Hi"
WARMUP_MAX_TOKENS = 8

# Hardware profile (populated at runtime when --hardware-metrics is used)
HARDWARE_PROFILE = None
