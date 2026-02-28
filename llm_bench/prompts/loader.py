"""Load YAML prompt bank files into Prompt objects."""

from dataclasses import dataclass
from pathlib import Path
import yaml

from llm_bench.config import PROMPTS_DIR


@dataclass
class Prompt:
    id: str
    title: str
    category: str
    system: str
    prompt: str


def load_category(filepath: Path) -> list[Prompt]:
    """Load prompts from a single category YAML file."""
    with open(filepath) as f:
        data = yaml.safe_load(f)
    category = data["category"]
    return [
        Prompt(
            id=p["id"],
            title=p["title"],
            category=category,
            system=p.get("system", ""),
            prompt=p["prompt"].strip(),
        )
        for p in data["prompts"]
    ]


def load_all_prompts(prompts_dir: Path = PROMPTS_DIR) -> list[Prompt]:
    """Load all prompts from all category YAML files."""
    prompts = []
    for yaml_file in sorted(prompts_dir.glob("*.yaml")):
        prompts.extend(load_category(yaml_file))
    return prompts


def load_prompts_by_category(category: str, prompts_dir: Path = PROMPTS_DIR) -> list[Prompt]:
    """Load prompts for a single category."""
    filepath = prompts_dir / f"{category}.yaml"
    if not filepath.exists():
        raise FileNotFoundError(f"Category file not found: {filepath}")
    return load_category(filepath)
