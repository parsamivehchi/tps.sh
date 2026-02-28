"""Hardware analysis data: specs, TPS projections, and cost modeling for Apple Silicon."""

from dataclasses import dataclass


@dataclass
class HardwareSpec:
    """Specification for an Apple Silicon Mac configuration."""
    name: str
    chip: str
    memory_bandwidth_gbs: float  # GB/s
    max_ram_gb: int
    gpu_cores: int
    neural_engine_tops: float  # trillion operations per second
    price_usd: int  # approximate
    form_factor: str  # "laptop", "desktop", "mini"
    notes: str = ""


@dataclass
class ModelSpec:
    """Memory footprint of a model at different quantization levels."""
    name: str
    params_b: float  # billions of parameters
    architecture: str  # "dense" or "moe"
    active_params_b: float  # for MoE, the actually-activated params per token
    size_fp16_gb: float  # FP16 model weight size
    size_q8_gb: float
    size_q4_gb: float


# ── Hardware Specifications ──

HARDWARE_SPECS: list[HardwareSpec] = [
    HardwareSpec(
        name="M2 Max MacBook Pro 32GB (current)",
        chip="M2 Max",
        memory_bandwidth_gbs=400,
        max_ram_gb=96,
        gpu_cores=38,
        neural_engine_tops=15.8,
        price_usd=0,  # already owned
        form_factor="laptop",
        notes="Your current setup. Actual benchmark data available.",
    ),
    HardwareSpec(
        name="M4 Pro Mac Mini 48GB",
        chip="M4 Pro",
        memory_bandwidth_gbs=273,
        max_ram_gb=64,
        gpu_cores=20,
        neural_engine_tops=38.0,
        price_usd=1799,
        form_factor="mini",
        notes="Budget always-on server. Lower bandwidth than M2 Max but newer architecture.",
    ),
    HardwareSpec(
        name="M4 Max MacBook Pro 64GB",
        chip="M4 Max",
        memory_bandwidth_gbs=546,
        max_ram_gb=128,
        gpu_cores=40,
        neural_engine_tops=38.0,
        price_usd=3499,
        form_factor="laptop",
        notes="Portable powerhouse. 37% more bandwidth than your M2 Max.",
    ),
    HardwareSpec(
        name="M4 Max Mac Studio 128GB",
        chip="M4 Max",
        memory_bandwidth_gbs=546,
        max_ram_gb=128,
        gpu_cores=40,
        neural_engine_tops=38.0,
        price_usd=4499,
        form_factor="desktop",
        notes="Desktop form factor with superior sustained thermals.",
    ),
    HardwareSpec(
        name="M3 Ultra Mac Studio 192GB",
        chip="M3 Ultra",
        memory_bandwidth_gbs=819,
        max_ram_gb=512,
        gpu_cores=80,
        neural_engine_tops=31.6,
        price_usd=5999,
        form_factor="desktop",
        notes="Maximum single-node. Can run 70B+ models at full precision.",
    ),
    HardwareSpec(
        name="M4 Ultra Mac Studio (upcoming)",
        chip="M4 Ultra",
        memory_bandwidth_gbs=819,
        max_ram_gb=512,
        gpu_cores=80,
        neural_engine_tops=76.0,
        price_usd=6999,
        form_factor="desktop",
        notes="Expected 2026. Same bandwidth class as M3 Ultra with newer arch.",
    ),
]

# ── Cluster Configurations ──

CLUSTER_CONFIGS = [
    {
        "name": "2x M4 Pro Mac Mini (exo)",
        "nodes": 2,
        "node_chip": "M4 Pro",
        "aggregate_bandwidth_gbs": 2 * 273,
        "aggregate_ram_gb": 2 * 48,
        "price_usd": 2 * 1799,
        "interconnect": "Thunderbolt 5 (120 Gbps)",
        "scaling_efficiency": 0.85,
        "notes": "Near-linear scaling for models that fit split across 2 nodes.",
    },
    {
        "name": "4x M4 Pro Mac Mini (exo)",
        "nodes": 4,
        "node_chip": "M4 Pro",
        "aggregate_bandwidth_gbs": 4 * 273,
        "aggregate_ram_gb": 4 * 48,
        "price_usd": 4 * 1799,
        "interconnect": "Thunderbolt 5 (120 Gbps)",
        "scaling_efficiency": 0.75,
        "notes": "Can run 70B+ models. Scaling loss from inter-node communication.",
    },
    {
        "name": "2x M4 Max Mac Studio (exo)",
        "nodes": 2,
        "node_chip": "M4 Max",
        "aggregate_bandwidth_gbs": 2 * 546,
        "aggregate_ram_gb": 2 * 128,
        "price_usd": 2 * 4499,
        "interconnect": "Thunderbolt 5 (120 Gbps)",
        "scaling_efficiency": 0.85,
        "notes": "Premium cluster. 256GB total for very large models.",
    },
]


# ── Model Memory Footprints ──

MODEL_SPECS: list[ModelSpec] = [
    ModelSpec("qwen3-coder (30B MoE)", 30.5, "moe", 8.0, 61.0, 30.5, 15.3),
    ModelSpec("qwen2.5-coder:14b", 14.7, "dense", 14.7, 29.4, 14.7, 7.4),
    ModelSpec("deepseek-r1:14b", 14.7, "dense", 14.7, 29.4, 14.7, 7.4),
    ModelSpec("glm-4.7-flash (~9B)", 9.4, "dense", 9.4, 18.8, 9.4, 4.7),
    ModelSpec("Llama 3.3 70B", 70.6, "dense", 70.6, 141.2, 70.6, 35.3),
    ModelSpec("Llama 3.1 405B", 405.0, "dense", 405.0, 810.0, 405.0, 202.5),
    ModelSpec("Qwen3 235B MoE", 235.0, "moe", 22.0, 470.0, 235.0, 117.5),
    ModelSpec("DeepSeek-V3 671B MoE", 671.0, "moe", 37.0, 1342.0, 671.0, 335.5),
    ModelSpec("Mistral Large 2 123B", 123.0, "dense", 123.0, 246.0, 123.0, 61.5),
]

# ── Actual Benchmark Results (Phase 1) ──

ACTUAL_BENCHMARKS = {
    "M2 Max MacBook Pro 32GB": {
        "qwen3-coder": {"tps": 48.8, "ttft_ms": 1100, "quality": 7.48},
        "qwen2.5-coder:14b": {"tps": 15.6, "ttft_ms": 1500, "quality": 6.64},
        "deepseek-r1:14b": {"tps": 14.6, "ttft_ms": 70200, "quality": 5.89},
        "glm-4.7-flash": {"tps": 10.2, "ttft_ms": 54800, "quality": 5.30},
    },
}

# ── Cloud API Pricing (for cost comparison) ──

CLOUD_PRICING = {
    "Claude Haiku 4.5": {
        "cost_per_1m_input": 1.0,
        "cost_per_1m_output": 5.0,
        "avg_tps": 169.7,
        "quality": 8.25,
    },
    "Claude Sonnet 4.6": {
        "cost_per_1m_input": 3.0,
        "cost_per_1m_output": 15.0,
        "avg_tps": 77.7,
        "quality": 8.59,
    },
    "Claude Opus 4.6": {
        "cost_per_1m_input": 5.0,
        "cost_per_1m_output": 25.0,
        "avg_tps": 76.6,
        "quality": 8.65,
    },
    "GPT-4o": {
        "cost_per_1m_input": 2.5,
        "cost_per_1m_output": 10.0,
        "avg_tps": 90.0,
        "quality": None,
    },
}


# ── Projection Functions ──

def projected_tps(
    bandwidth_gbs: float,
    model_size_gb: float,
    efficiency: float = 0.6,
) -> float:
    """Project tokens per second from memory bandwidth and model size.

    Formula: TPS ≈ (Bandwidth / Model Size) × Efficiency
    Real-world efficiency is typically 50-70% of theoretical maximum
    due to KV cache, framework overhead, and memory controller contention.
    """
    if model_size_gb <= 0:
        return 0.0
    return (bandwidth_gbs / model_size_gb) * efficiency


def projected_tps_moe(
    bandwidth_gbs: float,
    active_params_gb: float,
    total_params_gb: float,
    efficiency: float = 0.55,
) -> float:
    """Project TPS for Mixture-of-Experts models.

    MoE models only activate a subset of parameters per token,
    but the full model must reside in memory. The bandwidth bottleneck
    is proportional to active parameters, not total parameters.

    Lower default efficiency due to expert routing overhead.
    """
    if active_params_gb <= 0:
        return 0.0
    return (bandwidth_gbs / active_params_gb) * efficiency


def model_fits_in_ram(
    model_size_gb: float,
    ram_gb: int,
    kv_cache_overhead: float = 0.15,
    os_overhead_gb: float = 4.0,
) -> bool:
    """Check if a model can fit in available memory.

    Accounts for:
    - OS and system overhead (~4GB)
    - KV cache during inference (~15% of model size)
    """
    required = model_size_gb * (1 + kv_cache_overhead) + os_overhead_gb
    return required <= ram_gb


def cost_per_million_tokens(
    hardware_cost_usd: float,
    tps: float,
    lifespan_years: float = 3.0,
    electricity_cost_per_kwh: float = 0.15,
    power_draw_watts: float = 30.0,
    utilization_pct: float = 0.25,
) -> float:
    """Calculate amortized cost per 1M tokens for local inference.

    Assumes:
    - Hardware amortized over lifespan_years
    - Average power draw during inference
    - Utilization percentage (how often the machine is running inference)
    """
    if tps <= 0:
        return float("inf")

    # Hardware amortization
    hours_per_year = 365.25 * 24
    total_hours = hours_per_year * lifespan_years
    inference_hours = total_hours * utilization_pct
    tokens_total = tps * 3600 * inference_hours
    hardware_per_token = hardware_cost_usd / tokens_total if tokens_total > 0 else 0

    # Electricity
    kwh_per_hour = power_draw_watts / 1000
    electricity_per_hour = kwh_per_hour * electricity_cost_per_kwh
    electricity_per_token = electricity_per_hour / (tps * 3600) if tps > 0 else 0

    # Cost per million tokens
    return (hardware_per_token + electricity_per_token) * 1_000_000


def breakeven_tokens(
    hardware_cost_usd: float,
    local_tps: float,
    cloud_cost_per_1m_tokens: float,
    electricity_cost_per_kwh: float = 0.15,
    power_draw_watts: float = 30.0,
) -> float:
    """Calculate how many tokens until local inference breaks even vs cloud.

    Returns the number of output tokens at which the local hardware
    has paid for itself compared to cloud API pricing.
    """
    if cloud_cost_per_1m_tokens <= 0 or local_tps <= 0:
        return float("inf")

    # Marginal cost per token for local (electricity only)
    kwh_per_token = (power_draw_watts / 1000) / (local_tps * 3600)
    local_marginal_per_token = kwh_per_token * electricity_cost_per_kwh

    # Cloud cost per token
    cloud_per_token = cloud_cost_per_1m_tokens / 1_000_000

    # Net savings per token
    savings_per_token = cloud_per_token - local_marginal_per_token
    if savings_per_token <= 0:
        return float("inf")

    return hardware_cost_usd / savings_per_token


def project_all_hardware(
    model: ModelSpec,
    quantization: str = "q4",
) -> list[dict]:
    """Project TPS for a model across all hardware configurations.

    Returns a list of dicts with hardware name, projected TPS, and whether
    the model fits in RAM.
    """
    size_map = {"fp16": model.size_fp16_gb, "q8": model.size_q8_gb, "q4": model.size_q4_gb}
    model_size = size_map.get(quantization, model.size_q4_gb)

    # For MoE models, use active params for TPS calc
    if model.architecture == "moe":
        active_size_ratio = model.active_params_b / model.params_b
        active_size = model_size * active_size_ratio
    else:
        active_size = model_size

    results = []
    for hw in HARDWARE_SPECS:
        fits = model_fits_in_ram(model_size, hw.max_ram_gb)
        if model.architecture == "moe":
            tps = projected_tps_moe(
                hw.memory_bandwidth_gbs, active_size, model_size
            ) if fits else 0.0
        else:
            tps = projected_tps(
                hw.memory_bandwidth_gbs, model_size
            ) if fits else 0.0

        results.append({
            "hardware": hw.name,
            "chip": hw.chip,
            "bandwidth_gbs": hw.memory_bandwidth_gbs,
            "ram_gb": hw.max_ram_gb,
            "model_size_gb": round(model_size, 1),
            "fits": fits,
            "projected_tps": round(tps, 1),
            "price_usd": hw.price_usd,
            "form_factor": hw.form_factor,
        })

    # Add cluster configs
    for cluster in CLUSTER_CONFIGS:
        fits = model_fits_in_ram(model_size, cluster["aggregate_ram_gb"])
        eff = cluster["scaling_efficiency"]
        if model.architecture == "moe":
            tps = projected_tps_moe(
                cluster["aggregate_bandwidth_gbs"], active_size, model_size,
                efficiency=0.55 * eff,
            ) if fits else 0.0
        else:
            tps = projected_tps(
                cluster["aggregate_bandwidth_gbs"], model_size,
                efficiency=0.6 * eff,
            ) if fits else 0.0

        results.append({
            "hardware": cluster["name"],
            "chip": f"{cluster['nodes']}x {cluster['node_chip']}",
            "bandwidth_gbs": cluster["aggregate_bandwidth_gbs"],
            "ram_gb": cluster["aggregate_ram_gb"],
            "model_size_gb": round(model_size, 1),
            "fits": fits,
            "projected_tps": round(tps, 1),
            "price_usd": cluster["price_usd"],
            "form_factor": "cluster",
        })

    return results


# ── Framework Comparison ──

FRAMEWORK_COMPARISON = {
    "Ollama (llama.cpp)": {
        "efficiency_multiplier": 1.0,  # baseline
        "pros": [
            "Easy setup (brew install ollama)",
            "Built-in model management and API server",
            "Cross-platform (Mac, Linux, Windows)",
            "Large model library (ollama.ai)",
        ],
        "cons": [
            "Not fully optimized for Apple Silicon Metal",
            "Higher memory overhead from server architecture",
            "Limited batching support",
        ],
        "best_for": "Getting started, team servers, API compatibility",
    },
    "MLX (Apple)": {
        "efficiency_multiplier": 1.4,  # 30-50% faster than Ollama
        "pros": [
            "Built by Apple specifically for Apple Silicon",
            "Native Metal GPU acceleration",
            "30-50% faster than llama.cpp on same hardware",
            "Lower memory overhead",
        ],
        "cons": [
            "macOS only",
            "Smaller model ecosystem",
            "More manual setup required",
            "Less mature tooling",
        ],
        "best_for": "Maximum performance on Mac, research, single-user",
    },
    "llama.cpp (direct)": {
        "efficiency_multiplier": 1.1,  # slightly better than Ollama
        "pros": [
            "Maximum control and configurability",
            "Supports custom quantization formats (GGUF)",
            "Active development community",
            "Lower overhead than Ollama wrapper",
        ],
        "cons": [
            "Manual model download and conversion",
            "Command-line interface only",
            "No built-in API server (separate llama-server)",
        ],
        "best_for": "Power users, custom setups, embedded use",
    },
    "exo (distributed)": {
        "efficiency_multiplier": 0.85,  # per-node, but aggregate is higher
        "pros": [
            "Split models across multiple Macs",
            "Near-linear scaling with Thunderbolt 5",
            "Run models too large for a single machine",
            "Automatic peer discovery",
        ],
        "cons": [
            "Requires multiple machines",
            "Network latency adds to TTFT",
            "Setup complexity",
            "Still experimental for production",
        ],
        "best_for": "Models >64GB, team infrastructure, scaling",
    },
}


# ── Quantization Impact ──

QUANTIZATION_INFO = {
    "FP16": {
        "bits_per_param": 16,
        "size_multiplier": 1.0,
        "quality_retention": 1.0,  # baseline
        "description": "Full precision. Maximum quality, maximum size.",
    },
    "Q8": {
        "bits_per_param": 8,
        "size_multiplier": 0.5,
        "quality_retention": 0.99,
        "description": "8-bit quantization. Negligible quality loss, half the size.",
    },
    "Q6_K": {
        "bits_per_param": 6.5,
        "size_multiplier": 0.41,
        "quality_retention": 0.98,
        "description": "6-bit with k-quant. Good balance of size and quality.",
    },
    "Q4_K_M": {
        "bits_per_param": 4.8,
        "size_multiplier": 0.30,
        "quality_retention": 0.95,
        "description": "4-bit with k-quant medium. Most common for local inference.",
    },
    "Q4_0": {
        "bits_per_param": 4,
        "size_multiplier": 0.25,
        "quality_retention": 0.92,
        "description": "Basic 4-bit. Smallest practical size.",
    },
    "Q2_K": {
        "bits_per_param": 2.6,
        "size_multiplier": 0.16,
        "quality_retention": 0.80,
        "description": "2-bit. Significant quality loss, last resort for huge models.",
    },
}


# ── Decision Matrix ──

def recommend_hardware(
    budget_usd: int,
    primary_use: str,  # "personal", "team", "lab"
    model_size_requirement: str,  # "small" (<14B), "medium" (14-70B), "large" (70B+)
    portability_needed: bool = False,
) -> list[dict]:
    """Recommend hardware based on budget, use case, and requirements.

    Returns ranked list of recommendations with reasoning.
    """
    recommendations = []

    size_min_ram = {"small": 16, "medium": 64, "large": 192}
    min_ram = size_min_ram.get(model_size_requirement, 64)

    for hw in HARDWARE_SPECS:
        if hw.price_usd > budget_usd and hw.price_usd > 0:
            continue
        if hw.max_ram_gb < min_ram:
            continue
        if portability_needed and hw.form_factor not in ("laptop",):
            continue

        score = 0
        reasons = []

        # Bandwidth score (most important)
        score += hw.memory_bandwidth_gbs / 10
        reasons.append(f"{hw.memory_bandwidth_gbs} GB/s bandwidth")

        # RAM headroom
        ram_headroom = hw.max_ram_gb - min_ram
        if ram_headroom > 64:
            score += 20
            reasons.append("Plenty of RAM headroom for future models")
        elif ram_headroom > 0:
            score += 10

        # Value (bandwidth per dollar)
        if hw.price_usd > 0:
            bw_per_dollar = hw.memory_bandwidth_gbs / hw.price_usd * 1000
            score += bw_per_dollar
            reasons.append(f"${hw.price_usd:,} investment")

        # Form factor bonus
        if primary_use == "team" and hw.form_factor in ("mini", "desktop"):
            score += 10
            reasons.append("Good form factor for always-on server")
        elif primary_use == "personal" and hw.form_factor == "laptop":
            score += 5
            reasons.append("Portable for personal use")

        recommendations.append({
            "hardware": hw.name,
            "chip": hw.chip,
            "score": round(score, 1),
            "price_usd": hw.price_usd,
            "bandwidth_gbs": hw.memory_bandwidth_gbs,
            "max_ram_gb": hw.max_ram_gb,
            "reasons": reasons,
        })

    # Also check clusters
    for cluster in CLUSTER_CONFIGS:
        if cluster["price_usd"] > budget_usd:
            continue
        if cluster["aggregate_ram_gb"] < min_ram:
            continue

        score = 0
        reasons = []
        score += cluster["aggregate_bandwidth_gbs"] / 10
        reasons.append(f"{cluster['aggregate_bandwidth_gbs']} GB/s aggregate bandwidth")
        reasons.append(f"{cluster['nodes']} nodes, scalable")
        reasons.append(f"${cluster['price_usd']:,} total")

        if primary_use in ("team", "lab"):
            score += 15
            reasons.append("Ideal for shared infrastructure")

        recommendations.append({
            "hardware": cluster["name"],
            "chip": f"{cluster['nodes']}x {cluster['node_chip']}",
            "score": round(score, 1),
            "price_usd": cluster["price_usd"],
            "bandwidth_gbs": cluster["aggregate_bandwidth_gbs"],
            "max_ram_gb": cluster["aggregate_ram_gb"],
            "reasons": reasons,
        })

    recommendations.sort(key=lambda x: x["score"], reverse=True)
    return recommendations
