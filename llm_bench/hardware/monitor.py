"""Hardware metrics capture for macOS Apple Silicon via powermetrics and sysctl."""

import asyncio
import platform
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HardwareProfile:
    """Static hardware information about the current machine."""
    chip: str = ""
    cpu_cores: int = 0
    gpu_cores: int = 0
    memory_gb: int = 0
    memory_bandwidth_gbs: float = 0.0
    os_version: str = ""
    model_identifier: str = ""


@dataclass
class HardwareSnapshot:
    """Point-in-time hardware metrics during inference."""
    gpu_utilization_pct: Optional[float] = None
    gpu_frequency_mhz: Optional[float] = None
    cpu_power_watts: Optional[float] = None
    gpu_power_watts: Optional[float] = None
    thermal_pressure: str = "nominal"  # nominal, moderate, heavy, critical
    memory_used_gb: Optional[float] = None
    timestamp: float = 0.0


@dataclass
class HardwareMetrics:
    """Aggregated hardware metrics from a monitoring session."""
    profile: HardwareProfile = field(default_factory=HardwareProfile)
    avg_gpu_utilization_pct: Optional[float] = None
    max_gpu_utilization_pct: Optional[float] = None
    avg_cpu_power_watts: Optional[float] = None
    avg_gpu_power_watts: Optional[float] = None
    peak_thermal_pressure: str = "nominal"
    snapshots_count: int = 0


# ── Bandwidth lookup for known Apple Silicon chips ──
BANDWIDTH_TABLE = {
    "Apple M1": 68.25,
    "Apple M1 Pro": 200.0,
    "Apple M1 Max": 400.0,
    "Apple M1 Ultra": 800.0,
    "Apple M2": 100.0,
    "Apple M2 Pro": 200.0,
    "Apple M2 Max": 400.0,
    "Apple M2 Ultra": 800.0,
    "Apple M3": 100.0,
    "Apple M3 Pro": 150.0,
    "Apple M3 Max": 400.0,
    "Apple M3 Ultra": 819.2,
    "Apple M4": 120.0,
    "Apple M4 Pro": 273.0,
    "Apple M4 Max": 546.0,
    "Apple M4 Ultra": 819.2,
}

THERMAL_SEVERITY = {"nominal": 0, "moderate": 1, "heavy": 2, "critical": 3}
THERMAL_NAMES = {0: "nominal", 1: "moderate", 2: "heavy", 3: "critical"}


def get_hardware_profile() -> HardwareProfile:
    """Detect the current machine's hardware profile using sysctl and system_profiler."""
    profile = HardwareProfile()

    if platform.system() != "Darwin":
        profile.os_version = platform.platform()
        return profile

    profile.os_version = platform.mac_ver()[0]

    try:
        # Get chip name
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        )
        profile.chip = result.stdout.strip()

        # Get CPU core count
        result = subprocess.run(
            ["sysctl", "-n", "hw.ncpu"],
            capture_output=True, text=True, timeout=5,
        )
        profile.cpu_cores = int(result.stdout.strip())

        # Get total memory
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5,
        )
        profile.memory_gb = int(result.stdout.strip()) // (1024 ** 3)

        # Get model identifier
        result = subprocess.run(
            ["sysctl", "-n", "hw.model"],
            capture_output=True, text=True, timeout=5,
        )
        profile.model_identifier = result.stdout.strip()

    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass

    # Get GPU cores via system_profiler (slower but accurate)
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True, text=True, timeout=10,
        )
        import json
        data = json.loads(result.stdout)
        displays = data.get("SPDisplaysDataType", [])
        for display in displays:
            cores_str = display.get("sppci_cores", "")
            if cores_str:
                # Parse "38" from "38" or "38-core"
                match = re.search(r"(\d+)", str(cores_str))
                if match:
                    profile.gpu_cores = int(match.group(1))
                break
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, KeyError):
        pass

    # Look up bandwidth from table (match longest name first to avoid
    # "Apple M2" matching before "Apple M2 Max")
    for chip_name, bw in sorted(BANDWIDTH_TABLE.items(), key=lambda x: len(x[0]), reverse=True):
        if chip_name in profile.chip:
            profile.memory_bandwidth_gbs = bw
            break

    return profile


def _parse_thermal_pressure() -> str:
    """Read thermal pressure from sysctl."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "kern.thermalstate"],
            capture_output=True, text=True, timeout=2,
        )
        state = int(result.stdout.strip())
        return THERMAL_NAMES.get(state, "nominal")
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        return "nominal"


def _parse_memory_pressure() -> Optional[float]:
    """Read memory usage via vm_stat."""
    try:
        result = subprocess.run(
            ["vm_stat"],
            capture_output=True, text=True, timeout=2,
        )
        # Parse pages
        page_size = 16384  # 16KB on Apple Silicon
        pages_active = 0
        pages_wired = 0
        for line in result.stdout.splitlines():
            if "Pages active" in line:
                match = re.search(r"(\d+)", line.split(":")[1])
                if match:
                    pages_active = int(match.group(1))
            elif "Pages wired" in line:
                match = re.search(r"(\d+)", line.split(":")[1])
                if match:
                    pages_wired = int(match.group(1))

        used_bytes = (pages_active + pages_wired) * page_size
        return round(used_bytes / (1024 ** 3), 1)
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        return None


class HardwareMonitor:
    """Monitors hardware metrics during LLM inference.

    Uses lightweight sysctl/vm_stat polling instead of powermetrics (which requires sudo).
    Captures thermal state, memory usage, and timing data without elevated permissions.
    """

    def __init__(self, sample_interval_ms: int = 2000):
        self.sample_interval = sample_interval_ms / 1000.0
        self._snapshots: list[HardwareSnapshot] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._profile = get_hardware_profile()

    async def start(self):
        """Start background monitoring."""
        self._running = True
        self._snapshots = []
        self._task = asyncio.create_task(self._monitor_loop())

    async def stop(self) -> HardwareMetrics:
        """Stop monitoring and return aggregated metrics."""
        self._running = False
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            self._task = None

        return self._aggregate()

    async def _monitor_loop(self):
        """Background loop that captures snapshots at regular intervals."""
        while self._running:
            snapshot = HardwareSnapshot(
                thermal_pressure=_parse_thermal_pressure(),
                memory_used_gb=_parse_memory_pressure(),
                timestamp=time.time(),
            )
            self._snapshots.append(snapshot)
            await asyncio.sleep(self.sample_interval)

    def _aggregate(self) -> HardwareMetrics:
        """Aggregate snapshots into summary metrics."""
        metrics = HardwareMetrics(profile=self._profile)
        metrics.snapshots_count = len(self._snapshots)

        if not self._snapshots:
            return metrics

        # Thermal pressure — report worst seen
        worst_thermal = 0
        for s in self._snapshots:
            severity = THERMAL_SEVERITY.get(s.thermal_pressure, 0)
            worst_thermal = max(worst_thermal, severity)
        metrics.peak_thermal_pressure = THERMAL_NAMES.get(worst_thermal, "nominal")

        # GPU utilization (from snapshots that have it)
        gpu_utils = [s.gpu_utilization_pct for s in self._snapshots if s.gpu_utilization_pct is not None]
        if gpu_utils:
            metrics.avg_gpu_utilization_pct = round(sum(gpu_utils) / len(gpu_utils), 1)
            metrics.max_gpu_utilization_pct = round(max(gpu_utils), 1)

        # Power (from snapshots that have it)
        cpu_powers = [s.cpu_power_watts for s in self._snapshots if s.cpu_power_watts is not None]
        if cpu_powers:
            metrics.avg_cpu_power_watts = round(sum(cpu_powers) / len(cpu_powers), 1)

        gpu_powers = [s.gpu_power_watts for s in self._snapshots if s.gpu_power_watts is not None]
        if gpu_powers:
            metrics.avg_gpu_power_watts = round(sum(gpu_powers) / len(gpu_powers), 1)

        return metrics


def hardware_profile_to_dict(profile: HardwareProfile) -> dict:
    """Convert hardware profile to a serializable dict."""
    return {
        "chip": profile.chip,
        "cpu_cores": profile.cpu_cores,
        "gpu_cores": profile.gpu_cores,
        "memory_gb": profile.memory_gb,
        "memory_bandwidth_gbs": profile.memory_bandwidth_gbs,
        "os_version": profile.os_version,
        "model_identifier": profile.model_identifier,
    }


def hardware_metrics_to_dict(metrics: HardwareMetrics) -> dict:
    """Convert hardware metrics to a serializable dict."""
    return {
        "profile": hardware_profile_to_dict(metrics.profile),
        "avg_gpu_utilization_pct": metrics.avg_gpu_utilization_pct,
        "max_gpu_utilization_pct": metrics.max_gpu_utilization_pct,
        "avg_cpu_power_watts": metrics.avg_cpu_power_watts,
        "avg_gpu_power_watts": metrics.avg_gpu_power_watts,
        "peak_thermal_pressure": metrics.peak_thermal_pressure,
        "snapshots_count": metrics.snapshots_count,
    }
