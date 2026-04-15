"""
Performance telemetry and self-calibration for cinematography-analysis-tools.

Stores structured run logs at ~/.cache/cinematography-tools/perf_log.jsonl
Each line is a JSON object with timing breakdowns, hardware context, and
estimation accuracy. The correction factor self-calibrates over historical runs.
"""

import json
import os
import platform
import time
from pathlib import Path
from typing import Dict, Optional

# Constants
PERF_LOG_DIR = Path.home() / ".cache" / "cinematography-tools"
PERF_LOG_PATH = PERF_LOG_DIR / "perf_log.jsonl"


def _get_hardware_context() -> Dict:
    """Capture hardware info for the run log."""
    ctx = {
        "arch": platform.machine(),
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.mac_ver()[0]}",
    }
    try:
        import torch
        ctx["torch"] = torch.__version__
        ctx["mps"] = torch.backends.mps.is_available()
    except Exception:
        pass
    return ctx


class RunTimer:
    """Context-manager stopwatch for timing pipeline phases.

    Usage:
        timer = RunTimer()
        with timer.phase("model_load"):
            model = load_model()
        with timer.phase("inference"):
            results = predict(...)
        timer.finalize(video_duration=120.0, frames=240, mode="Cinematic Standard")
    """

    def __init__(self):
        self._phases: Dict[str, float] = {}
        self._current_phase: Optional[str] = None
        self._phase_start: float = 0.0
        self._wall_start: float = time.time()

    class _PhaseCtx:
        def __init__(self, timer: "RunTimer", name: str):
            self._timer = timer
            self._name = name

        def __enter__(self):
            self._start = time.time()
            return self

        def __exit__(self, *exc):
            elapsed = time.time() - self._start
            self._timer._phases[self._name] = round(elapsed, 3)

    def phase(self, name: str) -> "_PhaseCtx":
        return self._PhaseCtx(self, name)

    def finalize(
        self,
        video_duration: float,
        frames: int,
        mode: str,
        est_time: float = 0.0,
        extra: Optional[Dict] = None,
    ):
        """Write the completed run to the telemetry log."""
        wall_time = round(time.time() - self._wall_start, 3)

        entry = {
            "timestamp": time.time(),
            "video_duration_s": round(video_duration, 2),
            "frames_processed": frames,
            "mode": mode,
            "est_time_s": round(est_time, 2),
            "actual_time_s": wall_time,
            "error_ratio": round(wall_time / est_time, 3) if est_time > 0 else None,
            "throughput_fps": round(frames / wall_time, 1) if wall_time > 0 else 0,
            "phases": self._phases,
            "hardware": _get_hardware_context(),
        }
        if extra:
            entry["extra"] = extra

        try:
            PERF_LOG_DIR.mkdir(parents=True, exist_ok=True)
            with open(PERF_LOG_PATH, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"⚠️ Performance logging failed: {e}")

        # Print a human-readable summary to stdout
        print(f"\n{'='*60}")
        print(f"  RUN COMPLETE — {mode}")
        print(f"{'='*60}")
        print(f"  Video:       {video_duration:.0f}s ({video_duration/60:.1f} min)")
        print(f"  Frames:      {frames:,}")
        print(f"  Wall time:   {wall_time:.1f}s ({wall_time/60:.1f} min)")
        print(f"  Throughput:  {frames/wall_time:.1f} frames/sec" if wall_time > 0 else "")
        if est_time > 0:
            print(f"  Estimated:   {est_time:.1f}s  (off by {abs(wall_time - est_time):.1f}s)")
        print(f"  Phases:")
        for name, dur in self._phases.items():
            pct = (dur / wall_time * 100) if wall_time > 0 else 0
            print(f"    {name:20s} {dur:7.1f}s  ({pct:4.1f}%)")
        overhead = wall_time - sum(self._phases.values())
        if overhead > 0.5:
            print(f"    {'(overhead)':20s} {overhead:7.1f}s  ({overhead/wall_time*100:4.1f}%)")
        print(f"{'='*60}\n")


# Legacy interface — kept for backward compatibility
def log_performance(video_duration: float, fps: float, est_time: float, actual_time: float, mode_name: str):
    """Log a completed run (legacy format). Prefer RunTimer for new code."""
    try:
        PERF_LOG_DIR.mkdir(parents=True, exist_ok=True)

        entry = {
            "timestamp": time.time(),
            "video_duration_s": round(video_duration, 2),
            "fps": fps,
            "est_time_s": round(est_time, 2),
            "actual_time_s": round(actual_time, 2),
            "mode": mode_name,
            "error_ratio": round(actual_time / est_time if est_time > 0 else 1.0, 3)
        }

        with open(PERF_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")

    except Exception as e:
        print(f"⚠️ Performance logging failed: {e}")


def get_correction_factor() -> float:
    """Calculate the hardware correction factor from recent run history.

    Returns the average (actual / estimated) ratio from the last 10 runs.
    Values < 1.0 mean the estimator overestimates (hardware is faster than predicted).
    """
    if not PERF_LOG_PATH.exists():
        return 1.0

    try:
        with open(PERF_LOG_PATH, "r") as f:
            lines = f.readlines()

        ratios = []
        for line in lines[-10:]:
            if line.strip():
                entry = json.loads(line)
                r = entry.get("error_ratio")
                if r is not None and r > 0:
                    ratios.append(r)

        return sum(ratios) / len(ratios) if ratios else 1.0

    except Exception as e:
        print(f"⚠️ Error calculating performance correction: {e}")
        return 1.0


def get_perf_summary() -> str:
    """Get a human-readable status string based on historical accuracy."""
    if not PERF_LOG_PATH.exists():
        return "Baseline (Initial Calibration)"

    try:
        with open(PERF_LOG_PATH, "r") as f:
            count = sum(1 for line in f if line.strip())

        if count == 0:
            return "Baseline"
        elif count < 5:
            factor = get_correction_factor()
            return f"Calibrating ({count} runs, correction={factor:.2f}×)"
        else:
            factor = get_correction_factor()
            return f"Tuned ({count} runs, correction={factor:.2f}×)"
    except Exception:
        return "Baseline"
