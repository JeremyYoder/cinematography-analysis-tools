"""
Performance telemetry and self-calibration for cinematography-analysis-tools.
"""

import json
import os
import time
from pathlib import Path
from typing import List, Dict, Optional

# Constants
PERF_LOG_DIR = Path.home() / ".cache" / "cinematography-tools"
PERF_LOG_PATH = PERF_LOG_DIR / "perf_log.jsonl"

def log_performance(video_duration: float, fps: float, est_time: float, actual_time: float, mode_name: str):
    """Log a completed run to the local telemetry file."""
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
    """Calculate the hardware correction factor based on recent performance history."""
    if not PERF_LOG_PATH.exists():
        return 1.0
        
    try:
        history = []
        with open(PERF_LOG_PATH, "r") as f:
            lines = f.readlines()
            # Look at last 10 entries
            for line in lines[-10:]:
                if line.strip():
                    history.append(json.loads(line))
        
        if not history:
            return 1.0
            
        # We average the error ratios (actual / est)
        ratios = [h["error_ratio"] for h in history if "error_ratio" in h]
        if not ratios:
            return 1.0
            
        return sum(ratios) / len(ratios)
        
    except Exception as e:
        print(f"⚠️ Error calculating performance correction: {e}")
        return 1.0

def get_perf_summary() -> str:
    """Get a status message based on historical accuracy data."""
    if not PERF_LOG_PATH.exists():
        return "Baseline (Initial Calibration)"
        
    try:
        with open(PERF_LOG_PATH, "r") as f:
            count = len(f.readlines())
        
        if count == 0:
            return "Baseline"
        elif count < 5:
            return f"Synchronizing ({count} runs logged)"
        else:
            return f"Optimized (Based on {count} recent runs)"
    except:
        return "Baseline"
