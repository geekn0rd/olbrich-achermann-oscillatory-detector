"""Batch processing pipeline for DREAMS spindle-detector excerpts."""

import argparse
import contextlib
import glob
import os
import re
from typing import Literal

import mne
import numpy as np
import pandas as pd
from tqdm import tqdm

from .detector import detect_events
from .utils import load_hypnogram, save_events_as_scoring
from .validate import evaluate, load_expert_events

DATA_DIR = "DatabaseSpindles"
OUT_DIR = "results"

# Baseline lower and higher detector thresholds.
R_A = 0.90
R_B = 0.95

# Candidate threshold pairs used for per-excerpt/per-expert calibration.
# Every pair satisfies R_A < R_B, as required by the detector state machine.
THRESHOLD_PAIRS = (
    (0.85, 0.90),
    (0.88, 0.92),
    (0.90, 0.95),
    (0.92, 0.96),
    (0.94, 0.98),
    (0.96, 0.99),
)

FREQ_LOW = 11.5
FREQ_HIGH = 16.0

# DREAMS excerpts do not all use the same label for the central derivation.
CHANNEL_CANDIDATES = ["C3-A1", "CZ-A1"]


def find_excerpts(data_dir: str) -> list[str]:
    """Return sorted excerpt IDs found in ``data_dir``."""
    edf_files = glob.glob(os.path.join(data_dir, "excerpt*.edf"))
    ids = []
    for filename in edf_files:
        match = re.search(r"excerpt(\d+)\.edf$", os.path.basename(filename))
        if match:
            ids.append(match.group(1))
    return sorted(ids, key=int)


def process_excerpt(excerpt_id: str, pole_mode: Literal["max", "all"] = "all") -> dict:
    """Run detection, NREM-rate calculation, and optional validation."""
    edf_file = os.path.join(DATA_DIR, f"excerpt{excerpt_id}.edf")
    scoring_file = os.path.join(DATA_DIR, f"Hypnogram_excerpt{excerpt_id}.txt")
    visual1 = os.path.join(DATA_DIR, f"Visual_scoring1_excerpt{excerpt_id}.txt")
    visual2 = os.path.join(DATA_DIR, f"Visual_scoring2_excerpt{excerpt_id}.txt")

    result = {
        "excerpt": excerpt_id,
        "status": "ok",
        "channel": None,
        "n_total_events": None,
        "n_sigma_events": None,
        "n_nrem_events": None,
        "nrem_minutes": None,
        "nrem_spindle_rate": None,
        "validation": None,
    }

    if not os.path.exists(edf_file):
        result["status"] = "missing edf"
        return result

    raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
    fs = int(raw.info["sfreq"])

    channel = next((name for name in CHANNEL_CANDIDATES if name in raw.ch_names), None)
    if channel is None:
        result["status"] = "no matching channel"
        return result
    result["channel"] = channel

    signal = raw.get_data(picks=channel)[0]
    all_events = detect_events(signal, fs, R_A, R_B, pole_mode=pole_mode)
    result["n_total_events"] = len(all_events)

    freq_events = [
        event for event in all_events if FREQ_LOW <= event["frequency"] <= FREQ_HIGH
    ]
    result["n_sigma_events"] = len(freq_events)

    os.makedirs(OUT_DIR, exist_ok=True)
    save_events_as_scoring(
        all_events,
        os.path.join(OUT_DIR, f"algorithm_scoring_excerpt{excerpt_id}.txt"),
    )

    if not os.path.exists(scoring_file):
        result["status"] = "missing hypnogram"
        return result

    hypnogram = load_hypnogram(scoring_file)

    nrem_events = _nrem_events(all_events, hypnogram)
    nrem_minutes = np.sum(np.isin(hypnogram, [0, 1, 2, 3])) * 5.0 / 60.0

    result["n_nrem_events"] = len(nrem_events)
    result["nrem_minutes"] = nrem_minutes
    result["nrem_spindle_rate"] = (
        len(nrem_events) / nrem_minutes if nrem_minutes > 0 else None
    )

    validation = {}
    candidate_events_by_threshold = {(R_A, R_B): all_events}
    for expert, filename in (("expert1", visual1), ("expert2", visual2)):
        if not os.path.exists(filename):
            continue

        expert_events = load_expert_events(filename)
        durations = np.array(
            [event["duration"] for event in expert_events], dtype=float
        )
        result[f"{expert}_spindle_rate"] = (
            len(expert_events) / nrem_minutes if nrem_minutes > 0 else None
        )
        result[f"{expert}_duration_count"] = len(durations)
        result[f"{expert}_duration_mean"] = (
            float(np.mean(durations)) if durations.size else None
        )
        result[f"{expert}_duration_std"] = (
            float(np.std(durations)) if durations.size else None
        )
        result[f"{expert}_duration_min"] = (
            float(np.min(durations)) if durations.size else None
        )
        result[f"{expert}_duration_median"] = (
            float(np.median(durations)) if durations.size else None
        )
        result[f"{expert}_duration_max"] = (
            float(np.max(durations)) if durations.size else None
        )

        best = None
        for candidate_r_a, candidate_r_b in THRESHOLD_PAIRS:
            threshold_pair = (candidate_r_a, candidate_r_b)
            if threshold_pair not in candidate_events_by_threshold:
                candidate_events_by_threshold[threshold_pair] = detect_events(
                    signal,
                    fs,
                    candidate_r_a,
                    candidate_r_b,
                    pole_mode=pole_mode,
                )
            candidate_events = candidate_events_by_threshold[threshold_pair]
            candidate_nrem_events = _nrem_events(candidate_events, hypnogram)
            metrics = evaluate(candidate_nrem_events, expert_events)
            score = (metrics["f1"], metrics["mean_iou"])
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "r_a": candidate_r_a,
                    "r_b": candidate_r_b,
                    "events": candidate_events,
                    "metrics": metrics,
                }

        assert best is not None
        result[f"{expert}_r_a"] = best["r_a"]
        result[f"{expert}_r_b"] = best["r_b"]
        validation[expert] = best["metrics"]
        save_events_as_scoring(
            best["events"],
            os.path.join(
                OUT_DIR, f"algorithm_scoring_excerpt{excerpt_id}_{expert}.txt"
            ),
        )

    result["validation"] = validation or "missing visual scoring file(s)"
    return result


def _nrem_events(all_events: list[dict], hypnogram: np.ndarray) -> list[dict]:
    """Keep sigma-band events that start during an NREM epoch."""
    freq_events = [
        event for event in all_events if FREQ_LOW <= event["frequency"] <= FREQ_HIGH
    ]
    return [
        event
        for event in freq_events
        if hypnogram[int(event["t1"] // 5.0)] in {0, 1, 2, 3}
    ]


def _flatten_validation(validation: object) -> dict:
    """Flatten expert validation metrics into summary columns."""
    if not isinstance(validation, dict):
        return {}

    flattened = {}
    for expert, metrics in validation.items():
        if isinstance(metrics, dict):
            for metric, value in metrics.items():
                flattened[f"{expert}_{metric}"] = value
    return flattened


def main() -> None:
    """Process all available excerpts and save one combined summary."""
    parser = argparse.ArgumentParser(
        description="Run spindle detection on DREAMS excerpts"
    )
    parser.add_argument(
        "--pole-mode",
        choices=("max", "all"),
        default="all",
        help="track only the strongest pole in each window or track all poles (default: all)",
    )
    args = parser.parse_args()

    excerpt_ids = find_excerpts(DATA_DIR)

    all_results = []
    with tqdm(excerpt_ids, desc="Processing excerpts", unit="excerpt") as progress:
        for excerpt_id in progress:
            try:
                with (
                    open(os.devnull, "w") as devnull,
                    contextlib.redirect_stdout(devnull),
                    contextlib.redirect_stderr(devnull),
                ):
                    result = process_excerpt(excerpt_id, pole_mode=args.pole_mode)
            except Exception as e:  # noqa: BLE001
                result = {
                    "excerpt": excerpt_id,
                    "status": f"error: {e}",
                    "validation": None,
                }
            all_results.append(result)

    summary = pd.DataFrame(all_results)
    validation = pd.DataFrame(
        [_flatten_validation(value) for value in summary["validation"]]
    )
    summary_flat = pd.concat(
        [summary.drop(columns=["validation"]), validation],
        axis=1,
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    summary_path = os.path.join(OUT_DIR, "summary.csv")
    summary_flat.to_csv(summary_path, index=False)


if __name__ == "__main__":
    main()
