"""Batch processing pipeline for DREAMS spindle-detector excerpts."""

import glob
import os
import re

import mne
import numpy as np
import pandas as pd

from .detector import detect_events
from .utils import load_hypnogram, save_events_as_scoring
from .validate import validate

DATA_DIR = "DatabaseSpindles"
OUT_DIR = "results"

# Lower and higher detector thresholds.
R_A = 0.90
R_B = 0.95

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


def process_excerpt(excerpt_id: str) -> dict:
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
    all_events = detect_events(signal, fs, R_A, R_B)
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

    nrem_events = [
        event
        for event in freq_events
        if hypnogram[int(event["t1"] // 5.0)] in {0, 1, 2, 3}
    ]
    nrem_minutes = np.sum(np.isin(hypnogram, [0, 1, 2, 3])) * 5.0 / 60.0

    result["n_nrem_events"] = len(nrem_events)
    result["nrem_minutes"] = nrem_minutes
    result["nrem_spindle_rate"] = (
        len(nrem_events) / nrem_minutes if nrem_minutes > 0 else None
    )

    if os.path.exists(visual1) and os.path.exists(visual2):
        try:
            result["validation"] = validate(nrem_events, visual1, visual2)
        except Exception as e:  # noqa: BLE001
            result["validation"] = f"validate() failed: {e}"
    else:
        result["validation"] = "missing visual scoring file(s)"

    return result


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
    """Process all available excerpts and write a combined summary."""
    print("Hello from olbrich-achermann-spindle-detector! (batch mode)")

    excerpt_ids = find_excerpts(DATA_DIR)
    print(f"Found {len(excerpt_ids)} excerpt(s): {excerpt_ids}")

    all_results = []
    for excerpt_id in excerpt_ids:
        print(f"\n--- Processing excerpt{excerpt_id} ---")
        try:
            result = process_excerpt(excerpt_id)
        except Exception as e:  # noqa: BLE001
            result = {"excerpt": excerpt_id, "status": f"error: {e}"}
        all_results.append(result)

        short_status = result.get("status", "")
        if len(short_status) > 60:
            short_status = short_status[:57] + "..."
        print(f"  status={short_status} rate={result.get('nrem_spindle_rate')}")

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

    print("\n=== SUMMARY ACROSS ALL EXCERPTS ===")
    display_cols = [
        column
        for column in [
            "excerpt",
            "status",
            "channel",
            "n_total_events",
            "n_sigma_events",
            "n_nrem_events",
            "nrem_spindle_rate",
            "expert1_precision",
            "expert1_recall",
            "expert1_f1",
            "expert2_precision",
            "expert2_recall",
            "expert2_f1",
        ]
        if column in summary_flat.columns
    ]
    print(summary_flat[display_cols].to_string(index=False))
    print(f"\nSaved full summary to {summary_path}")

    ok = summary_flat[summary_flat["nrem_spindle_rate"].notna()]
    if not ok.empty:
        print("\n=== NREM SPINDLE RATE STATS (across excerpts with a rate) ===")
        print(pd.Series(ok["nrem_spindle_rate"]).describe())
    if not ok.empty and "expert2_f1" in ok.columns:
        print("\n=== F1 vs Expert 2 (across excerpts with validation) ===")
        print(pd.Series(ok["expert2_f1"]).describe())


if __name__ == "__main__":
    main()
