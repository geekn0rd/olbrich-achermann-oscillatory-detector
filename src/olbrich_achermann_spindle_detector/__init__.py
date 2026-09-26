import mne
import numpy as np
import pandas as pd

from olbrich_achermann_spindle_detector.detector import detect_events

from .utils import calculate_nrem_spindle_rate, load_hypnogram, save_events_as_scoring
from .validate import validate


def main() -> None:
    print("Hello from olbrich-achermann-spindle-detector!")
    # lower threshold
    r_a = 0.90
    # higher threshold
    r_b = 0.95

    edf_file = "DatabaseSpindles/excerpt1.edf"
    scoring_file = "DatabaseSpindles/Hypnogram_excerpt1.txt"

    raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
    hypnogram = load_hypnogram(scoring_file)

    fs = int(raw.info["sfreq"])

    print("Sampling frequency:", fs)
    print("Channels:", raw.ch_names)

    signal = raw.get_data(picks=["C3-A1"])[0]
    all_events = detect_events(signal, fs, r_a, r_b)


    nrem_rate = calculate_nrem_spindle_rate(
        all_events,
        hypnogram,
    )

    save_events_as_scoring(
        all_events,
        "algorithm_scoring_excerpt1.txt",
    )

    print(
        f"NREM spindle rate: "
        f"{nrem_rate:.3f} spindles/min"
    )

    freq_events = [
        e for e in all_events
        if 11.5 <= e["frequency"] <= 16.0
    ]

    gaps = []
    for i in range(1, len(freq_events)):
        previous = freq_events[i - 1]
        current = freq_events[i]

        previous_end = previous["t1"] + previous["duration"]
        gap = current["t1"] - previous_end

        if gap >= 0:
            gaps.append(gap)

    gaps = np.array(gaps)

    print("Number of gaps:", len(gaps))

    for threshold in [0.05, 0.10, 0.20, 0.25, 0.50, 0.75, 1.0]:
        print(
            f"gap <= {threshold:.2f}s: "
            f"{np.sum(gaps <= threshold)}"
        )


    nrem_events = [
        e for e in freq_events
        if hypnogram[int(e["t1"] // 5.0)] in {0, 1, 2, 3}
    ]

    nrem_minutes = (
        np.sum(np.isin(hypnogram, [0, 1, 2, 3])) * 5.0 / 60.0
    )

    print("Total detected events:", len(all_events))
    print("11.5–16 Hz events:", len(freq_events))
    print("11.5–16 Hz + NREM events:", len(nrem_events))
    print("NREM minutes:", nrem_minutes)
    print("Rate:", len(nrem_events) / nrem_minutes)

    validate(nrem_events, "/Users/a2m/Code/olbrich-achermann-spindle-detector/DatabaseSpindles/Visual_scoring1_excerpt1.txt", "/Users/a2m/Code/olbrich-achermann-spindle-detector/DatabaseSpindles/Visual_scoring2_excerpt1.txt")


    sigma = pd.DataFrame(freq_events)

    print("\n=== SIGMA EVENTS ===")

    print("\nDuration:")
    print(sigma["duration"].describe())

    print("\nMax R:")
    print(sigma["r"].describe())

    print("\nPeak frequency:")
    print(sigma["frequency"].describe())


    print("\n=== DURATION BINS ===")

    print(
        pd.cut(
            sigma["duration"],
            bins=[0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0],
            include_lowest=True
        ).value_counts().sort_index()
    )

    print("\n=== RMAX BINS ===")

    print(
        pd.cut(
            sigma["r"],
            bins=[0.95, 0.96, 0.97, 0.98, 0.99, 1.0],
            include_lowest=True
        ).value_counts().sort_index()
    )

    sigma_events = sorted(freq_events, key=lambda e: e["time"])

    intervals = np.array([
        sigma_events[i]["time"] -
        sigma_events[i - 1]["time"]
        for i in range(1, len(sigma_events))
    ])

    print("\n=== SIGMA INTER-EVENT INTERVALS ===")
    print("N:", len(intervals))
    print(pd.Series(intervals).describe())

    for threshold in [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
        print(
            f"interval <= {threshold:.2f}s: "
            f"{np.sum(intervals <= threshold)}"
        )

    sigma_events = sorted(freq_events, key=lambda e: e["t1"])

    overlaps = []

    for i in range(1, len(sigma_events)):
        prev = sigma_events[i - 1]
        curr = sigma_events[i]

        prev_end = prev["t2"]

        overlap = prev_end - curr["t1"]

        if overlap > 0:
            overlaps.append(overlap)

    print("\n=== OVERLAPPING SIGMA EVENTS ===")
    print("Number:", len(overlaps))
    print(pd.Series(overlaps).describe())

    for i in range(1, 10):
        prev = sigma_events[i - 1]
        curr = sigma_events[i]

        overlap = prev["t2"] - curr["t1"]

        if overlap > 0:
            print(
                f"prev: {prev['t1']:.4f}-{prev['t2']:.4f} "
                f"f={prev['frequency']:.3f} "
                f"r={prev['r']:.4f}\n"
                f"curr: {curr['t1']:.4f}-{curr['t2']:.4f} "
                f"f={curr['frequency']:.3f} "
                f"r={curr['r']:.4f}\n"
                f"overlap={overlap:.4f}s\n"
            )

