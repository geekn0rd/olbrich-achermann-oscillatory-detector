"""Plot a synthetic three-second oscillatory event using AR pole estimates.

Run with ``uv run --extra plots python examples/synthetic_demo.py``.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from olbrich_achermann_spindle_detector.detector import (
    AR_ORDER,
    FINE_SCAN_STEP_SECONDS,
    get_oscillators,
)

FS = 128
DURATION_SECONDS = 3.0
WINDOW_SECONDS = 1.0
R_A = 0.90
R_B = 0.95


def make_signal():
    """Create noise with a smoothly waxing-and-waning 13-Hz oscillation."""
    time = np.arange(round(FS * DURATION_SECONDS)) / FS
    rng = np.random.default_rng(7)
    noise = 0.35 * rng.standard_normal(time.size)

    # Raised-cosine envelope: the synthetic spindle lasts from 0.75 to 2.35 s.
    onset, offset = 0.75, 2.35
    envelope = np.zeros_like(time)
    active = (time >= onset) & (time <= offset)
    phase = (time[active] - onset) / (offset - onset)
    envelope[active] = np.sin(np.pi * phase) ** 2

    spindle = 2.0 * envelope * np.sin(2 * np.pi * 13.0 * time)
    return time, noise + spindle


def estimate_poles(signal):
    """Estimate poles and associate them across adjacent fine-scan windows."""
    times = np.arange(
        0.0,
        DURATION_SECONDS - WINDOW_SECONDS + FINE_SCAN_STEP_SECONDS,
        FINE_SCAN_STEP_SECONDS,
    )
    tracks = {}
    previous = []
    next_track_id = 0

    for window_time in times:
        start = round(window_time * FS)
        segment = signal[start : start + round(WINDOW_SECONDS * FS)]
        if len(segment) != round(WINDOW_SECONDS * FS):
            continue

        oscillators = get_oscillators(segment, p=AR_ORDER, fs=FS)
        current = []

        if previous and oscillators:
            costs = np.array(
                [
                    [
                        abs(old["frequency"] - oscillator["frequency"])
                        for oscillator in oscillators
                    ]
                    for old in previous
                ]
            )
            old_indices, current_indices = linear_sum_assignment(costs)
            assignments = dict(zip(current_indices, old_indices, strict=True))
        else:
            assignments = {}

        for index, oscillator in enumerate(oscillators):
            if index in assignments:
                track_id = previous[assignments[index]]["track_id"]
            else:
                track_id = next_track_id
                next_track_id += 1
                tracks[track_id] = {"time": [], "r": [], "frequency": []}

            track = tracks[track_id]
            track["time"].append(window_time + WINDOW_SECONDS / 2)
            track["r"].append(float(oscillator["r"]))
            track["frequency"].append(float(oscillator["frequency"]))
            current.append({"track_id": track_id, "frequency": oscillator["frequency"]})

        previous = current

    return tracks


def main():
    time, signal = make_signal()
    tracks = estimate_poles(signal)
    colors = plt.get_cmap("tab10")

    figure, (signal_axis, radius_axis, frequency_axis) = plt.subplots(
        3,
        1,
        figsize=(11, 8),
        sharex=True,
        constrained_layout=True,
    )

    signal_axis.plot(time, signal, color="black", linewidth=0.8)
    signal_axis.set_ylabel("Amplitude")
    signal_axis.set_title("Synthetic 3-second signal and AR(8) pole estimates")
    signal_axis.grid(True, alpha=0.25)

    for track_id, track in tracks.items():
        color = colors(track_id % colors.N)
        label = f"Pole {track_id + 1}"
        radius_axis.plot(
            track["time"],
            track["r"],
            "o-",
            color=color,
            markersize=3,
            linewidth=1,
            label=label,
        )
        frequency_axis.plot(
            track["time"],
            track["frequency"],
            "o-",
            color=color,
            markersize=3,
            linewidth=1,
            label=label,
        )

    radius_axis.axhline(R_A, color="tab:orange", linestyle="--", label=f"$r_a$ = {R_A}")
    radius_axis.axhline(R_B, color="tab:red", linestyle="--", label=f"$r_b$ = {R_B}")
    radius_axis.set_ylabel("Pole radius $r$")
    radius_axis.set_ylim(bottom=0)
    radius_axis.legend(loc="best")
    radius_axis.grid(True, alpha=0.25)

    frequency_axis.axhspan(
        11.5, 16.0, color="tab:green", alpha=0.10, label="Sigma band"
    )
    frequency_axis.set_ylabel("Frequency (Hz)")
    frequency_axis.set_xlabel("Time (s)")
    frequency_axis.set_xlim(0, DURATION_SECONDS)
    frequency_axis.set_ylim(0, FS / 2)
    frequency_axis.legend(loc="best")
    frequency_axis.grid(True, alpha=0.25)

    figure.suptitle(
        "Synthetic demonstration only — not a reproduction of the paper's EEG figure",
        fontsize=10,
    )
    plt.show()


if __name__ == "__main__":
    main()
