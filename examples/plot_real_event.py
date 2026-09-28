"""Plot one real DREAMS detection with its EEG and tracked AR poles.

Example:
    uv run --extra plots python examples/plot_real_event.py
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

from olbrich_achermann_spindle_detector.detector import (
    AR_ORDER,
    FINE_SCAN_STEP_SECONDS,
    _match_oscillators,
    detect_events,
    get_oscillators,
)

DATA_DIR = Path("DatabaseSpindles")
DEFAULT_EDF = DATA_DIR / "excerpt6.edf"

DEFAULT_EVENT_START = 951.3175
R_A = 0.85
R_B = 0.90
PLOT_SECONDS = 3.0
WINDOW_SECONDS = 1.0


def track_poles(signal, fs, clip_start, plot_start, plot_end):
    """Estimate and associate AR poles over overlapping one-second windows."""
    step = FINE_SCAN_STEP_SECONDS
    max_window_start = min(len(signal) / fs - WINDOW_SECONDS, plot_end - clip_start)
    window_starts = np.arange(0.0, max_window_start + step / 2, step)
    tracks = {}
    previous = []
    next_track_id = 0

    for window_start in window_starts:
        sample_start = round(window_start * fs)
        segment = signal[sample_start : sample_start + round(WINDOW_SECONDS * fs)]
        if len(segment) != round(WINDOW_SECONDS * fs):
            continue

        oscillators = get_oscillators(segment, p=AR_ORDER, fs=fs)
        current = []
        # Use the same one-to-one nearest-frequency matching as the detector.
        for track, oscillator in _match_oscillators(previous, oscillators):
            if "track_id" not in track:
                track_id = next_track_id
                next_track_id += 1
                track["track_id"] = track_id
                tracks[track_id] = {"time": [], "r": [], "frequency": []}
            else:
                track_id = track["track_id"]

            plot_track = tracks[track_id]
            radius = float(oscillator["r"])
            plot_track["time"].append(clip_start + window_start)
            plot_track["r"].append(radius)
            plot_track["frequency"].append(float(oscillator["frequency"]))

            previous_radius = track["previous_r"]
            if not track["event_started"]:
                if (
                    previous_radius is not None
                    and previous_radius <= R_B
                    and radius > R_B
                ):
                    track["event_started"] = True
            elif radius < R_A:
                continue

            track["previous_r"] = radius
            track["frequency"] = oscillator["frequency"]
            current.append(track)

        previous = current

    return tracks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edf", type=Path, default=DEFAULT_EDF)
    parser.add_argument("--event-start", type=float, default=DEFAULT_EVENT_START)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/excerpt6_matched_event.png"),
    )
    args = parser.parse_args()

    if not args.edf.exists():
        raise FileNotFoundError(f"EDF file not found: {args.edf}")

    plot_start = max(0.0, args.event_start - 1.0)
    plot_end = plot_start + PLOT_SECONDS
    # Include the second before the plotted window so coarse-to-fine scanning
    # retains the detector's one-second look-back context.
    clip_start = max(0, math.floor(plot_start) - 1)
    clip_end = plot_end + WINDOW_SECONDS + 0.1

    raw = mne.io.read_raw_edf(args.edf, preload=False, verbose="ERROR")
    channel = next((name for name in ("C3-A1", "CZ-A1") if name in raw.ch_names), None)
    if channel is None:
        raise ValueError(f"No supported EEG channel found; available: {raw.ch_names}")

    fs = int(raw.info["sfreq"])
    first_sample = round(clip_start * fs)
    last_sample = min(round(clip_end * fs), raw.n_times)
    signal = raw.get_data(picks=channel, start=first_sample, stop=last_sample)[0]
    del raw

    # The clip begins on an integer-second boundary to preserve the detector's
    # one-second coarse-scan alignment with the recording.
    events = detect_events(signal, fs, R_A, R_B, pole_mode="all")
    global_events = [
        {
            **event,
            "t1": float(event["t1"]) + clip_start,
            "t2": float(event["t2"]) + clip_start,
            "time": float(event["time"]) + clip_start,
        }
        for event in events
    ]
    selected = min(global_events, key=lambda event: abs(event["t1"] - args.event_start))

    tracks = track_poles(signal, fs, clip_start, plot_start, plot_end)
    local_time = np.arange(len(signal)) / fs + clip_start
    visible = (local_time >= plot_start) & (local_time <= plot_end)
    figure, (signal_axis, radius_axis, frequency_axis) = plt.subplots(
        3,
        1,
        figsize=(12, 8),
        sharex=True,
        constrained_layout=True,
    )

    signal_axis.plot(local_time[visible], signal[visible], color="black", linewidth=0.8)
    signal_axis.set_ylabel("Amplitude (V)")
    signal_axis.set_title(f"DREAMS excerpt 6 — {channel}, {fs} Hz")
    signal_axis.grid(True, alpha=0.25)

    colors = plt.get_cmap("tab10")
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
    radius_axis.set_ylim(bottom=0.7)
    radius_axis.legend(loc="upper right", ncols=2, fontsize="small")
    radius_axis.grid(True, alpha=0.25)

    frequency_axis.axhspan(
        11.5, 16.0, color="tab:green", alpha=0.10, label="Sigma band"
    )
    frequency_axis.set_ylabel("Frequency (Hz)")
    frequency_axis.set_xlabel("Time in excerpt (s)")
    frequency_axis.set_xlim(plot_start, plot_end)
    frequency_axis.set_ylim(0, 20)
    frequency_axis.legend(loc="upper right", ncols=2, fontsize="small")
    frequency_axis.grid(True, alpha=0.25)

    for axis in (signal_axis, radius_axis, frequency_axis):
        axis.axvline(
            selected["t1"],
            color="tab:blue",
            linestyle=":",
            linewidth=1.5,
            label="Detected event boundary" if axis is signal_axis else None,
        )
        axis.axvline(
            selected["t2"],
            color="tab:blue",
            linestyle=":",
            linewidth=1.5,
        )

    signal_axis.legend(loc="upper right")
    figure.suptitle(f"Detected event (thresholds {R_A}/{R_B})")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"Channel: {channel}; sampling rate: {fs} Hz")
    print(f"Detector event: {selected['t1']:.4f}–{selected['t2']:.4f} s")
    print(f"Plot saved to {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
