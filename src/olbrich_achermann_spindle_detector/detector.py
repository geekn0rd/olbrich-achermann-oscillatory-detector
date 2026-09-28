"""Autoregressive oscillator-based oscillatory-event detection.

The implementation follows the paper's two-stage scan: a coarse one-second
scan identifies candidate regions, then a fine scan tracks the oscillator
poles until their event termination criteria are met.
"""

from typing import Literal, cast

import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm

AR_ORDER = 8
FINE_SCAN_STEP_SECONDS = 1 / 16


def get_oscillators(segment, p, fs):
    """Fit an AR(p) model and return its positive-frequency complex poles."""
    coefficients_ar, _ = sm.regression.linear_model.burg(segment, order=p)

    # z^p - a[0]z^(p-1) - ... - a[p-1] = 0
    coefficients = np.concatenate([[1.0], -coefficients_ar])
    roots = np.roots(coefficients)

    oscillators = []
    for pole in roots:
        # Keep one pole from each complex-conjugate pair.
        if pole.imag <= 0:
            continue

        radius = np.abs(pole)
        angle = np.angle(pole)
        frequency = angle * fs / (2 * np.pi)

        # if not (11.5 <= frequency <= 16.0):
        #     continue

        oscillators.append(
            {
                "pole": pole,
                "r": radius,
                "theta": angle,
                "frequency": frequency,
            }
        )

    return oscillators


def _new_track(oscillator):
    """Create state for one oscillator track."""
    return {
        "frequency": oscillator["frequency"],
        "previous_r": None,
        "event_started": False,
        "t1": None,
        "last_rb_crossing": None,
        "max_r": -np.inf,
        "peak_time": None,
        "peak_frequency": None,
    }


def _append_completed_event(events, track):
    """Append a completed event when the track has a valid end crossing."""
    t1 = track["t1"]
    t2 = track["last_rb_crossing"]

    if t2 is None:
        return

    events.append(
        {
            "t1": t1,
            "t2": t2,
            "time": track["peak_time"],
            "frequency": track["peak_frequency"],
            "r": track["max_r"],
            "duration": cast(float, t2) - cast(float, t1) + 1.0,
        }
    )


def _select_oscillators(oscillators, pole_mode):
    """Select either the strongest pole or all poles for tracking."""
    if pole_mode == "all":
        return oscillators
    if pole_mode == "max":
        return (
            [max(oscillators, key=lambda oscillator: oscillator["r"])]
            if oscillators
            else []
        )
    raise ValueError("pole_mode must be either 'max' or 'all'")


def _match_oscillators(tracks, oscillators):
    """Match current poles to previous tracks by nearest frequency.

    Each current oscillator is assigned at most once. Unmatched current poles
    become new tracks, which allows multiple modes to be detected in the same
    fine-scan region.
    """
    available_tracks = list(tracks)
    matches = []
    unmatched_oscillators = []

    for oscillator in oscillators:
        if not available_tracks:
            unmatched_oscillators.append(oscillator)
            continue

        track = min(
            available_tracks,
            key=lambda candidate: abs(oscillator["frequency"] - candidate["frequency"]),
        )
        available_tracks.remove(track)
        matches.append((track, oscillator))

    matches.extend(
        (_new_track(oscillator), oscillator) for oscillator in unmatched_oscillators
    )
    return matches


def detect_events(
    signal,
    fs,
    r_a,
    r_b,
    pole_mode: Literal["max", "all"] = "all",
):
    """Detect events while tracking the strongest pole or all poles.

    ``pole_mode="max"`` keeps only the pole with the largest radius in each
    analysis window. ``pole_mode="all"`` tracks every positive-frequency pole.
    """
    if pole_mode not in {"max", "all"}:
        raise ValueError("pole_mode must be either 'max' or 'all'")

    print("Signal length:", len(signal))
    print("Duration:", len(signal) / fs, "seconds")

    all_events = []
    pole_trace = []
    start = 0

    progress = tqdm(total=len(signal) / fs, desc="Scanning")
    last_progress = 0

    while start < len(signal) - fs:
        # Coarse scan: non-overlapping one-second segments.
        segment = signal[start : start + fs]
        oscillators = _select_oscillators(
            get_oscillators(segment, p=AR_ORDER, fs=fs), pole_mode
        )

        candidate_found = any(oscillator["r"] > r_a for oscillator in oscillators)
        if not candidate_found:
            start += fs
            progress.update(1)
            continue

        # Fine scan: overlapping one-second segments at 1/16-second spacing.
        fine_start = max(0, start - fs)
        tracks = []
        event_detected = False
        time = fine_start / fs

        for time in np.arange(
            fine_start / fs,
            len(signal) / fs - 1,
            FINE_SCAN_STEP_SECONDS,
        ):
            idx = round(time * fs)
            segment = signal[idx : idx + fs]
            if len(segment) < fs:
                break

            oscillators = _select_oscillators(
                get_oscillators(segment, p=AR_ORDER, fs=fs), pole_mode
            )

            for rank, oscillator in enumerate(oscillators):
                pole_trace.append(
                    {
                        "time": time,
                        "rank": rank,
                        "r": oscillator["r"],
                        "frequency": oscillator["frequency"],
                    }
                )

            matched_tracks = _match_oscillators(tracks, oscillators)
            next_tracks = []

            for track, oscillator in matched_tracks:
                radius = oscillator["r"]
                frequency = oscillator["frequency"]

                if not track["event_started"]:
                    previous_r = track["previous_r"]
                    upward_crossing = (
                        previous_r is not None and previous_r <= r_b and radius > r_b
                    )

                    if upward_crossing:
                        track["event_started"] = True
                        track["t1"] = time
                        track["max_r"] = radius
                        track["peak_time"] = time
                        track["peak_frequency"] = frequency
                        event_detected = True
                else:
                    if radius > track["max_r"]:
                        track["max_r"] = radius
                        track["peak_time"] = time
                        track["peak_frequency"] = frequency

                    previous_r = track["previous_r"]
                    if previous_r is not None and previous_r >= r_b and radius < r_b:
                        track["last_rb_crossing"] = time

                    if radius < r_a:
                        _append_completed_event(all_events, track)
                        continue

                track["previous_r"] = radius
                track["frequency"] = frequency
                next_tracks.append(track)

            tracks = next_tracks

            # Once all started events have ended and no pole remains above the
            # lower threshold, this candidate region is complete.
            active_events = any(track["event_started"] for track in tracks)
            any_candidate = any(oscillator["r"] > r_a for oscillator in oscillators)
            if event_detected and not active_events and not any_candidate:
                break

        # Move the coarse scan past the fine-scanned event region.
        if event_detected:
            start = int((cast(float, time) + 1.0) * fs)
        else:
            start += fs

        current_progress = start / fs
        progress.update(current_progress - last_progress)
        last_progress = current_progress

    progress.close()
    pole_df = pd.DataFrame(pole_trace)
    print(pole_df.head(20))
    if not pole_df.empty:
        print(pole_df.describe())

    return all_events
