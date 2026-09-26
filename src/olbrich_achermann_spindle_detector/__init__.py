import mne
import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm

from .utils import calculate_nrem_spindle_rate, load_hypnogram, save_events_as_scoring
from .validate import validate


def get_oscillators(segment, p, fs):
    """
    Fit AR(p) to a 1-second segment and return
    the positive-frequency complex poles.
    """

    a, _ = sm.regression.linear_model.burg(segment, order=p)

    # z^p - a[0]z^(p-1) - ... - a[p-1] = 0
    coefficients = np.concatenate([[1.0], -a])
    roots = np.roots(coefficients)

    oscillators = []

    for z in roots:

        # Keep only one pole from each complex-conjugate pair
        if z.imag <= 0:
            continue

        r = np.abs(z)
        theta = np.angle(z)

        frequency = theta * fs / (2 * np.pi)

        oscillators.append(
            {
                "pole": z,
                "r": r,
                "theta": theta,
                "frequency": frequency,
            }
        )

    return oscillators


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

    print("Signal length:", len(signal))
    print("Duration:", len(signal) / fs, "seconds")

    all_events = []
    pole_trace = []
    start = 0

    pbar = tqdm(total=len(signal) / fs, desc="Scanning")
    last_progress = 0
    while start < len(signal) - fs:

        # -----------------------------
        # Coarse scan (1 s step)
        # -----------------------------

        segment = signal[start:start + fs]

        oscillators = get_oscillators(
            segment,
            p=8,
            fs=fs
        )

        candidate_found = any(
            osc["r"] > r_a
            for osc in oscillators
        )


        if not candidate_found:

            start += fs
            pbar.update(1)
            continue


        # -----------------------------
        # Fine scan (1/16 s step)
        # -----------------------------

        fine_start = max(0, start - fs)

        event_started = False

        active_frequency = None

        t1 = None
        t2 = None

        previous_r = None
        last_rb_crossing = None

        max_r = -np.inf
        peak_time = None
        peak_frequency = None


        for time in np.arange(
            fine_start / fs,
            len(signal) / fs - 1,
            1 / 16,
        ):

            idx = round(time * fs)

            segment = signal[idx:idx + fs]

            if len(segment) < fs:
                break


            oscillators = get_oscillators(
                segment,
                p=8,
                fs=fs,
            )


            # save pole information
            for rank, osc in enumerate(oscillators):

                pole_trace.append(
                    {
                        "time": time,
                        "rank": rank,
                        "r": osc["r"],
                        "frequency": osc["frequency"],
                    }
                )


            # -----------------------------
            # Select oscillator
            # -----------------------------

            if not event_started:

                # strongest pole starts event
                osc = max(
                    oscillators,
                    key=lambda x: x["r"]
                )

            else:

                # follow same oscillator
                osc = min(
                    oscillators,
                    key=lambda x:
                        abs(
                            x["frequency"]
                            -
                            active_frequency
                        )
                )


            active_frequency = osc["frequency"]

            r = osc["r"]
            f = osc["frequency"]


            # -----------------------------
            # Event start
            # -----------------------------

            if not event_started:

                if previous_r is not None:

                    upward_cross = (
                        previous_r <= r_b
                        and r > r_b
                    )

                else:
                    upward_cross = False


                if upward_cross:

                    event_started = True

                    t1 = time

                    max_r = r
                    peak_time = time
                    peak_frequency = f


                previous_r = r
                continue


            # -----------------------------
            # Event ongoing
            # -----------------------------

            if r > max_r:

                max_r = r
                peak_time = time
                peak_frequency = f


            # downward crossing of rb
            if (
                previous_r is not None
                and previous_r >= r_b
                and r < r_b
            ):

                last_rb_crossing = time


            # final event end
            if r < r_a:

                t2 = last_rb_crossing

                if t2 is not None:

                    all_events.append(
                        {
                            "t1": t1,
                            "t2": t2,

                            "time": peak_time,
                            "frequency": peak_frequency,

                            "r": max_r,

                            "duration": t2 - t1 + 1.0,
                        }
                    )

                break


            previous_r = r


        # -----------------------------
        # Move coarse scan forward
        # -----------------------------

        if event_started:

            # jump past the complete detected event
            start = int((time + 1.0) * fs)

        else:

            start += fs


        current_progress = start / fs
        pbar.update(current_progress - last_progress)
        last_progress = current_progress

    pbar.close()
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

    pole_df = pd.DataFrame(pole_trace)
    print(pole_df.head(20))
    print(pole_df.describe())
