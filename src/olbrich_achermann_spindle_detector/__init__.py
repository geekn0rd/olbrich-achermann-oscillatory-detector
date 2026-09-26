import mne
import numpy as np
import statsmodels.api as sm
from tqdm import tqdm

from utils import save_events_as_scoring


def calculate_nrem_spindle_rate(events, hypnogram, epoch_duration=5.0):
    """
    Calculate spindle rate per minute of NREM sleep.
    """

    # NREM = S1, S2, S3, S4
    nrem_codes = {0, 1, 2, 3}

    # Total NREM duration
    nrem_epochs = np.isin(hypnogram, list(nrem_codes))
    nrem_seconds = np.sum(nrem_epochs) * epoch_duration

    # Count events whose t1 falls in an NREM epoch
    nrem_events = 0

    for event in events:

        t1 = event["t1"]

        epoch = int(t1 // epoch_duration)

        if epoch >= len(hypnogram):
            continue

        if hypnogram[epoch] in nrem_codes:
            nrem_events += 1

    nrem_minutes = nrem_seconds / 60

    if nrem_minutes == 0:
        return 0.0

    return nrem_events / nrem_minutes

def load_hypnogram(filename):
    """
    Load DREAMS hypnogram.

    Each value represents one 5-second epoch.

    DREAMS codes:
        5  = Wake
        4  = REM
        3  = S1 / N1
        2  = S2 / N2
        1  = S3 / N3
        0  = S4 / N3
       -1  = Movement
       -2/-3 = Unknown
    """

    stages = []

    with open(filename, "r") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if line.lower() == "[hypnogram]":
                continue

            stages.append(int(line))

    return np.array(stages)


def find_events(fine_results, r_a, r_b, td=1.0):
    """
    Find oscillatory events from the fine-scan results.

    Parameters
    ----------
    fine_results : list
        Results from the fine scan.

    r_a : float
        Lower threshold for candidate events.

    r_b : float
        Higher threshold defining the oscillatory event.

    td : float
        Segment length. The paper uses td = 1 s.

    Returns
    -------
    events : list
        Detected events with t1, t2, and duration.
    """

    events = []

    for k in range(len(fine_results[0]["oscillators"])):

        in_candidate = False
        event_started = False

        t1 = None
        last_below_rb = None

        for result in fine_results:

            time = result["time"]
            oscillators = result["oscillators"]

            if k >= len(oscillators):
                continue

            r = oscillators[k]["r"]

            # --------------------------------------------------
            # Candidate event: rk exceeds ra
            # --------------------------------------------------

            if not in_candidate:

                if r > r_a:
                    in_candidate = True
                else:
                    continue

            # --------------------------------------------------
            # Event starts when rk crosses rb upwards
            # --------------------------------------------------

            if not event_started:

                if r > r_b:
                    t1 = time
                    event_started = True

                continue

            # --------------------------------------------------
            # Event is active
            # --------------------------------------------------

            if r < r_b:
                last_below_rb = time

            # --------------------------------------------------
            # Candidate event ends when rk falls below ra
            # --------------------------------------------------

            if r < r_a:

                if last_below_rb is not None:

                    t2 = last_below_rb

                    duration = t2 - t1 + td

                    events.append(
                        {
                            "oscillator": k,
                            "t1": t1,
                            "t2": t2,
                            "duration": duration,
                        }
                    )

                break

    return events

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
    r_a = 0.9
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

    candidates = []
    for start in range(0, len(signal) - fs + 1, fs):
        segment = signal[start : start + fs]
        oscillators = get_oscillators(segment, p=8, fs=fs)

        for osc in oscillators:
            if osc["r"] > r_a:
                # print(f"\nOscillator {i} in segment {start // fs + 1}")
                # print(f"  r         = {osc['r']:.6f}")
                # print(f"  frequency = {osc['frequency']:.3f} Hz")
                candidate_start = max(0, start - fs)
                if candidate_start not in candidates:
                    candidates.append(candidate_start)

    print(len(candidates))

    all_events = []
    for candidate in tqdm(candidates, desc="Candidates"):
        fine_times = np.arange(
            candidate / fs,
            len(signal) / fs - 1,
            1 / 16,
        )

        fine_results = []

        for time in fine_times:

            start = round(time * fs)

            segment = signal[start : start + fs]

            if len(segment) < fs:
                break

            oscillators = get_oscillators(
                segment,
                p=8,
                fs=fs,
            )

            fine_results.append(
                {
                    "start_sample": start,
                    "time": time,
                    "oscillators": oscillators,
                }
            )


        events = find_events(fine_results, r_a, r_b)

        for event in events:
            all_events.append(event)
            # print(
            #     f"oscillator {event['oscillator']}: "
            #     f"t1={event['t1']:.4f}, "
            #     f"t2={event['t2']:.4f}, "
            #     f"duration={event['duration']:.2f}"
            # )
            #

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


