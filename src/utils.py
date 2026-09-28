import numpy as np


def save_events_as_scoring(
    events,
    filename,
    channel="C3-A1",
    header="algorithm_Spindles",
):
    """
    Save detected spindle events in DREAMS-like scoring format.

    Format:

        [algorithm_Spindles/C3-A1]
        start_time    duration    oscillator

    Parameters
    ----------
    events : list of dict
        Output from find_events().

    filename : str
        Output filename.

    channel : str
        EEG channel name.

    header : str
        Annotation type/name.
    """

    with open(filename, "w") as f:

        # Header
        f.write(
            f"[{header}/{channel}]\n"
        )

        # Events
        for event in events:

            t1 = event["t1"]
            duration = event["duration"]

            f.write(
                f"{t1:10.4f}\t"
                f"{duration:10.4f}\n"
            )

    print(
        f"Saved {len(events)} events to {filename}"
    )


def calculate_nrem_spindle_rate(
    events,
    hypnogram,
    epoch_duration=5.0,
    min_freq=11.5,
    max_freq=16.0,
):
    nrem_codes = {0, 1, 2, 3}

    nrem_epochs = np.isin(hypnogram, list(nrem_codes))
    nrem_seconds = np.sum(nrem_epochs) * epoch_duration

    nrem_events = 0

    for event in events:
        frequency = event["frequency"]

        if not (min_freq <= frequency <= max_freq):
            continue

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
