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
            oscillator = event["oscillator"]

            f.write(
                f"{t1:10.4f}\t"
                f"{duration:10.4f}\t"
                f"{oscillator}\n"
            )

    print(
        f"Saved {len(events)} events to {filename}"
    )
