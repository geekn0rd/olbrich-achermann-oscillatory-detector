"""Inspect duplicate events in an algorithm scoring file."""

from pathlib import Path

import pandas as pd

SCORING_FILE = Path("algorithm_scoring_excerpt1.txt")


def main() -> None:
    """Print basic duplicate and uniqueness statistics for a scoring file."""
    events = pd.read_csv(
        SCORING_FILE,
        sep=r"\s+",
        skiprows=1,
        header=None,
        names=["start", "duration", "oscillator"],
    )

    print("Total events:", len(events))
    print("Unique exact events:", events.drop_duplicates().shape[0])

    print("\nMost repeated exact events:")
    print(events.value_counts().head(20))

    unique_start_duration = events[["start", "duration"]].drop_duplicates()
    print("\nUnique start/duration events:", unique_start_duration.shape[0])


if __name__ == "__main__":
    main()
