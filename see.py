import pandas as pd

df = pd.read_csv(
    "algorithm_scoring_excerpt1.txt",
    sep=r"\s+",
    skiprows=1,
    header=None,
    names=["start", "duration", "oscillator"],
)

print("Total events:", len(df))
print("Unique exact events:", df.drop_duplicates().shape[0])

print("\nMost repeated exact events:")
print(
    df.value_counts()
      .head(20)
)

print("\nUnique start/duration events:",
      df[["start", "duration"]].drop_duplicates().shape[0]
)
