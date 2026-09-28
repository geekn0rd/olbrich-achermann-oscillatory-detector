# Olbrich–Achermann Spindle Detector

A Python implementation of an automatic sleep-spindle detector based on autoregressive oscillator analysis. The batch entry point scans DREAMS EEG excerpts, detects candidate spindle events, calculates the NREM spindle rate, and optionally compares detections with expert annotations.

> **Data notice:** The DREAMS recordings and annotations are not included in this repository. Obtain them separately and place the permitted files in `DatabaseSpindles/`, respecting the license distributed with that dataset.

## Requirements

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) (recommended)
- DREAMS excerpt data in `DatabaseSpindles/`

## Installation

```bash
uv sync
```

The project also exposes a console command:

```bash
uv run olbrich-achermann-spindle-detector
```

## Usage

The batch runner discovers files named `excerpt<N>.edf` in `DatabaseSpindles/` and processes every matching excerpt:

```bash
uv run olbrich-achermann-spindle-detector
```

By default, the detector follows all positive-frequency poles. To follow only
one pole per analysis window—the pole with the largest radius—run:

```bash
uv run olbrich-achermann-spindle-detector --pole-mode max
```

The same option is available in Python as `detect_events(..., pole_mode="all")`
or `detect_events(..., pole_mode="max")`. For each excerpt it may write an algorithm scoring file to `results/`. After the batch completes, the combined metrics are written to `results/summary.csv`.

The repository also contains a small inspection helper for an algorithm scoring file:

```bash
uv run python see.py
```

## Project layout

```text
src/olbrich_achermann_spindle_detector/
├── batch.py      # Batch orchestration and summary reporting
├── detector.py   # Autoregressive oscillator detection
├── utils.py      # DREAMS I/O and NREM rate helpers
└── validate.py   # Expert annotation matching and metrics
see.py            # Scoring-file inspection helper
```

## Development

Compile-check the Python sources with:

```bash
uv run python -m compileall -q src see.py
```

No automated test suite is currently included. The detector prints progress and summary statistics during a batch run to make data-processing failures visible.

## License

The source code is maintained in this repository. Dataset licensing is governed by the DREAMS license included with the downloaded data; see `DatabaseSpindles/DREAMS_databases_License.txt` when the dataset is available locally.
