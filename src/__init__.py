"""Automatic sleep-spindle detection and DREAMS validation."""

from .batch import find_excerpts, main, process_excerpt
from .detector import SpindleDetector, detect_events, get_oscillators

__all__ = [
    "SpindleDetector",
    "detect_events",
    "find_excerpts",
    "get_oscillators",
    "main",
    "process_excerpt",
]
