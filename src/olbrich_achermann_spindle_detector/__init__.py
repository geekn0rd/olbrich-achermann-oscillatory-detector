"""Automatic sleep-spindle detection and DREAMS validation."""

from .batch import find_excerpts, main, process_excerpt
from .detector import detect_events, get_oscillators

__all__ = [
    "detect_events",
    "find_excerpts",
    "get_oscillators",
    "main",
    "process_excerpt",
]
