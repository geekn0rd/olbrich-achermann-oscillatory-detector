# validate.py

import re

import numpy as np
from scipy.optimize import linear_sum_assignment


def load_expert_events(filename):
    """
    DREAMS expert spindle annotations contain
    start time and spindle duration.
    """

    events = []

    with open(filename, "r") as f:

        for line in f:

            line = line.strip()

            if not line or line.startswith("#"):
                continue

            numbers = re.findall(
                r"[-+]?(?:\d*\.\d+|\d+\.?\d*)",
                line,
            )

            if not numbers:
                continue

            start = float(numbers[0])
            duration = float(numbers[1])

            end = start + duration

            events.append({
                "t1": start,
                "t2": end,
                "duration": duration,
            })

    return events


def interval_iou(a, b):
    """Intersection over Union of two time intervals."""

    start = max(a[0], b[0])
    end = min(a[1], b[1])

    intersection = max(0.0, end - start)

    union = (
        max(a[1], b[1])
        - min(a[0], b[0])
    )

    if union == 0:
        return 0.0

    return intersection / union


def match_events(
    detected_events,
    expert_events,
    iou_threshold=0.3,
):
    """
    Match detected events to expert events
    using maximum IoU and one-to-one matching.
    """

    detected = [
        (e["t1"], e["t2"])
        for e in detected_events
    ]

    expert = [
        (e["t1"], e["t2"])
        for e in expert_events
    ]

    if not detected or not expert:
        return []

    iou_matrix = np.zeros(
        (len(detected), len(expert))
    )

    for i, d in enumerate(detected):

        for j, e in enumerate(expert):

            iou_matrix[i, j] = interval_iou(d, e)

    rows, cols = linear_sum_assignment(
        -iou_matrix
    )

    matches = []

    for i, j in zip(rows, cols):

        iou = iou_matrix[i, j]

        if iou >= iou_threshold:

            matches.append({
                "detected_index": i,
                "expert_index": j,
                "iou": iou,
            })

    return matches


def evaluate(
    detected_events,
    expert_events,
    iou_threshold=0.3,
):
    """
    Calculate precision, recall, F1 and IoU.
    """

    matches = match_events(
        detected_events,
        expert_events,
        iou_threshold=iou_threshold,
    )

    tp = len(matches)

    fp = len(detected_events) - tp
    fn = len(expert_events) - tp

    precision = (
        tp / len(detected_events)
        if detected_events
        else 0.0
    )

    recall = (
        tp / len(expert_events)
        if expert_events
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    mean_iou = (
        np.mean([m["iou"] for m in matches])
        if matches
        else 0.0
    )

    return {
        "detected": len(detected_events),
        "expert": len(expert_events),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
    }


def validate(
    detected_events,
    expert1_file,
    expert2_file,
    iou_threshold=0.3,
):
    """
    Compare detector against both DREAMS experts.
    """

    expert1 = load_expert_events(
        expert1_file
    )

    expert2 = load_expert_events(
        expert2_file
    )

    result1 = evaluate(
        detected_events,
        expert1,
        iou_threshold,
    )

    result2 = evaluate(
        detected_events,
        expert2,
        iou_threshold,
    )

    print("\n" + "=" * 60)
    print("DREAMS VALIDATION")
    print("=" * 60)

    print("\nExpert 1")
    print("-" * 40)

    print_results(result1)

    print("\nExpert 2")
    print("-" * 40)

    print_results(result2)

    return {
        "expert1": result1,
        "expert2": result2,
    }


def print_results(result):

    print(f"Detected : {result['detected']}")
    print(f"Expert   : {result['expert']}")
    print(f"TP       : {result['tp']}")
    print(f"FP       : {result['fp']}")
    print(f"FN       : {result['fn']}")
    print(f"Precision: {result['precision']:.3f}")
    print(f"Recall   : {result['recall']:.3f}")
    print(f"F1       : {result['f1']:.3f}")
    print(f"Mean IoU : {result['mean_iou']:.3f}")
