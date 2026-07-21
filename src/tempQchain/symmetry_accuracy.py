import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

LABELS_INT = MappingProxyType({"AFTER": 0, "BEFORE": 1, "INCLUDES": 2, "IS INCLUDED": 3, "SIMULTANEOUS": 4, "VAGUE": 5})

def label_to_string(label_int: int) -> str:
    int_to_label = {v: k.lower() for k, v in LABELS_INT.items()}
    return int_to_label.get(label_int, "unknown")

def load_constraint_data(file_path: str) -> list[dict[str, Any]]:
    with open(file_path, "r") as f:
        return json.load(f)

def is_symmetry_constraint(constraint_type: str) -> bool:
    return constraint_type.lower() == "symmetric"

def analyze_symmetry_accuracy(data: list[dict[str, Any]]) -> dict[str, Any]:
    total_symmetry = 0
    correct_conclusions = 0
    incorrect_conclusions = 0

    for batch in data:
        if not is_symmetry_constraint(batch["constraint"]):
            continue
        # Get the related question and the primary
        related = batch["related"][0]
        primary = batch["primary"]

        primary_label = label_to_string(primary["prediction"])
        related_label = label_to_string(related["prediction"])

        if primary_label not in ["vague", "simultaneous"] and related_label not in ["vague", "simultaneous"]:
            continue

        total_symmetry += 1


        # For symmetry: if primary prediction is vague or simultaneous,
        # check if primary and related predictions are the same
        if primary_label == related_label:
            correct_conclusions += 1
        else:
            incorrect_conclusions += 1

    accuracy = correct_conclusions / total_symmetry if total_symmetry > 0 else 0.0

    return {
        "total_symmetry_batches": total_symmetry,
        "correct_conclusions": correct_conclusions,
        "incorrect_conclusions": incorrect_conclusions,
        "accuracy": accuracy,
    }

def calculate_symmetry_accuracy(file_path: str) -> dict[str, Any]:
    """Calculate symmetry accuracy from constraint results file."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    data = load_constraint_data(file_path)
    results = analyze_symmetry_accuracy(data)
    return results