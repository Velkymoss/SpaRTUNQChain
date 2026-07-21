import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

LABELS_INT = MappingProxyType({"AFTER": 0, "BEFORE": 1, "INCLUDES": 2, "IS INCLUDED": 3, "SIMULTANEOUS": 4, "VAGUE": 5})


trans_rules = {
    ("before", "before"): ["before"],
    ("after", "after"): ["after"],
    ("includes", "includes"): ["includes"],
    ("is included", "is included"): ["is included"],
    ("simultaneous", "simultaneous"): ["simultaneous"],
    ("before", "simultaneous"): ["before"],
    ("after", "simultaneous"): ["after"],
    ("includes", "simultaneous"): ["includes"],
    ("is included", "simultaneous"): ["is included"],
    ("simultaneous", "before"): ["before"],
    ("simultaneous", "after"): ["after"],
    ("simultaneous", "includes"): ["includes"],
    ("simultaneous", "is included"): ["is included"],
}


def label_to_string(label_int: int) -> str:
    int_to_label = {v: k.lower() for k, v in LABELS_INT.items()}
    return int_to_label.get(label_int, "unknown")


def load_constraint_data(file_path: str) -> list[dict[str, Any]]:
    with open(file_path, "r") as f:
        return json.load(f)


def is_transitive_constraint(constraint_type: str) -> bool:
    return constraint_type.lower() == "transitive"


def analyze_transitive_accuracy(data: list[dict[str, Any]]) -> dict[str, Any]:
    total_transitive = 0
    correct_conclusions = 0
    incorrect_conclusions = 0
    rule_stats = {}

    for batch in data:
        if not is_transitive_constraint(batch["constraint"]):
            continue

        # Get the two related questions (premises) and the primary (conclusion)
        related_1 = batch["related"][0]
        related_2 = batch["related"][1]
        primary = batch["primary"]

        # Create rule key from the two premises
        rule_key = (
            label_to_string(related_1["prediction"]),
            label_to_string(related_2["prediction"]),
        )

        # Check if this rule key exists in trans_rules
        if rule_key in trans_rules:
            total_transitive += 1
            if rule_key not in rule_stats:
                rule_stats[rule_key] = {"total": 0, "correct": 0, "incorrect": 0}

            expected_labels = trans_rules[rule_key]
            conclusion_label = label_to_string(primary["prediction"])

            # Check if conclusion is in the expected labels
            if conclusion_label == expected_labels[0]:
                correct_conclusions += 1
                rule_stats[rule_key]["correct"] += 1
                rule_stats[rule_key]["total"] += 1

            else:
                incorrect_conclusions += 1
                rule_stats[rule_key]["incorrect"] += 1
                rule_stats[rule_key]["total"] += 1

    accuracy = correct_conclusions / total_transitive if total_transitive > 0 else 0.0

    return {
        "total_transitive_batches": total_transitive,
        "correct_conclusions": correct_conclusions,
        "incorrect_conclusions": incorrect_conclusions,
        "accuracy": accuracy,
        "rule_stats": rule_stats,
    }


def calculate_transitive_accuracy(file_path: str) -> dict[str, Any]:
    """Calculate transitive accuracy from constraint results file."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load data
    data = load_constraint_data(file_path)

    # Analyze
    results = analyze_transitive_accuracy(data)

    return results
