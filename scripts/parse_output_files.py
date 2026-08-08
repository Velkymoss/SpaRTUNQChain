import re

import numpy as np


def extract_metric_values(criteria: str, strings: list[str]) -> np.ndarray:
    """Extracts float values from strings like 'Validation Loss: 1.747406'."""
    # return [float(s.split(criteria)[1]) for s in strings]
    return np.array([float(s.split(criteria)[1]) for s in strings])


def parse_training_log(file_path: str):
    """Parses a model training log file to extract per-epoch metrics:

    - Training Loss
    - Training Macro F1
    - Validation Loss
    - Validation Macro F1

    Returns a pandas DataFrame with the extracted metrics.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = r"Training Macro F1: \d+\.\d+"
    training_macro_f1_values = re.findall(pattern, content, flags=re.IGNORECASE)
    tranining_macro_f1_values = extract_metric_values("Training Macro F1: ", training_macro_f1_values)

    pattern = r"Training Loss: \d+\.\d+"
    training_loss_values = re.findall(pattern, content, flags=re.IGNORECASE)
    training_loss_values = extract_metric_values("Training Loss: ", training_loss_values)

    pattern = r"Validation Macro F1: \d+\.\d+"
    validation_macro_f1_values = re.findall(pattern, content, flags=re.IGNORECASE)
    validation_macro_f1_values = extract_metric_values("Validation Macro F1: ", validation_macro_f1_values)

    pattern = r"Validation Loss: \d+\.\d+"
    validation_loss_values = re.findall(pattern, content, flags=re.IGNORECASE)
    validation_loss_values = extract_metric_values("Validation Loss: ", validation_loss_values)

    pattern = r"Test Macro F1: \d+\.\d+"
    test_macro_f1 = re.findall(pattern, content, flags=re.IGNORECASE)
    test_macro_f1 = extract_metric_values("Test Macro F1: ", test_macro_f1)[0]

    pattern = r"Test Loss: \d+\.\d+"
    test_loss = re.findall(pattern, content, flags=re.IGNORECASE)
    test_loss = extract_metric_values("Test Loss: ", test_loss)[0]

    pattern = r"Test F1 Scores:(.*?)Test Macro F1:"
    match = re.search(pattern, content, flags=re.DOTALL | re.IGNORECASE)

    if match:
        extracted_text = match.group(1).strip()

    pattern = r"(\w+) F1: (\d+\.\d+)"
    matches = re.findall(pattern, extracted_text, flags=re.IGNORECASE)
    class_f1_scores = {relation: float(score) for relation, score in matches}

    train_val_data = {
        "training_loss": training_loss_values,
        "training_f1": tranining_macro_f1_values,
        "validation_loss": validation_loss_values,
        "validation_f1": validation_macro_f1_values,
    }

    test_data = {
        "test_loss": test_loss,
        "test_macro_f1": test_macro_f1,
        "class_f1_scores": class_f1_scores,
    }

    return train_val_data, test_data


if __name__ == "__main__":
    path = "runs/baseline_seed_2.out"
    train_val_metrics, test_metrics = parse_training_log(path)
    print("Train/Validation Metrics:")
    print(train_val_metrics)
    print("\nTest Metrics:")
    print(test_metrics)
