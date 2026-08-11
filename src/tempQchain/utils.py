import math
import random
from collections import Counter

import torch
import tqdm
from domiknows.program.lossprogram import LearningBasedProgram
from domiknows.program.model.base import Mode


def get_avg_loss(
    program: LearningBasedProgram, dataset: list[dict[str, str]], cur_device: str | None, mode: str
) -> float:
    if cur_device is not None:
        program.model.to(cur_device)
    program.model.mode(Mode.TEST)
    program.model.reset()
    train_loss = 0
    total_loss = 0
    with torch.no_grad():
        for data_item in tqdm.tqdm(dataset, f"Calculating {mode} loss" if mode else "Calculating loss"):
            loss, _, *output = program.model(data_item)
            total_loss += 1
            train_loss += loss
    return train_loss / total_loss


def get_train_labels(dataset: list[dict[str, str]]) -> list[int]:
    labels = []
    for batch in dataset:
        batch_labels = batch["labels"].split("@@")
        batch_labels = [int(label) for label in batch_labels]
        labels.extend(batch_labels)
    return labels


def get_class_distribution(dataset: list[dict[str, str]]) -> dict[int, dict[str, int | float]]:
    labels = get_train_labels(dataset)
    total_samples = len(labels)

    class_counts = Counter(labels)

    distribution = {}
    for class_label in sorted(class_counts.keys()):
        count = class_counts[class_label]
        distribution[class_label] = {
            "count": count,
            "percentage": (count / total_samples) * 100 if total_samples > 0 else 0.0,
        }

    return distribution


def sample_batches(batches: list[dict[str, str]], ratio: float, ratio_seed: int | None = 42) -> list[dict[str, str]]:
    """Sample a random fraction (0.0 to 1.0) of batches for training.

    Args:
        batches: List of batches.
        ratio: Fraction of batches to sample, between 0.0 and 1.0.
        ratio_seed: Optional integer seed for reproducible random selection.

    Returns:
        A list containing the sampled subset of batches.
    """
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"Ratio must be between 0.0 and 1.0 inclusive, got {ratio}")

    k = math.ceil(len(batches) * ratio)

    if ratio_seed is not None:
        # Create an isolated local generator instance
        rng = random.Random(ratio_seed)
        return rng.sample(batches, k=k)

    return random.sample(batches, k=k)  
