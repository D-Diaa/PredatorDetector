from typing import Dict

import numpy as np


def estimate_pos_weight(labels: np.ndarray, method: str = 'balanced') -> float:
    """
    Estimate positive class weight using different strategies.

    Args:
        labels: Array of binary labels
        method: Weight estimation method
            - 'balanced': inverse class frequency
            - 'effective': effective samples based on imbalance ratio
            - 'focal': weight for focal loss style adjustment
            - 'sqrt': square root of inverse class frequency

    Returns:
        Estimated positive class weight
    """
    pos_count = np.sum(labels == 1)
    neg_count = np.sum(labels == 0)

    if pos_count == 0:
        return 1.0

    imbalance_ratio = neg_count / pos_count

    if method == 'balanced':
        # Simple inverse class frequency
        return imbalance_ratio

    elif method == 'effective':
        # Based on "Learning from Imbalanced Data" paper
        # Uses effective number of samples
        beta = 0.9999  # Hyperparameter controlling smoothing
        effective_pos = (1 - beta ** pos_count) / (1 - beta)
        effective_neg = (1 - beta ** neg_count) / (1 - beta)
        return effective_neg / effective_pos

    elif method == 'focal':
        # Inspired by Focal Loss paper
        # Stronger weight for highly imbalanced cases
        gamma = 2.0  # Focusing parameter
        return (imbalance_ratio) ** (1 / gamma)

    elif method == 'sqrt':
        # Square root scaling to prevent too extreme weights
        return np.sqrt(imbalance_ratio)

    else:
        raise ValueError(f"Unknown weighting method: {method}")


def select_best_weight_method(results: Dict[str, Dict[str, float]],
                              metric: str = 'f1') -> str:
    """
    Select the best weight method based on a metric.

    Args:
        results: Results dictionary from evaluate_weight_methods
        metric: Metric to use for selection

    Returns:
        Best performing method
    """
    method_scores = {method: scores[metric] for method, scores in results.items()}
    best_method = max(method_scores.items(), key=lambda x: x[1])[0]
    return best_method
