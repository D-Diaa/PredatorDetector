from typing import Dict
from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score
)
from tqdm import tqdm


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    predictions, true_labels = [], []

    progress_bar = tqdm(train_loader, desc='Training')
    for batch_idx, batch in enumerate(progress_bar):
        sequences = batch['sequence'].to(device)
        labels = batch['label'].float().to(device)

        optimizer.zero_grad()
        logits = model(sequences)
        loss = criterion(logits, labels)
        if torch.isnan(loss):
            print(f'NaN loss detected at batch {batch_idx}')
            return total_loss / len(train_loader), np.array(predictions), np.array(true_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        predictions.extend(torch.sigmoid(logits).cpu().detach().numpy())
        true_labels.extend(labels.cpu().numpy())

        # Calculate metrics with current threshold
        batch_metrics = calculate_metrics_threshold(
            np.array(true_labels[-len(labels):]),
            np.array(predictions[-len(labels):])
        )[0]

        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'f1': f'{batch_metrics["f1"]:.4f}'
        })

    return total_loss / len(train_loader), np.array(predictions), np.array(true_labels)


def evaluate(model, data_loader, criterion, device, threshold=None):
    model.eval()
    total_loss = 0
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            sequences = batch['sequence'].to(device)
            labels = batch['label'].float().to(device)

            logits = model(sequences)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            predictions.extend(torch.sigmoid(logits).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # Calculate metrics with optimal threshold
    metrics, new_threshold = calculate_metrics_threshold(
        np.array(true_labels),
        np.array(predictions),
        threshold
    )

    return total_loss / len(data_loader), predictions, true_labels, metrics, new_threshold

def calculate_metrics_threshold(y_true: np.ndarray, y_scores: np.ndarray, threshold=None) -> Tuple[
    Dict[str, float], float]:
    """
    Calculate metrics with optimal threshold selection.

    Args:
        y_true: True labels
        y_scores: Predicted probabilities
        threshold: Threshold for binary classification

    Returns:
        Dictionary of metrics and optimal threshold
    """
    if threshold is None:
        if len(np.unique(y_true)) > 1:
            precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
            optimal_idx = np.argmax(f1_scores)
            threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        else:
            threshold = 0.5

    # Calculate metrics with optimal threshold
    y_pred = (y_scores >= threshold).astype(int)
    metrics = {
        "optimal_threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.5
    }

    return metrics, threshold

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
