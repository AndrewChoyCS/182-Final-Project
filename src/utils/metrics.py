"""
Evaluation metrics.
"""

import torch
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
import numpy as np


def compute_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute accuracy."""
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    return accuracy_score(labels, preds)


def compute_topk_accuracy(
    preds: torch.Tensor,
    labels: torch.Tensor,
    k: int = 5,
) -> float:
    """Compute top-k accuracy."""
    if preds.dim() == 1:
        # If predictions are class indices, we need logits
        # This is a simplified version - assumes preds are already top-k
        return compute_accuracy(preds, labels)
    
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    
    try:
        return top_k_accuracy_score(labels, preds, k=k)
    except:
        # Fallback implementation
        topk_preds = np.argsort(preds, axis=1)[:, -k:]
        correct = np.sum([labels[i] in topk_preds[i] for i in range(len(labels))])
        return correct / len(labels)


def compute_f1_score(
    preds: torch.Tensor,
    labels: torch.Tensor,
    average: str = "macro",
) -> float:
    """Compute F1 score."""
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    return f1_score(labels, preds, average=average)

