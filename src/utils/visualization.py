"""
Visualization utilities for experiments.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import List, Dict, Optional
from pathlib import Path


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    train_accs: Optional[List[float]] = None,
    val_accs: Optional[List[float]] = None,
    save_path: Optional[str] = None,
):
    """
    Plot training curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accs: List of training accuracies
        val_accs: List of validation accuracies
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses
    axes[0].plot(train_losses, label="Train Loss")
    if val_losses:
        axes[0].plot(val_losses, label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracies
    if train_accs:
        axes[1].plot(train_accs, label="Train Acc")
    if val_accs:
        axes[1].plot(val_accs, label="Val Acc")
    if train_accs or val_accs:
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Training and Validation Accuracy")
        axes[1].legend()
        axes[1].grid(True)
    else:
        axes[1].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
):
    """
    Plot confusion matrix.
    
    Args:
        predictions: Predicted labels
        labels: True labels
        class_names: List of class names
        save_path: Path to save the plot
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(labels.cpu().numpy(), predictions.cpu().numpy())
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    
    plt.close()


def visualize_attention(
    model: torch.nn.Module,
    image: torch.Tensor,
    save_path: Optional[str] = None,
):
    """
    Visualize attention maps from a vision transformer.
    
    Args:
        model: Vision transformer model
        image: Input image tensor [1, C, H, W]
        save_path: Path to save the visualization
    """
    # This is a placeholder - implement based on your specific model architecture
    # You would need to extract attention weights from the model
    pass

