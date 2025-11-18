"""
Linear probe evaluation utilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD
from tqdm import tqdm
from typing import Dict, Any, Optional

from ..utils.metrics import compute_accuracy, compute_topk_accuracy


class LinearProbe(nn.Module):
    """Linear classifier for probing."""
    
    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


def evaluate_linear_probe(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    device: str = "cuda",
    epochs: int = 10,
    lr: float = 0.001,
) -> Dict[str, float]:
    """
    Evaluate linear probe performance.
    
    Args:
        model: Base model to extract features from
        train_loader: DataLoader for training the probe
        val_loader: DataLoader for evaluation
        num_classes: Number of classes
        device: Device to run evaluation on
        epochs: Number of training epochs
        lr: Learning rate
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    model = model.to(device)
    
    # Extract features to determine dimension
    sample_batch = next(iter(train_loader))
    sample_image = sample_batch["image"][:1].to(device)
    
    with torch.no_grad():
        if hasattr(model, "encode_image"):
            sample_features = model.encode_image(sample_image)
        else:
            # For classification models, use intermediate features
            sample_features = model(sample_image)
        
        feature_dim = sample_features.shape[-1]
    
    # Create linear probe
    probe = LinearProbe(feature_dim, num_classes).to(device)
    optimizer = SGD(probe.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # Train probe
    probe.train()
    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Training probe {epoch + 1}/{epochs}")
        
        for batch in pbar:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            # Extract features
            with torch.no_grad():
                if hasattr(model, "encode_image"):
                    features = model.encode_image(images)
                else:
                    features = model(images)
            
            # Train probe
            optimizer.zero_grad()
            logits = probe(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    # Evaluate probe
    probe.eval()
    all_preds = []
    all_labels = []
    
    pbar = tqdm(val_loader, desc="Evaluating probe")
    
    with torch.no_grad():
        for batch in pbar:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            # Extract features
            if hasattr(model, "encode_image"):
                features = model.encode_image(images)
            else:
                features = model(images)
            
            # Get predictions
            logits = probe(features)
            preds = logits.argmax(dim=1)
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    if all_preds and all_labels:
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        accuracy = compute_accuracy(all_preds, all_labels)
        top5_acc = compute_topk_accuracy(all_preds, all_labels, k=5)
        
        return {
            "linear_probe_accuracy": accuracy,
            "linear_probe_top5_accuracy": top5_acc,
        }
    
    return {}

