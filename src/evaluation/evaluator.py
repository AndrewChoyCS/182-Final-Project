"""
Model evaluation utilities.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, Optional, List

from ..utils.metrics import compute_accuracy, compute_topk_accuracy, compute_f1_score


class Evaluator:
    """Evaluator class for model evaluation."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            dataloader: DataLoader for evaluation
            metrics: List of metrics to compute
        
        Returns:
            Dictionary of metric values
        """
        if metrics is None:
            metrics = ["accuracy", "top5_accuracy"]
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        pbar = tqdm(dataloader, desc="Evaluating")
        
        for batch in pbar:
            images = batch["image"].to(self.device)
            labels = batch.get("label")
            texts = batch.get("text")
            
            if texts is not None and isinstance(texts, torch.Tensor):
                texts = texts.to(self.device)
            elif isinstance(texts, dict):
                texts = {k: v.to(self.device) for k, v in texts.items()}
            
            # Forward pass
            outputs = self.model(images, texts)
            
            # Extract predictions
            if "logits_per_image" in outputs:
                # CLIP model
                logits = outputs["logits_per_image"]
                preds = logits.argmax(dim=1)
            else:
                # Classification model
                logits = outputs
                preds = logits.argmax(dim=1)
            
            if labels is not None:
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
                all_probs.append(torch.softmax(logits, dim=1).cpu())
        
        # Compute metrics
        results = {}
        
        if all_preds and all_labels:
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            
            if "accuracy" in metrics:
                results["accuracy"] = compute_accuracy(all_preds, all_labels)
            
            if "top5_accuracy" in metrics and all_probs:
                all_probs = torch.cat(all_probs, dim=0)
                results["top5_accuracy"] = compute_topk_accuracy(all_probs, all_labels, k=5)
            
            if "f1_score" in metrics:
                results["f1_score"] = compute_f1_score(all_preds, all_labels)
        
        return results

