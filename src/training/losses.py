"""
Loss functions for training.
"""

import torch
import torch.nn.functional as F


def clip_loss(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    """
    Compute CLIP contrastive loss.
    
    Args:
        logits_per_image: [batch_size, batch_size] similarity matrix
        logits_per_text: [batch_size, batch_size] similarity matrix (transpose)
    
    Returns:
        Contrastive loss
    """
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, device=logits_per_image.device)
    
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    
    return (loss_i + loss_t) / 2.0


def classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute classification cross-entropy loss.
    
    Args:
        logits: [batch_size, num_classes] logits
        labels: [batch_size] ground truth labels
    
    Returns:
        Classification loss
    """
    return F.cross_entropy(logits, labels)

