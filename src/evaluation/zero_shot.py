"""
Zero-shot evaluation utilities.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, List, Optional

from ..utils.metrics import compute_accuracy, compute_topk_accuracy


def evaluate_zero_shot(
    model: nn.Module,
    dataloader: DataLoader,
    class_names: List[str],
    device: str = "cuda",
    text_template: str = "a photo of a {}",
) -> Dict[str, float]:
    """
    Evaluate zero-shot classification performance.
    
    Args:
        model: CLIP model
        dataloader: DataLoader for evaluation
        class_names: List of class names
        device: Device to run evaluation on
        text_template: Template for text prompts
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    model = model.to(device)
    
    # Create text prompts for all classes
    text_prompts = [text_template.format(name) for name in class_names]
    
    # Tokenize text prompts
    from ..utils.tokenizer import get_tokenizer
    tokenizer_fn = get_tokenizer("clip")
    text_tokens = tokenizer_fn(text_prompts)
    
    # Handle different tokenizer output formats
    if isinstance(text_tokens, dict):
        text_tokens = {k: v.to(device) for k, v in text_tokens.items()}
    else:
        text_tokens = text_tokens.to(device)
    
    # Encode text prompts once
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)
    
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Zero-shot evaluation")
    
    with torch.no_grad():
        for batch in pbar:
            images = batch["image"].to(device)
            labels = batch.get("label")
            
            if labels is None:
                continue
            
            # Encode images
            image_features = model.encode_image(images)
            image_features = torch.nn.functional.normalize(image_features, dim=-1)
            
            # Compute similarity
            similarity = image_features @ text_features.T
            
            # Get predictions
            preds = similarity.argmax(dim=1)
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    if all_preds and all_labels:
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        accuracy = compute_accuracy(all_preds, all_labels)
        
        return {
            "zero_shot_accuracy": accuracy,
        }
    
    return {}

