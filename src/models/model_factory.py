"""
Model factory for creating different model architectures.
"""

import torch
from typing import Dict, Any
from .clip_model import CLIPModel
from .vit_model import VisionTransformer


def get_model(model_config: Dict[str, Any]) -> torch.nn.Module:
    """
    Create a model based on configuration.
    
    Args:
        model_config: Model configuration dictionary
    
    Returns:
        Model instance
    """
    model_type = model_config.get("type", "clip").lower()
    
    if model_type == "clip":
        return CLIPModel(
            vision_model=model_config.get("vision_model", "ViT-B/32"),
            embed_dim=model_config.get("embed_dim", 512),
            temperature=model_config.get("temperature", 0.07),
            freeze_vision=model_config.get("freeze_vision", False),
            freeze_text=model_config.get("freeze_text", False),
            use_pretrained=model_config.get("use_pretrained", True),
        )
    elif model_type == "vit":
        return VisionTransformer(
            img_size=model_config.get("image_size", 224),
            patch_size=model_config.get("patch_size", 16),
            embed_dim=model_config.get("embed_dim", 768),
            depth=model_config.get("depth", 12),
            num_heads=model_config.get("num_heads", 12),
            num_classes=model_config.get("num_classes", 1000),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

