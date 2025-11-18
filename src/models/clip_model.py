"""
CLIP (Contrastive Language-Image Pre-training) model implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import clip as openai_clip


class CLIPModel(nn.Module):
    """
    CLIP model for vision-language learning.
    """
    
    def __init__(
        self,
        vision_model: str = "ViT-B/32",
        embed_dim: int = 512,
        temperature: float = 0.07,
        freeze_vision: bool = False,
        freeze_text: bool = False,
        use_pretrained: bool = True,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.temperature = nn.Parameter(torch.ones([]) * temperature)
        self.freeze_vision = freeze_vision
        self.freeze_text = freeze_text
        
        # Load pretrained CLIP if available
        if use_pretrained:
            try:
                self.vision_encoder, self.text_encoder, self.preprocess = openai_clip.load(
                    vision_model, device="cpu"
                )
                # Override embed_dim if using pretrained
                self.embed_dim = self.vision_encoder.visual.output_dim
            except Exception as e:
                print(f"Could not load pretrained CLIP: {e}")
                print("Initializing from scratch...")
                use_pretrained = False
        
        if not use_pretrained:
            # Initialize from scratch (simplified version)
            from transformers import CLIPVisionModel, CLIPTextModel, CLIPProcessor
            
            self.vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
            self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.embed_dim = self.vision_encoder.config.projection_dim
        
        # Projection layers if needed
        if hasattr(self.vision_encoder, 'visual'):
            # OpenAI CLIP structure
            vision_dim = self.vision_encoder.visual.output_dim
        else:
            # HuggingFace CLIP structure
            vision_dim = self.vision_encoder.config.projection_dim
        
        if vision_dim != embed_dim:
            self.vision_proj = nn.Linear(vision_dim, embed_dim)
        else:
            self.vision_proj = nn.Identity()
        
        if hasattr(self.text_encoder, 'transformer'):
            # OpenAI CLIP structure
            text_dim = self.text_encoder.transformer.width
        else:
            # HuggingFace CLIP structure
            text_dim = self.text_encoder.config.projection_dim
        
        if text_dim != embed_dim:
            self.text_proj = nn.Linear(text_dim, embed_dim)
        else:
            self.text_proj = nn.Identity()
        
        # Freeze encoders if specified
        if freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
        
        if freeze_text:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to embeddings."""
        if hasattr(self.vision_encoder, 'visual'):
            # OpenAI CLIP
            features = self.vision_encoder.encode_image(images)
        else:
            # HuggingFace CLIP
            outputs = self.vision_encoder(pixel_values=images)
            features = outputs.pooler_output
        
        return self.vision_proj(features)
    
    def encode_text(self, texts) -> torch.Tensor:
        """Encode texts to embeddings."""
        if hasattr(self.text_encoder, 'transformer'):
            # OpenAI CLIP
            features = self.text_encoder.encode_text(texts)
        else:
            # HuggingFace CLIP
            if isinstance(texts, dict):
                outputs = self.text_encoder(**texts)
            else:
                outputs = self.text_encoder(input_ids=texts)
            features = outputs.pooler_output
        
        return self.text_proj(features)
    
    def forward(
        self,
        images: torch.Tensor,
        texts,
        return_features: bool = False,
    ) -> dict:
        """
        Forward pass.
        
        Args:
            images: Batch of images [B, C, H, W]
            texts: Batch of text tokens or tokenized texts
            return_features: Whether to return raw features
        
        Returns:
            Dictionary with logits, features, etc.
        """
        # Encode images and texts
        image_features = self.encode_image(images)
        text_features = self.encode_text(texts)
        
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute similarity
        logits_per_image = (image_features @ text_features.T) * self.temperature
        logits_per_text = logits_per_image.T
        
        output = {
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_text,
        }
        
        if return_features:
            output["image_features"] = image_features
            output["text_features"] = text_features
        
        return output


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

