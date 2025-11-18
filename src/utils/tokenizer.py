"""
Text tokenization utilities.
"""

from typing import Optional, Union, List
import clip as openai_clip
from transformers import CLIPProcessor, AutoTokenizer


def get_tokenizer(
    model_type: str = "clip",
    text_model: str = "bert-base-uncased",
    max_length: int = 77,
) -> callable:
    """
    Get tokenizer function.
    
    Args:
        model_type: Type of model (clip, bert, etc.)
        text_model: Name of the text model
        max_length: Maximum sequence length
    
    Returns:
        Tokenizer function
    """
    if model_type.lower() == "clip":
        try:
            # Try OpenAI CLIP tokenizer
            _, _, preprocess = openai_clip.load("ViT-B/32", device="cpu")
            tokenizer = openai_clip.tokenize
            return lambda texts: tokenizer(texts, truncate=True).to("cpu")
        except:
            # Fallback to HuggingFace CLIP
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            return lambda texts: processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
    else:
        # Use HuggingFace tokenizer
        tokenizer = AutoTokenizer.from_pretrained(text_model)
        return lambda texts: tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

