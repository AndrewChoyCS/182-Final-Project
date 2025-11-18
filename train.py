#!/usr/bin/env python3
"""
Main training script for CS182 Final Project.
"""

import argparse
import torch
import random
import numpy as np
from pathlib import Path

from src.utils.config import load_config
from src.models.model_factory import get_model
from src.data import get_dataset, get_dataloader, get_transforms
from src.utils.tokenizer import get_tokenizer
from src.training import Trainer


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Train model for CS182 Final Project")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set seed
    seed = config.get("experiment", {}).get("seed", 42)
    set_seed(seed)
    
    # Setup device
    device = config.get("experiment", {}).get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    
    # Get data transforms
    data_config = config.get("data", {})
    train_transform = get_transforms(
        image_size=data_config.get("image_size", 224),
        augment=data_config.get("augment", True),
    )
    val_transform = get_transforms(
        image_size=data_config.get("image_size", 224),
        augment=False,
    )
    
    # Get tokenizer
    model_config = config.get("model", {})
    tokenizer = get_tokenizer(
        model_type=model_config.get("type", "clip"),
        text_model=model_config.get("text_model", "bert-base-uncased"),
    )
    
    # Load datasets
    dataset_name = data_config.get("dataset", "cifar10")
    data_dir = data_config.get("data_dir", "./data")
    
    train_dataset = get_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        split="train",
        transform=train_transform,
        tokenizer=tokenizer,
    )
    
    val_dataset = get_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        split="val",
        transform=val_transform,
        tokenizer=tokenizer,
    )
    
    # Create data loaders
    train_loader = get_dataloader(
        train_dataset,
        batch_size=data_config.get("batch_size", 32),
        shuffle=True,
        num_workers=config.get("experiment", {}).get("num_workers", 4),
    )
    
    val_loader = get_dataloader(
        val_dataset,
        batch_size=data_config.get("val_batch_size", 64),
        shuffle=False,
        num_workers=config.get("experiment", {}).get("num_workers", 4),
    )
    
    # Create model
    model = get_model(model_config)
    print(f"Created model: {model_config.get('type', 'clip')}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    training_config = config.get("training", {})
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        device=device,
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()
    
    print("Training completed!")


if __name__ == "__main__":
    main()

