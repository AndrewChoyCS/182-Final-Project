#!/usr/bin/env python3
"""
Evaluation script for CS182 Final Project.
"""

import argparse
import torch
from pathlib import Path

from src.utils.config import load_config
from src.models.model_factory import get_model
from src.data import get_dataset, get_dataloader, get_transforms
from src.utils.tokenizer import get_tokenizer
from src.evaluation import Evaluator, evaluate_zero_shot, evaluate_linear_probe


def main():
    parser = argparse.ArgumentParser(description="Evaluate model for CS182 Final Project")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--zero_shot",
        action="store_true",
        help="Run zero-shot evaluation",
    )
    parser.add_argument(
        "--linear_probe",
        action="store_true",
        help="Run linear probe evaluation",
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    device = config.get("experiment", {}).get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    
    # Get data transforms
    data_config = config.get("data", {})
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
    
    # Load dataset
    dataset_name = data_config.get("dataset", "cifar10")
    data_dir = data_config.get("data_dir", "./data")
    
    val_dataset = get_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        split="val",
        transform=val_transform,
        tokenizer=tokenizer,
    )
    
    val_loader = get_dataloader(
        val_dataset,
        batch_size=data_config.get("val_batch_size", 64),
        shuffle=False,
        num_workers=config.get("experiment", {}).get("num_workers", 4),
    )
    
    # Create model
    model = get_model(model_config)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from {args.checkpoint}")
    
    # Evaluation configuration
    eval_config = config.get("evaluation", {})
    metrics = eval_config.get("metrics", ["accuracy", "top5_accuracy"])
    
    # Standard evaluation
    evaluator = Evaluator(model, device=device)
    results = evaluator.evaluate(val_loader, metrics=metrics)
    
    print("\n=== Evaluation Results ===")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    
    # Zero-shot evaluation
    if args.zero_shot or eval_config.get("zero_shot", False):
        if hasattr(val_dataset, "class_names"):
            class_names = val_dataset.class_names
        elif dataset_name.lower() == "cifar10":
            class_names = [
                "airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"
            ]
        else:
            class_names = [str(i) for i in range(data_config.get("num_classes", 10))]
        
        print("\n=== Zero-shot Evaluation ===")
        zero_shot_results = evaluate_zero_shot(
            model=model,
            dataloader=val_loader,
            class_names=class_names,
            device=device,
        )
        for metric, value in zero_shot_results.items():
            print(f"{metric}: {value:.4f}")
    
    # Linear probe evaluation
    if args.linear_probe or eval_config.get("linear_probe", False):
        # Need train loader for linear probe
        train_transform = get_transforms(
            image_size=data_config.get("image_size", 224),
            augment=False,  # No augmentation for feature extraction
        )
        
        train_dataset = get_dataset(
            dataset_name=dataset_name,
            data_dir=data_dir,
            split="train",
            transform=train_transform,
            tokenizer=tokenizer,
        )
        
        train_loader = get_dataloader(
            train_dataset,
            batch_size=data_config.get("batch_size", 32),
            shuffle=True,
            num_workers=config.get("experiment", {}).get("num_workers", 4),
        )
        
        print("\n=== Linear Probe Evaluation ===")
        linear_probe_results = evaluate_linear_probe(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=data_config.get("num_classes", 10),
            device=device,
        )
        for metric, value in linear_probe_results.items():
            print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()

