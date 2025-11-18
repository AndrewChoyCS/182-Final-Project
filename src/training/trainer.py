"""
Training utilities and trainer class.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from ..models.clip_model import clip_loss
from ..utils.metrics import compute_accuracy, compute_topk_accuracy


class Trainer:
    """Trainer class for model training."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or {}
        self.device = device
        
        # Training configuration
        self.epochs = self.config.get("epochs", 10)
        self.learning_rate = self.config.get("learning_rate", 1e-4)
        self.weight_decay = self.config.get("weight_decay", 0.01)
        self.warmup_steps = self.config.get("warmup_steps", 1000)
        self.gradient_accumulation_steps = self.config.get("gradient_accumulation_steps", 1)
        self.mixed_precision = self.config.get("mixed_precision", False)
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        # Setup scheduler
        total_steps = len(train_loader) * self.epochs // self.gradient_accumulation_steps
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - self.warmup_steps,
            eta_min=1e-6,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_steps],
        )
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision else None
        
        # Logging
        self.save_dir = Path(self.config.get("save_dir", "./checkpoints"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_every = self.config.get("log_every", 100)
        self.save_every = self.config.get("save_every", 1)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_acc = 0.0
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            images = batch["image"].to(self.device)
            texts = batch.get("text")
            if texts is not None and isinstance(texts, torch.Tensor):
                texts = texts.to(self.device)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                if isinstance(texts, dict):
                    # HuggingFace tokenizer output
                    texts = {k: v.to(self.device) for k, v in texts.items()}
                
                outputs = self.model(images, texts)
                
                # Compute loss
                if "logits_per_image" in outputs:
                    # CLIP loss
                    loss = clip_loss(
                        outputs["logits_per_image"],
                        outputs["logits_per_text"],
                    )
                else:
                    # Classification loss
                    labels = batch["label"].to(self.device)
                    loss = nn.functional.cross_entropy(outputs, labels)
            
            # Backward pass
            loss = loss / self.gradient_accumulation_steps
            
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Logging
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            if self.global_step % self.log_every == 0:
                pbar.set_postfix({
                    "loss": f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                })
        
        avg_loss = total_loss / num_batches
        return {"loss": avg_loss}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        num_batches = 0
        
        pbar = tqdm(self.val_loader, desc="Validation")
        
        for batch in pbar:
            images = batch["image"].to(self.device)
            texts = batch.get("text")
            labels = batch.get("label")
            
            if texts is not None and isinstance(texts, torch.Tensor):
                texts = texts.to(self.device)
            
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                if isinstance(texts, dict):
                    texts = {k: v.to(self.device) for k, v in texts.items()}
                
                outputs = self.model(images, texts)
                
                # Compute loss
                if "logits_per_image" in outputs:
                    loss = clip_loss(
                        outputs["logits_per_image"],
                        outputs["logits_per_text"],
                    )
                    # For CLIP, compute accuracy from similarity
                    preds = outputs["logits_per_image"].argmax(dim=1)
                else:
                    if labels is not None:
                        labels = labels.to(self.device)
                        loss = nn.functional.cross_entropy(outputs, labels)
                        preds = outputs.argmax(dim=1)
                    else:
                        loss = torch.tensor(0.0)
                        preds = outputs.argmax(dim=1)
            
            total_loss += loss.item()
            num_batches += 1
            
            if labels is not None:
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
        
        metrics = {"loss": total_loss / num_batches}
        
        if all_preds and all_labels:
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            metrics["accuracy"] = compute_accuracy(all_preds, all_labels)
            metrics["top5_accuracy"] = compute_topk_accuracy(all_preds, all_labels, k=5)
        
        return metrics
    
    def save_checkpoint(self, filename: str = "checkpoint.pth"):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_acc": self.best_val_acc,
        }
        
        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        filepath = self.save_dir / filename
        torch.save(checkpoint, filepath)
        self.logger.info(f"Saved checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_acc = checkpoint.get("best_val_acc", 0.0)
        
        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        self.logger.info(f"Loaded checkpoint from {filepath}")
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            self.logger.info(
                f"Epoch {epoch + 1}/{self.epochs} - Train Loss: {train_metrics['loss']:.4f}"
            )
            
            # Validate
            if self.val_loader is not None:
                val_metrics = self.validate()
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.epochs} - Val Loss: {val_metrics.get('loss', 0):.4f}, "
                    f"Val Acc: {val_metrics.get('accuracy', 0):.4f}"
                )
                
                # Save best model
                if val_metrics.get("accuracy", 0) > self.best_val_acc:
                    self.best_val_acc = val_metrics["accuracy"]
                    self.save_checkpoint("best_model.pth")
            
            # Save checkpoint
            if (epoch + 1) % self.save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth")
        
        self.logger.info("Training completed!")

