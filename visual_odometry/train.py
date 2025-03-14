#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training script for the visual odometry system.

This script handles:
- Setting up the DataLoader using dataset_distribution.py
- Initializing the model, optimizer, and loss function
- Implementing a training loop with validation
- Saving checkpoints for the best-performing model
- Logging metrics (loss, translation error, rotation error)
- Implementing learning rate scheduling
- Supporting resuming from checkpoints
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
import argparse
from typing import Dict, List, Tuple, Optional, Union
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from visual_odometry.config import (
    MODEL_CONFIG, TRAIN_CONFIG, DATASET_CONFIG, DEVICE, LOG_CONFIG, create_directories
)
from visual_odometry.models.base_model import create_model
from visual_odometry.models.loss import create_loss_function
from visual_odometry.dataset import (
    DatasetConfig, RGBDScenesDataset, create_dataloaders
)
from visual_odometry.utils.evaluation import evaluate_trajectory
from visual_odometry.utils.visualization import plot_trajectory_3d

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_CONFIG["log_level"]),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, verbose: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait after validation loss stops improving
            min_delta: Minimum change in validation loss to qualify as improvement
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should be stopped.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if training should be stopped, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            # Validation loss improved
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            # Validation loss did not improve
            self.counter += 1
            if self.verbose:
                logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False


def create_optimizer(model: nn.Module, config: Dict) -> optim.Optimizer:
    """
    Create optimizer for the model.
    
    Args:
        model: Model to optimize
        config: Training configuration
        
    Returns:
        Initialized optimizer
    """
    return optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )


def create_scheduler(optimizer: optim.Optimizer, config: Dict) -> optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        config: Training configuration
        
    Returns:
        Initialized scheduler
    """
    scheduler_config = config["scheduler"]
    scheduler_type = scheduler_config["type"]
    
    if scheduler_type == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_config["factor"],
            patience=scheduler_config["patience"],
            min_lr=scheduler_config["min_lr"],
            verbose=True
        )
    elif scheduler_type == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get("step_size", 10),
            gamma=scheduler_config["factor"],
            verbose=True
        )
    elif scheduler_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["epochs"],
            eta_min=scheduler_config["min_lr"],
            verbose=True
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    loss: float,
    best_loss: float,
    save_path: str,
    is_best: bool = False
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state to save
        scheduler: Scheduler state to save
        epoch: Current epoch
        loss: Current loss
        best_loss: Best loss so far
        save_path: Path to save the checkpoint
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'loss': loss,
        'best_loss': best_loss
    }
    
    # Save checkpoint
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved to {save_path}")
    
    # If this is the best model, save a copy
    if is_best:
        best_path = os.path.join(os.path.dirname(save_path), 'best_model.pth')
        torch.save(checkpoint, best_path)
        logger.info(f"Best model saved to {best_path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    checkpoint_path: str
) -> Tuple[nn.Module, optim.Optimizer, optim.lr_scheduler._LRScheduler, int, float]:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        checkpoint_path: Path to the checkpoint
        
    Returns:
        Tuple containing:
        - Loaded model
        - Loaded optimizer
        - Loaded scheduler
        - Epoch
        - Best loss
    """
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return model, optimizer, scheduler, 0, float('inf')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if available
    if scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Get epoch and loss
    epoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']
    
    logger.info(f"Loaded checkpoint from epoch {epoch} with loss {checkpoint['loss']:.6f}")
    
    return model, optimizer, scheduler, epoch, best_loss


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    log_interval: int = 10
) -> Dict[str, float]:
    """
    Train the model for one epoch.
    
    Args:
        model: Model to train
        dataloader: DataLoader for training data
        loss_fn: Loss function
        optimizer: Optimizer
        device: Device to train on
        log_interval: Interval for logging
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    # Metrics for tracking
    translation_errors = []
    rotation_errors = []
    
    # Progress bar
    pbar = tqdm(dataloader, desc="Training")
    
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        for key in batch:
            batch[key] = batch[key].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        pred_pose = model(batch)
        
        # Compute loss
        if isinstance(loss_fn, nn.MSELoss) or isinstance(loss_fn, nn.modules.loss.MSELoss):
            loss = loss_fn(pred_pose, batch['rel_pose'])
            loss_components = {'total_loss': loss}
        else:
            loss, loss_components = loss_fn(pred_pose, batch['rel_pose'])
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update metrics
        batch_size = batch['rgb1'].size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        # Compute translation and rotation errors
        with torch.no_grad():
            # Translation error (Euclidean distance)
            trans_error = torch.norm(pred_pose[:, 4:] - batch['rel_pose'][:, 4:], dim=1)
            translation_errors.append(trans_error.cpu().numpy())
            
            # Rotation error (angular distance between quaternions)
            pred_q = pred_pose[:, :4]
            target_q = batch['rel_pose'][:, :4]
            
            # Normalize quaternions
            pred_q = torch.nn.functional.normalize(pred_q, p=2, dim=1)
            target_q = torch.nn.functional.normalize(target_q, p=2, dim=1)
            
            # Compute dot product
            dot_product = torch.sum(pred_q * target_q, dim=1).abs()
            dot_product = torch.clamp(dot_product, -1.0, 1.0)
            
            # Convert to angle in degrees
            rot_error = 2 * torch.acos(dot_product) * 180.0 / torch.pi
            rotation_errors.append(rot_error.cpu().numpy())
        
        # Update progress bar
        if batch_idx % log_interval == 0:
            pbar.set_postfix({
                'loss': loss.item(),
                'trans_err': trans_error.mean().item(),
                'rot_err': rot_error.mean().item()
            })
    
    # Compute average metrics
    avg_loss = total_loss / total_samples
    avg_translation_error = np.mean(np.concatenate(translation_errors))
    avg_rotation_error = np.mean(np.concatenate(rotation_errors))
    
    metrics = {
        'loss': avg_loss,
        'translation_error': avg_translation_error,
        'rotation_error': avg_rotation_error
    }
    
    # Add loss components if available
    if isinstance(loss_components, dict):
        for key, value in loss_components.items():
            if isinstance(value, torch.Tensor):
                metrics[key] = value.item()
    
    return metrics


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Validate the model.
    
    Args:
        model: Model to validate
        dataloader: DataLoader for validation data
        loss_fn: Loss function
        device: Device to validate on
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    # Metrics for tracking
    translation_errors = []
    rotation_errors = []
    
    # Collect all predictions and ground truth for trajectory evaluation
    all_pred_poses = []
    all_gt_poses = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            # Move data to device
            for key in batch:
                batch[key] = batch[key].to(device)
            
            # Forward pass
            pred_pose = model(batch)
            
            # Compute loss
            if isinstance(loss_fn, nn.MSELoss) or isinstance(loss_fn, nn.modules.loss.MSELoss):
                loss = loss_fn(pred_pose, batch['rel_pose'])
                loss_components = {'total_loss': loss}
            else:
                loss, loss_components = loss_fn(pred_pose, batch['rel_pose'])
            
            # Update metrics
            batch_size = batch['rgb1'].size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Compute translation and rotation errors
            # Translation error (Euclidean distance)
            trans_error = torch.norm(pred_pose[:, 4:] - batch['rel_pose'][:, 4:], dim=1)
            translation_errors.append(trans_error.cpu().numpy())
            
            # Rotation error (angular distance between quaternions)
            pred_q = pred_pose[:, :4]
            target_q = batch['rel_pose'][:, :4]
            
            # Normalize quaternions
            pred_q = torch.nn.functional.normalize(pred_q, p=2, dim=1)
            target_q = torch.nn.functional.normalize(target_q, p=2, dim=1)
            
            # Compute dot product
            dot_product = torch.sum(pred_q * target_q, dim=1).abs()
            dot_product = torch.clamp(dot_product, -1.0, 1.0)
            
            # Convert to angle in degrees
            rot_error = 2 * torch.acos(dot_product) * 180.0 / torch.pi
            rotation_errors.append(rot_error.cpu().numpy())
            
            # Collect poses for trajectory evaluation
            all_pred_poses.append(pred_pose.cpu().numpy())
            all_gt_poses.append(batch['rel_pose'].cpu().numpy())
    
    # Compute average metrics
    avg_loss = total_loss / total_samples
    avg_translation_error = np.mean(np.concatenate(translation_errors))
    avg_rotation_error = np.mean(np.concatenate(rotation_errors))
    
    metrics = {
        'loss': avg_loss,
        'translation_error': avg_translation_error,
        'rotation_error': avg_rotation_error
    }
    
    # Add loss components if available
    if isinstance(loss_components, dict):
        for key, value in loss_components.items():
            if isinstance(value, torch.Tensor):
                metrics[key] = value.item()
    
    # Concatenate all poses
    all_pred_poses = np.concatenate(all_pred_poses, axis=0)
    all_gt_poses = np.concatenate(all_gt_poses, axis=0)
    
    # Evaluate trajectory
    # Note: This is a simplified evaluation since we're using relative poses
    # For a complete evaluation, we would need to reconstruct the full trajectory
    trajectory_metrics = evaluate_trajectory(
        all_pred_poses[:100],  # Use a subset for efficiency
        all_gt_poses[:100],
        metrics=['translation_error', 'rotation_error']
    )
    
    # Add trajectory metrics
    metrics['trajectory_translation_error'] = trajectory_metrics['translation_error']['mean']
    metrics['trajectory_rotation_error'] = trajectory_metrics['rotation_error']['mean']
    
    return metrics


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    device: torch.device,
    config: Dict,
    start_epoch: int = 0,
    best_loss: float = float('inf'),
    writer: Optional[SummaryWriter] = None,
    use_multi_gpu: bool = True
) -> nn.Module:
    """
    Train the model.
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        loss_fn: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        config: Training configuration
        start_epoch: Starting epoch (for resuming training)
        best_loss: Best validation loss (for resuming training)
        writer: TensorBoard writer
        use_multi_gpu: Whether to use multiple GPUs for training
        
    Returns:
        Trained model
    """
    # Create directories
    checkpoint_dir = config["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=config["early_stopping_patience"])
    
    # Move model to device
    model = model.to(device)
    
    # Use multiple GPUs if available and requested
    if use_multi_gpu and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for training")
        model = DataParallel(model)
    
    # Training loop
    for epoch in range(start_epoch, config["epochs"]):
        logger.info(f"Epoch {epoch+1}/{config['epochs']}")
        
        # Train for one epoch
        train_metrics = train_epoch(
            model, train_loader, loss_fn, optimizer, device, config["log_interval"]
        )
        
        # Validate
        val_metrics = validate(model, val_loader, loss_fn, device)
        
        # Log metrics
        logger.info(f"Train Loss: {train_metrics['loss']:.6f}, "
                   f"Train Translation Error: {train_metrics['translation_error']:.6f} m, "
                   f"Train Rotation Error: {train_metrics['rotation_error']:.6f} deg")
        
        logger.info(f"Val Loss: {val_metrics['loss']:.6f}, "
                   f"Val Translation Error: {val_metrics['translation_error']:.6f} m, "
                   f"Val Rotation Error: {val_metrics['rotation_error']:.6f} deg")
        
        # Log to TensorBoard
        if writer is not None:
            # Log training metrics
            for key, value in train_metrics.items():
                writer.add_scalar(f"train/{key}", value, epoch)
            
            # Log validation metrics
            for key, value in val_metrics.items():
                writer.add_scalar(f"val/{key}", value, epoch)
            
            # Log learning rate
            writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], epoch)
        
        # Update learning rate scheduler
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
        
        # Check if this is the best model
        is_best = val_metrics['loss'] < best_loss
        if is_best:
            best_loss = val_metrics['loss']
        
        # Save checkpoint
        if config["save_best_only"] and not is_best:
            # Skip saving if we only want to save the best model and this is not it
            pass
        else:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
            
            # Save model (handle DataParallel wrapper)
            model_to_save = model.module if isinstance(model, DataParallel) else model
            
            save_checkpoint(
                model_to_save, optimizer, scheduler, epoch+1, val_metrics['loss'], best_loss,
                checkpoint_path, is_best
            )
        
        # Check early stopping
        if early_stopping(val_metrics['loss']):
            logger.info(f"Early stopping triggered after epoch {epoch+1}")
            break
    
    # Return the base model (not the DataParallel wrapper)
    if isinstance(model, DataParallel):
        return model.module
    else:
        return model


def main(args):
    """Main function for training."""
    # Create directories
    create_directories()
    
    # Set up TensorBoard writer
    if LOG_CONFIG["tensorboard"]:
        writer = SummaryWriter(log_dir=LOG_CONFIG["log_dir"])
    else:
        writer = None
    
    # Create dataset configuration
    dataset_config = DatasetConfig(
        img_size=DATASET_CONFIG["img_size"],
        max_depth=DATASET_CONFIG["max_depth"]
    )
    
    # Create dataloaders
    dataloaders = create_dataloaders(
        dataset_config,
        batch_size=TRAIN_CONFIG["batch_size"],
        num_workers=TRAIN_CONFIG["num_workers"]
    )
    
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    
    # Create model
    model = create_model(MODEL_CONFIG)
    
    # Create loss function
    loss_fn = create_loss_function(
        TRAIN_CONFIG.get("loss_type", "pose"),
        {
            'rotation_weight': TRAIN_CONFIG["rotation_weight"],
            'translation_weight': TRAIN_CONFIG["translation_weight"]
        }
    )
    
    # Create optimizer
    optimizer = create_optimizer(model, TRAIN_CONFIG)
    
    # Create scheduler
    scheduler = create_scheduler(optimizer, TRAIN_CONFIG)
    
    # Initialize variables for resuming training
    start_epoch = 0
    best_loss = float('inf')
    
    # Resume from checkpoint if specified
    if args.resume:
        model, optimizer, scheduler, start_epoch, best_loss = load_checkpoint(
            model, optimizer, scheduler, args.resume
        )
    
    # Train the model
    model = train(
        model, train_loader, val_loader, loss_fn, optimizer, scheduler,
        DEVICE, TRAIN_CONFIG, start_epoch, best_loss, writer, 
        use_multi_gpu=getattr(args, 'use_multi_gpu', True)  # Default to True if not specified
    )
    
    # Close TensorBoard writer
    if writer is not None:
        writer.close()
    
    return model


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train visual odometry model")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    # Train the model
    model = main(args) 