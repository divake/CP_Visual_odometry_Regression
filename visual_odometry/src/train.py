#!/usr/bin/env python3
"""
Training script for the Visual Odometry model.

This script handles the training loop for the visual odometry model, including:
- Loading the dataset and model
- Setting up the optimizer and scheduler
- Training for a specified number of epochs
- Validating the model at regular intervals
- Saving checkpoints and logs
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

import config
from dataloader import create_data_loaders
from model import get_model
from loss import get_loss_function
from utils import plot_loss_curves


def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model (nn.Module): The model to train
        train_loader (DataLoader): DataLoader for training data
        loss_fn (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        device (str): Device to use for training
        
    Returns:
        dict: Dictionary containing training metrics:
            - 'loss': Average total loss
            - 'translation_loss': Average translation loss
            - 'rotation_loss': Average rotation loss
            - 'quat_norm_loss': Average quaternion normalization loss
    """
    model.train()
    total_loss = 0.0
    total_translation_loss = 0.0
    total_rotation_loss = 0.0
    total_quat_norm_loss = 0.0
    num_batches = len(train_loader)
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch_idx, (img1, img2, rel_pose) in enumerate(progress_bar):
        # Move data to device
        img1 = img1.to(device)
        img2 = img2.to(device)
        
        # Split relative pose into rotation and translation
        gt_rotation = rel_pose[:, :4].to(device)  # w, x, y, z
        gt_translation = rel_pose[:, 4:7].to(device)  # x, y, z
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        pred_rotation, pred_translation = model(img1, img2)
        
        # Compute loss
        loss, translation_loss, rotation_loss, quat_norm_loss = loss_fn(
            pred_rotation, pred_translation, gt_rotation, gt_translation
        )
        
        # Backward pass and optimize
        loss.backward()
        
        # Apply gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_VALUE)
        
        optimizer.step()
        
        # Update losses
        total_loss += loss.item()
        total_translation_loss += translation_loss.item()
        total_rotation_loss += rotation_loss.item()
        total_quat_norm_loss += quat_norm_loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'trans_loss': f"{translation_loss.item():.4f}",
            'rot_loss': f"{rotation_loss.item():.4f}"
        })
    
    # Calculate average losses
    avg_loss = total_loss / num_batches
    avg_translation_loss = total_translation_loss / num_batches
    avg_rotation_loss = total_rotation_loss / num_batches
    avg_quat_norm_loss = total_quat_norm_loss / num_batches
    
    metrics = {
        'loss': avg_loss,
        'translation_loss': avg_translation_loss,
        'rotation_loss': avg_rotation_loss,
        'quat_norm_loss': avg_quat_norm_loss
    }
    
    return metrics


def validate(model, val_loader, loss_fn, device):
    """
    Validate the model.
    
    Args:
        model (nn.Module): The model to validate
        val_loader (DataLoader): DataLoader for validation data
        loss_fn (nn.Module): Loss function
        device (str): Device to use for validation
        
    Returns:
        dict: Dictionary containing validation metrics:
            - 'loss': Average total loss
            - 'translation_loss': Average translation loss
            - 'rotation_loss': Average rotation loss
            - 'quat_norm_loss': Average quaternion normalization loss
    """
    model.eval()
    total_loss = 0.0
    total_translation_loss = 0.0
    total_rotation_loss = 0.0
    total_quat_norm_loss = 0.0
    num_batches = len(val_loader)
    
    progress_bar = tqdm(val_loader, desc="Validation")
    
    with torch.no_grad():
        for batch_idx, (img1, img2, rel_pose) in enumerate(progress_bar):
            # Move data to device
            img1 = img1.to(device)
            img2 = img2.to(device)
            
            # Split relative pose into rotation and translation
            gt_rotation = rel_pose[:, :4].to(device)  # w, x, y, z
            gt_translation = rel_pose[:, 4:7].to(device)  # x, y, z
            
            # Forward pass
            pred_rotation, pred_translation = model(img1, img2)
            
            # Compute loss
            loss, translation_loss, rotation_loss, quat_norm_loss = loss_fn(
                pred_rotation, pred_translation, gt_rotation, gt_translation
            )
            
            # Update losses
            total_loss += loss.item()
            total_translation_loss += translation_loss.item()
            total_rotation_loss += rotation_loss.item()
            total_quat_norm_loss += quat_norm_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'val_loss': f"{loss.item():.4f}",
                'val_trans_loss': f"{translation_loss.item():.4f}",
                'val_rot_loss': f"{rotation_loss.item():.4f}"
            })
    
    # Calculate average losses
    avg_loss = total_loss / num_batches
    avg_translation_loss = total_translation_loss / num_batches
    avg_rotation_loss = total_rotation_loss / num_batches
    avg_quat_norm_loss = total_quat_norm_loss / num_batches
    
    metrics = {
        'loss': avg_loss,
        'translation_loss': avg_translation_loss,
        'rotation_loss': avg_rotation_loss,
        'quat_norm_loss': avg_quat_norm_loss
    }
    
    return metrics


def save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, metrics, is_best, filename):
    """
    Save a checkpoint of the model.
    
    Args:
        model (nn.Module): The model to save
        optimizer (torch.optim.Optimizer): The optimizer
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler
        epoch (int): Current epoch
        best_val_loss (float): Best validation loss so far
        metrics (dict): Dictionary of metrics to save
        is_best (bool): Whether this is the best model so far
        filename (str): Path to save the checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_val_loss': best_val_loss,
        'metrics': metrics,
        'model_type': config.MODEL_TYPE  # Add model type for compatibility
    }
    
    torch.save(checkpoint, filename)
    
    if is_best:
        best_filename = os.path.join(os.path.dirname(filename), 'best_model.pth')
        torch.save(checkpoint, best_filename)


def load_checkpoint(model, optimizer=None, scheduler=None, filename=None):
    """
    Load a checkpoint.
    
    Args:
        model (nn.Module): The model to load weights into
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The scheduler to load state into
        filename (str, optional): Path to the checkpoint file
        
    Returns:
        tuple: (model, optimizer, scheduler, epoch, best_val_loss, metrics)
    """
    if not os.path.exists(filename):
        return model, optimizer, scheduler, 0, float('inf'), {}
    
    checkpoint = torch.load(filename)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    metrics = checkpoint.get('metrics', {})
    
    return model, optimizer, scheduler, epoch, best_val_loss, metrics


def train(resume_from=None):
    """
    Train the visual odometry model.
    
    Args:
        resume_from (str, optional): Path to checkpoint to resume training from
        
    Returns:
        str: Path to the best model checkpoint
    """
    # Create output directories if they don't exist
    os.makedirs(os.path.join(config.RESULTS_DIR, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(config.RESULTS_DIR, "logs"), exist_ok=True)
    
    # Set device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, _ = create_data_loaders(
        dataset_path=config.DATASET_PATH,
        scene_id=config.SCENE_ID,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        img_height=config.IMG_HEIGHT,
        img_width=config.IMG_WIDTH,
        use_augmentation=config.USE_AUGMENTATION
    )
    print(f"Created data loaders: {len(train_loader)} training batches, {len(val_loader)} validation batches")
    
    # Create model
    model = get_model(model_type=config.MODEL_TYPE, pretrained=config.USE_PRETRAINED)
    model = model.to(device)
    print(f"Created model: {config.MODEL_TYPE}")
    
    # Create loss function
    loss_fn = get_loss_function(loss_type=config.LOSS_TYPE)
    print(f"Using loss function: {config.LOSS_TYPE}")
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.LR_SCHEDULER_FACTOR,
        patience=config.LR_SCHEDULER_PATIENCE,
        verbose=True
    )
    
    # Initialize training variables
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_translation_losses = []
    val_translation_losses = []
    
    # If resuming from checkpoint
    if resume_from:
        print(f"Resuming training from {resume_from}")
        model, optimizer, scheduler, epoch, best_val_loss, metrics = load_checkpoint(
            model, optimizer, scheduler, resume_from
        )
        start_epoch = epoch + 1
        
        # Extract metrics from the checkpoint
        if metrics:
            if 'train_losses' in metrics:
                train_losses = metrics['train_losses']
            if 'val_losses' in metrics:
                val_losses = metrics['val_losses']
            if 'train_translation_losses' in metrics:
                train_translation_losses = metrics['train_translation_losses']
            if 'val_translation_losses' in metrics:
                val_translation_losses = metrics['val_translation_losses']
    
    # Training loop
    print(f"Starting training from epoch {start_epoch} for {config.EPOCHS} epochs")
    for epoch in range(start_epoch, start_epoch + config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{start_epoch + config.EPOCHS}")
        
        # Train
        train_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        train_loss = train_metrics['loss']
        train_translation_loss = train_metrics['translation_loss']
        
        # Validate
        val_metrics = validate(model, val_loader, loss_fn, device)
        val_loss = val_metrics['loss']
        val_translation_loss = val_metrics['translation_loss']
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_translation_losses.append(train_translation_loss)
        val_translation_losses.append(val_translation_loss)
        
        # Log metrics
        print(f"Train Loss: {train_loss:.6f}, Translation Loss: {train_translation_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}, Translation Loss: {val_translation_loss:.6f}")
        
        # Save metrics
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_translation_loss': train_translation_loss,
            'val_translation_loss': val_translation_loss,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print(f"New best validation loss: {best_val_loss:.6f}")
        
        # Save all training history
        metrics_history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_translation_losses': train_translation_losses,
            'val_translation_losses': val_translation_losses
        }
        
        # Combine metrics
        all_metrics = {**metrics, **metrics_history}
        
        # Save checkpoint
        checkpoint_path = os.path.join(config.RESULTS_DIR, 'checkpoints', f'checkpoint_epoch_{epoch+1}.pth')
        best_model_path = os.path.join(config.RESULTS_DIR, 'checkpoints', 'best_model.pth')
        save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, all_metrics, is_best, checkpoint_path)
        
    # Plot loss curves
    loss_curves_path = os.path.join(config.RESULTS_DIR, 'logs', 'loss_curves.png')
    plot_loss_curves(
        train_losses, 
        val_losses, 
        train_translation_losses, 
        val_translation_losses, 
        loss_curves_path
    )
    
    return best_model_path


if __name__ == "__main__":
    # Run training
    best_model_path = train()
    print(f"Training completed. Best model saved at: {best_model_path}") 