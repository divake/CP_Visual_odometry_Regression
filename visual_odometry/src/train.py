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
        'metrics': metrics
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
    Main training function.
    
    Args:
        resume_from (str, optional): Path to checkpoint to resume from
        
    Returns:
        tuple: (model, train_losses, val_losses)
    """
    # Create directories if they don't exist
    os.makedirs(os.path.join(config.RESULTS_DIR, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(config.RESULTS_DIR, "logs"), exist_ok=True)
    
    # Set device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, _ = create_data_loaders()
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model, optimizer, loss function, and scheduler
    model = get_model(model_type="standard", pretrained=config.RESNET_PRETRAINED)
    model = model.to(device)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    loss_fn = get_loss_function(loss_type="combined")
    
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.LR_SCHEDULER_STEP_SIZE,
        gamma=config.LR_SCHEDULER_GAMMA
    )
    
    # Initialize training variables
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_metrics = []
    val_metrics = []
    
    # Resume from checkpoint if specified
    if resume_from:
        if os.path.exists(resume_from):
            print(f"Resuming from checkpoint: {resume_from}")
            model, optimizer, scheduler, start_epoch, best_val_loss, metrics = load_checkpoint(
                model, optimizer, scheduler, resume_from
            )
            
            if 'train_losses' in metrics:
                train_losses = metrics['train_losses']
            
            if 'val_losses' in metrics:
                val_losses = metrics['val_losses']
            
            if 'train_metrics' in metrics:
                train_metrics = metrics['train_metrics']
            
            if 'val_metrics' in metrics:
                val_metrics = metrics['val_metrics']
            
            start_epoch += 1  # Start from the next epoch
        else:
            print(f"Warning: Checkpoint file {resume_from} not found. Starting from scratch.")
    
    # Training loop
    print(f"Starting training from epoch {start_epoch + 1}/{config.NUM_EPOCHS}")
    
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")
        
        # Train one epoch
        train_metric = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        train_losses.append(train_metric['loss'])
        train_metrics.append(train_metric)
        
        # Validate
        val_metric = validate(model, val_loader, loss_fn, device)
        val_losses.append(val_metric['loss'])
        val_metrics.append(val_metric)
        
        # Update learning rate
        scheduler.step()
        
        # Print metrics
        current_lr = scheduler.get_last_lr()[0]
        print(f"LR: {current_lr:.6f}, "
              f"Train Loss: {train_metric['loss']:.4f}, "
              f"Val Loss: {val_metric['loss']:.4f}, "
              f"Train Trans Loss: {train_metric['translation_loss']:.4f}, "
              f"Val Trans Loss: {val_metric['translation_loss']:.4f}, "
              f"Train Rot Loss: {train_metric['rotation_loss']:.4f}, "
              f"Val Rot Loss: {val_metric['rotation_loss']:.4f}")
        
        # Save checkpoint
        is_best = val_metric['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metric['loss']
        
        # Save metrics for checkpoint
        metrics = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }
        
        # Save checkpoint
        if (epoch + 1) % config.SAVE_CHECKPOINT_FREQ == 0 or is_best or epoch == config.NUM_EPOCHS - 1:
            checkpoint_path = os.path.join(config.RESULTS_DIR, "checkpoints", f"checkpoint_epoch_{epoch+1}.pth")
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, metrics, is_best, checkpoint_path)
        
        # Generate and save loss plot
        if (epoch + 1) % config.SAVE_CHECKPOINT_FREQ == 0 or epoch == config.NUM_EPOCHS - 1:
            plot_loss_curves(train_losses, val_losses, config.LOSS_PLOT_PATH)
            
            # Save loss values to file
            with open(os.path.join(config.RESULTS_DIR, "logs", "losses.json"), 'w') as f:
                json.dump({
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics
                }, f, indent=4)
    
    print("\nTraining complete!")
    return model, train_losses, val_losses


if __name__ == "__main__":
    # Run training
    model, train_losses, val_losses = train()
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(config.RESULTS_DIR, "checkpoints", "final_model.pth"))
    
    # Plot final loss curves
    plot_loss_curves(train_losses, val_losses, config.LOSS_PLOT_PATH) 