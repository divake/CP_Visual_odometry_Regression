#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration file for the Visual Odometry system.

This file contains all the configuration parameters for:
- Model architecture
- Training hyperparameters
- Dataset settings
- Evaluation metrics
- Visualization options
"""

import os
import torch
from typing import Dict, List, Tuple, Optional

# Dataset paths
DATASET_ROOT = "/ssd_4TB/divake/LSF_regression/dataset_rgbd_scenes_v2"
IMGS_DIR = os.path.join(DATASET_ROOT, "imgs")
PC_DIR = os.path.join(DATASET_ROOT, "pc")

# Default scene-based splits
DEFAULT_SPLITS = {
    "train": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"],
    "val": ["11"],
    "test": ["12", "13", "14"]
}

# Model configuration
MODEL_CONFIG = {
    "backbone": "resnet18",  # Options: resnet18, resnet34, resnet50
    "pretrained": True,      # Use ImageNet pretrained weights
    "input_channels": 4,     # RGB + Depth
    "fc_layers": [512*2, 256, 128, 7],  # Fully connected layers dimensions
    "dropout_rate": 0.3,     # Increased dropout rate for better regularization
    "init_method": "kaiming"  # Weight initialization method: xavier, kaiming
}

# Training configuration
TRAIN_CONFIG = {
    "batch_size": 16,
    "num_workers": 4,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,    # Increased weight decay for better regularization
    "epochs": 100,
    "early_stopping_patience": 15,  # Increased patience
    "rotation_weight": 10.0,  # Weight for rotation loss component
    "translation_weight": 1.0,  # Weight for translation loss component
    "loss_type": "improved_pose",  # Use our improved pose loss
    "scheduler": {
        "type": "plateau",  # Options: plateau, step, cosine
        "patience": 5,
        "factor": 0.5,
        "min_lr": 1e-6
    },
    "checkpoint_dir": "checkpoints",
    "log_interval": 10,  # Log training metrics every N batches
    "save_best_only": True  # Only save the best model based on validation loss
}

# Dataset configuration
DATASET_CONFIG = {
    "img_size": (640, 480),  # (width, height)
    "max_depth": 10.0,       # Maximum depth value in meters
    "max_frame_gap": 1,      # Maximum frame gap for consecutive pairs
    "augmentation": {
        "enabled": True,
        "brightness": 0.2,   # Brightness adjustment range
        "contrast": 0.2,     # Contrast adjustment range
        "hue": 0.1,          # Hue adjustment range
        "saturation": 0.2,   # Saturation adjustment range
        "random_crop": False  # Whether to use random cropping
    }
}

# Evaluation configuration
EVAL_CONFIG = {
    "metrics": ["ate", "rpe", "translation_error", "rotation_error", "drift"],
    "visualization": {
        "plot_trajectory": True,
        "plot_errors": True,
        "save_visualizations": True,
        "visualization_dir": "visualizations"
    },
    "report_file": "evaluation_report.json"
}

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logging configuration
LOG_CONFIG = {
    "log_dir": "logs",
    "tensorboard": True,
    "log_level": "INFO"
}

# Create necessary directories
def create_directories():
    """Create necessary directories for the project."""
    dirs = [
        TRAIN_CONFIG["checkpoint_dir"],
        EVAL_CONFIG["visualization"]["visualization_dir"],
        LOG_CONFIG["log_dir"]
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True) 