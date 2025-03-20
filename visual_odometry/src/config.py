#!/usr/bin/env python3
"""
Configuration settings for the Visual Odometry system.

This file contains all the configuration parameters used throughout the project,
including dataset paths, model parameters, training hyperparameters, and output settings.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_PATH = "/ssd_4TB/divake/LSF_regression/dataset_rgbd_scenes_v2"
RESULTS_DIR = os.path.join(BASE_DIR, "Results")

# Create necessary directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "visualizations"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "predictions"), exist_ok=True)

# Dataset settings
# Modified to focus only on scene_02 for both training and testing
SCENE_ID = "02"
TRAIN_SCENES = [SCENE_ID]  # Use only scene_02 for training
TEST_SCENES = [SCENE_ID]   # Use scene_02 for testing as well
TRAIN_VAL_SPLIT = 0.8  # 80% for training, 20% for validation
TRAIN_FRAME_RATIO = 0.8  # Use 80% of scene_02 frames for training
TEST_FULL_TRAJECTORY = True  # Test on 100% of the trajectory

# Image preprocessing
IMG_HEIGHT = 240
IMG_WIDTH = 320
NORMALIZE_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
NORMALIZE_STD = [0.229, 0.224, 0.225]   # ImageNet std
USE_AUGMENTATION = True  # Whether to use data augmentation during training

# Model parameters
MODEL_TYPE = "enhanced"  # Model type: "standard", "siamese", or "enhanced"
USE_PRETRAINED = True    # Whether to use pretrained weights
RESNET_PRETRAINED = True
FEATURE_DIMENSION = 512  # Output dimension of ResNet18 features
FC_DROPOUT = 0.3        # Dropout rate for fully connected layers (reduced from 0.5)

# Translation and rotation output dimensions
TRANSLATION_DIM = 3     # x, y, z
ROTATION_DIM = 4        # quaternion: w, x, y, z

# Training hyperparameters
BATCH_SIZE = 32
TEST_BATCH_SIZE = 32  # Batch size for testing
NUM_WORKERS = 4
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.00005  # Reduced for better generalization
EPOCHS = 100  # Increased number of epochs for better convergence
LR_SCHEDULER_FACTOR = 0.5  # Changed from 0.1 for smoother LR reduction
LR_SCHEDULER_PATIENCE = 7  # Increased patience for more stable training
GRADIENT_CLIP_VALUE = 1.0  # Value for gradient clipping (to prevent exploding gradients)

# Loss function parameters
LOSS_TYPE = "robust_translation_only"
TRANSLATION_LOSS_WEIGHT = 1.0
ROTATION_LOSS_WEIGHT = 0.0  # Set to 0 for translation-only mode
QUATERNION_NORM_WEIGHT = 0.0  # Set to 0 for translation-only mode
# Robust loss parameters
HUBER_WEIGHT = 0.4
WEIGHTED_WEIGHT = 0.2
SCALE_INV_WEIGHT = 0.1
SCALE_NORM_WEIGHT = 0.2  # Added for the new scale normalization component
GEOMETRIC_WEIGHT = 0.1   # Added for the new geometric consistency component

# Checkpoint settings
SAVE_CHECKPOINT_FREQ = 5  # Save checkpoint every N epochs
BEST_MODEL_PATH = os.path.join(RESULTS_DIR, "checkpoints", "best_model.pth")
LAST_MODEL_PATH = os.path.join(RESULTS_DIR, "checkpoints", "last_model.pth")

# Testing settings
TRAJECTORY_OUTPUT_PATH = os.path.join(RESULTS_DIR, "predictions", "trajectory.csv")
METRICS_OUTPUT_PATH = os.path.join(RESULTS_DIR, "predictions", "metrics.json")

# Visualization settings
LOSS_PLOT_PATH = os.path.join(RESULTS_DIR, "visualizations", "loss_curves.png")
TRAJECTORY_PLOT_PATH = os.path.join(RESULTS_DIR, "visualizations", "trajectory_comparison.png")
TRAJECTORY_3D_PLOT_PATH = os.path.join(RESULTS_DIR, "visualizations", "trajectory_3d.png")
ERROR_PLOT_PATH = os.path.join(RESULTS_DIR, "visualizations", "error_analysis.png")

# Device configuration (CUDA or CPU)
DEVICE = "cuda"  # Change to "cpu" if you don't have a GPU 