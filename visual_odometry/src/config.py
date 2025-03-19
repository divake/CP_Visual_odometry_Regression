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
TRAIN_SCENES = ["01", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14"]
TEST_SCENES = ["02"]  # We'll test on scene_02
TRAIN_VAL_SPLIT = 0.8  # 80% for training, 20% for validation

# Image preprocessing
IMG_HEIGHT = 240
IMG_WIDTH = 320
NORMALIZE_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
NORMALIZE_STD = [0.229, 0.224, 0.225]   # ImageNet std

# Model parameters
RESNET_PRETRAINED = True
FEATURE_DIMENSION = 512  # Output dimension of ResNet18 features
FC_DROPOUT = 0.5        # Dropout rate for fully connected layers

# Translation and rotation output dimensions
TRANSLATION_DIM = 3     # x, y, z
ROTATION_DIM = 4        # quaternion: w, x, y, z

# Training hyperparameters
BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
NUM_EPOCHS = 50
LR_SCHEDULER_STEP_SIZE = 20
LR_SCHEDULER_GAMMA = 0.1

# Loss function weights
TRANSLATION_LOSS_WEIGHT = 1.0
ROTATION_LOSS_WEIGHT = 1.0
QUATERNION_NORM_WEIGHT = 0.1  # Weight for quaternion normalization loss

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