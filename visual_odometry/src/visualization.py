#!/usr/bin/env python3
"""
Visualization utilities for the Visual Odometry system.

This module provides functions for visualizing:
- Training and validation loss curves
- Predicted and ground truth trajectories (2D and 3D)
- Error analysis and distributions
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import config
from utils import (
    load_trajectory,
    plot_trajectory_2d,
    plot_trajectory_3d,
    plot_error_analysis,
    plot_loss_curves
)


def visualize_losses(logs_path=None, output_path=None):
    """
    Visualize training and validation losses from log file.
    
    Args:
        logs_path (str, optional): Path to the logs file. Defaults to None.
        output_path (str, optional): Path to save the plot. Defaults to None.
    """
    if logs_path is None:
        logs_path = os.path.join(config.RESULTS_DIR, "logs", "losses.json")
    
    if output_path is None:
        output_path = config.LOSS_PLOT_PATH
    
    if not os.path.exists(logs_path):
        print(f"Logs file not found at {logs_path}")
        return
    
    # Load logs
    with open(logs_path, 'r') as f:
        logs = json.load(f)
    
    train_losses = logs.get('train_losses', [])
    val_losses = logs.get('val_losses', [])
    
    if not train_losses or not val_losses:
        print("No loss data found in logs")
        return
    
    # Plot losses
    plot_loss_curves(train_losses, val_losses, output_path)
    print(f"Loss curves visualized and saved to {output_path}")
    
    # Plot detailed loss components if available
    if 'train_metrics' in logs and 'val_metrics' in logs:
        train_metrics = logs['train_metrics']
        val_metrics = logs['val_metrics']
        
        # Extract translation and rotation losses
        train_trans_losses = [metric.get('translation_loss', 0) for metric in train_metrics]
        val_trans_losses = [metric.get('translation_loss', 0) for metric in val_metrics]
        train_rot_losses = [metric.get('rotation_loss', 0) for metric in train_metrics]
        val_rot_losses = [metric.get('rotation_loss', 0) for metric in val_metrics]
        
        # Plot translation losses
        plt.figure(figsize=(10, 6))
        plt.plot(train_trans_losses, label='Train')
        plt.plot(val_trans_losses, label='Validation')
        plt.title('Translation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        # Save plot
        trans_output_path = os.path.join(os.path.dirname(output_path), "translation_loss.png")
        plt.savefig(trans_output_path)
        plt.close()
        
        # Plot rotation losses
        plt.figure(figsize=(10, 6))
        plt.plot(train_rot_losses, label='Train')
        plt.plot(val_rot_losses, label='Validation')
        plt.title('Rotation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        # Save plot
        rot_output_path = os.path.join(os.path.dirname(output_path), "rotation_loss.png")
        plt.savefig(rot_output_path)
        plt.close()
        
        print(f"Component loss curves saved to {os.path.dirname(output_path)}")


def visualize_trajectory_comparison(pred_trajectory_path=None, gt_trajectory_path=None):
    """
    Visualize predicted and ground truth trajectories.
    
    Args:
        pred_trajectory_path (str, optional): Path to predicted trajectory file. Defaults to None.
        gt_trajectory_path (str, optional): Path to ground truth trajectory file. Defaults to None.
    """
    if pred_trajectory_path is None:
        pred_trajectory_path = os.path.join(config.RESULTS_DIR, "predictions", "predicted_trajectory.csv")
    
    if gt_trajectory_path is None:
        gt_trajectory_path = os.path.join(config.RESULTS_DIR, "predictions", "ground_truth_trajectory.csv")
    
    if not os.path.exists(pred_trajectory_path) or not os.path.exists(gt_trajectory_path):
        print("Trajectory files not found")
        return
    
    # Load trajectories
    pred_trajectory = load_trajectory(pred_trajectory_path)
    gt_trajectory = load_trajectory(gt_trajectory_path)
    
    # Check if trajectories have the same length
    if len(pred_trajectory) != len(gt_trajectory):
        print(f"Warning: Trajectory lengths differ (pred: {len(pred_trajectory)}, gt: {len(gt_trajectory)})")
        # Use the minimum length
        min_len = min(len(pred_trajectory), len(gt_trajectory))
        pred_trajectory = pred_trajectory[:min_len]
        gt_trajectory = gt_trajectory[:min_len]
    
    # Visualize 2D trajectory
    plot_trajectory_2d(gt_trajectory, pred_trajectory, config.TRAJECTORY_PLOT_PATH)
    print(f"2D trajectory comparison saved to {config.TRAJECTORY_PLOT_PATH}")
    
    # Visualize 3D trajectory
    plot_trajectory_3d(gt_trajectory, pred_trajectory, config.TRAJECTORY_3D_PLOT_PATH)
    print(f"3D trajectory comparison saved to {config.TRAJECTORY_3D_PLOT_PATH}")


def visualize_error_analysis(pred_trajectory_path=None, gt_trajectory_path=None):
    """
    Visualize error analysis between predicted and ground truth trajectories.
    
    Args:
        pred_trajectory_path (str, optional): Path to predicted trajectory file. Defaults to None.
        gt_trajectory_path (str, optional): Path to ground truth trajectory file. Defaults to None.
    """
    if pred_trajectory_path is None:
        pred_trajectory_path = os.path.join(config.RESULTS_DIR, "predictions", "predicted_trajectory.csv")
    
    if gt_trajectory_path is None:
        gt_trajectory_path = os.path.join(config.RESULTS_DIR, "predictions", "ground_truth_trajectory.csv")
    
    if not os.path.exists(pred_trajectory_path) or not os.path.exists(gt_trajectory_path):
        print("Trajectory files not found")
        return
    
    # Load trajectories
    pred_trajectory = load_trajectory(pred_trajectory_path)
    gt_trajectory = load_trajectory(gt_trajectory_path)
    
    # Check if trajectories have the same length
    if len(pred_trajectory) != len(gt_trajectory):
        print(f"Warning: Trajectory lengths differ (pred: {len(pred_trajectory)}, gt: {len(gt_trajectory)})")
        # Use the minimum length
        min_len = min(len(pred_trajectory), len(gt_trajectory))
        pred_trajectory = pred_trajectory[:min_len]
        gt_trajectory = gt_trajectory[:min_len]
    
    # Visualize error analysis
    plot_error_analysis(gt_trajectory, pred_trajectory, config.ERROR_PLOT_PATH)
    print(f"Error analysis saved to {config.ERROR_PLOT_PATH}")


def visualize_trajectory_segments(pred_trajectory_path=None, gt_trajectory_path=None, segment_size=100):
    """
    Visualize trajectory segments for detailed comparison.
    
    Args:
        pred_trajectory_path (str, optional): Path to predicted trajectory file. Defaults to None.
        gt_trajectory_path (str, optional): Path to ground truth trajectory file. Defaults to None.
        segment_size (int, optional): Number of poses in each segment. Defaults to 100.
    """
    if pred_trajectory_path is None:
        pred_trajectory_path = os.path.join(config.RESULTS_DIR, "predictions", "predicted_trajectory.csv")
    
    if gt_trajectory_path is None:
        gt_trajectory_path = os.path.join(config.RESULTS_DIR, "predictions", "ground_truth_trajectory.csv")
    
    if not os.path.exists(pred_trajectory_path) or not os.path.exists(gt_trajectory_path):
        print("Trajectory files not found")
        return
    
    # Load trajectories
    pred_trajectory = load_trajectory(pred_trajectory_path)
    gt_trajectory = load_trajectory(gt_trajectory_path)
    
    # Check if trajectories have the same length
    if len(pred_trajectory) != len(gt_trajectory):
        print(f"Warning: Trajectory lengths differ (pred: {len(pred_trajectory)}, gt: {len(gt_trajectory)})")
        # Use the minimum length
        min_len = min(len(pred_trajectory), len(gt_trajectory))
        pred_trajectory = pred_trajectory[:min_len]
        gt_trajectory = gt_trajectory[:min_len]
    
    # Create segments directory
    segments_dir = os.path.join(config.RESULTS_DIR, "visualizations", "segments")
    os.makedirs(segments_dir, exist_ok=True)
    
    # Get number of segments
    num_poses = len(pred_trajectory)
    num_segments = (num_poses + segment_size - 1) // segment_size  # Ceiling division
    
    for i in range(num_segments):
        # Get segment range
        start_idx = i * segment_size
        end_idx = min((i + 1) * segment_size, num_poses)
        
        # Extract segment
        pred_segment = pred_trajectory[start_idx:end_idx]
        gt_segment = gt_trajectory[start_idx:end_idx]
        
        # Create output paths
        segment_2d_path = os.path.join(segments_dir, f"segment_{i+1}_2d.png")
        segment_3d_path = os.path.join(segments_dir, f"segment_{i+1}_3d.png")
        
        # Visualize 2D trajectory
        plot_trajectory_2d(gt_segment, pred_segment, segment_2d_path)
        
        # Visualize 3D trajectory
        plot_trajectory_3d(gt_segment, pred_segment, segment_3d_path)
    
    print(f"Trajectory segments visualized and saved to {segments_dir}")


def visualize_all():
    """Visualize all available data."""
    print("Visualizing all data...")
    
    # Check if directories exist
    if not os.path.exists(config.RESULTS_DIR):
        print(f"Results directory not found at {config.RESULTS_DIR}")
        return
    
    # Visualize losses
    visualize_losses()
    
    # Visualize trajectories
    visualize_trajectory_comparison()
    
    # Visualize error analysis
    visualize_error_analysis()
    
    # Visualize trajectory segments
    visualize_trajectory_segments()
    
    print("All visualizations complete!")


if __name__ == "__main__":
    visualize_all() 