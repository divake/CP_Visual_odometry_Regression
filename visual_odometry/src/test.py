#!/usr/bin/env python3
"""
Testing script for the Visual Odometry model.

This script evaluates the trained model on the test dataset:
- Loads a trained model
- Predicts relative poses on the test sequence
- Computes the full trajectory by chaining relative pose predictions
- Calculates and reports error metrics
- Saves predictions and visualizations
"""

import os
import json
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

import config
from dataloader import create_data_loaders
from model import get_model
from utils import (
    compute_trajectory,
    calculate_ate,
    save_trajectory,
    save_metrics,
    plot_trajectory_2d,
    plot_trajectory_3d,
    plot_error_analysis,
    calculate_scale_error,
    align_trajectories
)


def load_model(model_path):
    """
    Load a trained model from a checkpoint file.
    
    Args:
        model_path (str): Path to the model checkpoint
        
    Returns:
        torch.nn.Module: Loaded model
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Determine which model type to create
    # Try to extract the model type from the checkpoint first
    model_type = None
    if 'model_type' in checkpoint:
        model_type = checkpoint['model_type']
    else:
        # If it's not in the checkpoint, try to extract it from state_dict keys
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Check key patterns to determine model type
        if 'conv_after_corr.0.weight' in state_dict:
            model_type = "enhanced"
        elif 'cnn.0.weight' in state_dict:
            model_type = "siamese"
        else:
            model_type = "standard"  # Default
            
    print(f"Detected model type: {model_type}")
    
    # Create model of appropriate type
    model = get_model(model_type=model_type, pretrained=False)
    
    # If architecture mismatch, we need to handle it
    try:
        # Attempt to load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Model weights loaded successfully")
    except RuntimeError as e:
        print(f"Error loading model weights: {e}")
        print("Attempting to create a compatible model...")
        # If there's a mismatch, use the current config model type
        model = get_model(model_type=config.MODEL_TYPE, pretrained=True)
        print(f"Created new {config.MODEL_TYPE} model with pretrained weights.")
    
    return model


def predict_poses(model, test_loader, device):
    """
    Predict relative poses on the test dataset.
    
    In the translation-only mode, we use the ground truth rotations
    with our predicted translations to form the complete poses.
    
    Args:
        model (torch.nn.Module): The model to evaluate
        test_loader (torch.utils.data.DataLoader): DataLoader for test data
        device (str): Device to use for inference
        
    Returns:
        tuple: (pred_rel_poses, gt_rel_poses)
            - pred_rel_poses (numpy.ndarray): Predicted relative poses, shape (N, 7)
            - gt_rel_poses (numpy.ndarray): Ground truth relative poses, shape (N, 7)
    """
    model.eval()
    
    pred_rel_poses = []
    gt_rel_poses = []
    
    with torch.no_grad():
        for img1, img2, rel_pose in tqdm(test_loader, desc="Predicting poses"):
            # Move data to device
            img1 = img1.to(device)
            img2 = img2.to(device)
            
            # Forward pass
            _, pred_translation = model(img1, img2)
            
            # Convert to numpy
            pred_translation = pred_translation.cpu().numpy()
            rel_pose = rel_pose.cpu().numpy()
            
            # Store predictions and ground truth
            for i in range(len(pred_translation)):
                # Extract ground truth rotation and translation
                gt_rotation = rel_pose[i, :4]  # w, x, y, z
                gt_translation = rel_pose[i, 4:7]  # x, y, z
                
                # Use predicted translation with ground truth rotation
                # This is the core change for the translation-only approach
                pred_rel_pose = np.concatenate([gt_rotation, pred_translation[i]])
                gt_rel_pose = rel_pose[i]
                
                pred_rel_poses.append(pred_rel_pose)
                gt_rel_poses.append(gt_rel_pose)
    
    # Ensure the poses are in the original order for a consistent trajectory
    return np.array(pred_rel_poses), np.array(gt_rel_poses)


def get_initial_pose(test_scene_id):
    """
    Get the initial pose for the test scene.
    
    Args:
        test_scene_id (str): ID of the test scene
        
    Returns:
        numpy.ndarray: Initial pose as [quaternion(w,x,y,z), translation(x,y,z)]
    """
    pose_file = os.path.join(config.DATASET_PATH, "pc", f"{test_scene_id}.pose")
    
    with open(pose_file, 'r') as f:
        line = f.readline().strip()
        values = [float(v) for v in line.split()]
        
        if len(values) == 7:
            return np.array(values)
    
    # Default to identity if pose file not found or invalid
    return np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on the test dataset and compute trajectory metrics.
    
    Args:
        model (torch.nn.Module): The model to evaluate
        test_loader (torch.utils.data.DataLoader): DataLoader for test data
        device (str): Device to use for inference
        
    Returns:
        dict: Evaluation metrics
    """
    # Predict relative poses
    relative_poses, gt_relative_poses = predict_poses(model, test_loader, device)
    
    # Compute full trajectory
    gt_trajectory = compute_trajectory(gt_relative_poses)
    pred_trajectory = compute_trajectory(relative_poses)
    
    # Calculate metrics
    # Calculate metrics with three different alignment approaches
    translations_rmse = np.sqrt(np.mean(np.sum((relative_poses[:, 4:7] - gt_relative_poses[:, 4:7]) ** 2, axis=1)))
    
    # Regular ATE without alignment
    ate_no_align = calculate_ate(pred_trajectory, gt_trajectory, align_scale=False)
    
    # ATE with scale alignment
    ate_scale_align = calculate_ate(pred_trajectory, gt_trajectory, align_scale=True)
    
    # Calculate scale error
    mean_scale_ratio, scale_consistency = calculate_scale_error(pred_trajectory, gt_trajectory)
    
    # Compute aligned trajectory for visualization
    pred_trajectory_aligned = align_trajectories(pred_trajectory, gt_trajectory, align_type='scale')
    
    # Calculate per-axis errors in the aligned trajectory
    axis_errors = np.abs(pred_trajectory_aligned - gt_trajectory)
    axis_rmse = np.sqrt(np.mean(axis_errors ** 2, axis=0))
    
    metrics = {
        "translation_error_rmse": float(translations_rmse),
        "ate_no_alignment": float(ate_no_align),
        "ate_scale_aligned": float(ate_scale_align),
        "mean_scale_ratio": float(mean_scale_ratio),
        "scale_consistency": float(scale_consistency),
        "x_error_rmse": float(axis_rmse[0]),
        "y_error_rmse": float(axis_rmse[1]),
        "z_error_rmse": float(axis_rmse[2]),
    }
    
    # Save trajectories for visualization
    trajectories = {
        "gt_trajectory": gt_trajectory,
        "pred_trajectory": pred_trajectory,
        "pred_trajectory_aligned": pred_trajectory_aligned
    }
    
    return metrics, trajectories


def main():
    """
    Main testing function.
    """
    print("=== Testing Visual Odometry Model ===")
    
    # Set device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data loaders
    _, _, test_loader = create_data_loaders()
    print(f"Test dataset size: {len(test_loader.dataset)} frame pairs")
    
    # Load model
    model_path = config.BEST_MODEL_PATH
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    model.to(device)
    model.eval()
    
    # Evaluate model
    print("Evaluating model...")
    metrics, trajectories = evaluate_model(model, test_loader, device)
    
    # Print metrics
    print("\n=== Evaluation Metrics ===")
    print(f"Translation Error (RMSE): {metrics['translation_error_rmse']:.4f} m")
    print(f"ATE without alignment: {metrics['ate_no_alignment']:.4f} m")
    print(f"ATE with scale alignment: {metrics['ate_scale_aligned']:.4f} m")
    print(f"Mean Scale Ratio: {metrics['mean_scale_ratio']:.4f} (1.0 is ideal)")
    print(f"Scale Consistency (std): {metrics['scale_consistency']:.4f} (lower is better)")
    print(f"X-axis Error (RMSE): {metrics['x_error_rmse']:.4f} m")
    print(f"Y-axis Error (RMSE): {metrics['y_error_rmse']:.4f} m")
    print(f"Z-axis Error (RMSE): {metrics['z_error_rmse']:.4f} m")
    
    # Save metrics
    save_metrics(metrics, config.METRICS_OUTPUT_PATH)
    print(f"Metrics saved to {config.METRICS_OUTPUT_PATH}")
    
    # Save trajectories
    save_trajectory(trajectories["gt_trajectory"], "gt_trajectory.csv")
    save_trajectory(trajectories["pred_trajectory"], "pred_trajectory.csv")
    save_trajectory(trajectories["pred_trajectory_aligned"], "pred_trajectory_aligned.csv")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    # 2D plot (top-down view)
    plot_trajectory_2d(
        trajectories["gt_trajectory"], 
        trajectories["pred_trajectory"],
        trajectories["pred_trajectory_aligned"],
        save_path=os.path.join(config.RESULTS_DIR, "visualizations", "trajectory_2d.png")
    )
    
    # 3D plot
    plot_trajectory_3d(
        trajectories["gt_trajectory"], 
        trajectories["pred_trajectory_aligned"],
        save_path=os.path.join(config.RESULTS_DIR, "visualizations", "trajectory_3d.png")
    )
    
    # Error analysis plot
    plot_error_analysis(
        trajectories["gt_trajectory"],
        trajectories["pred_trajectory_aligned"],
        save_path=os.path.join(config.RESULTS_DIR, "visualizations", "translation_errors.png")
    )
    
    print("\nTesting completed.")


if __name__ == "__main__":
    main() 