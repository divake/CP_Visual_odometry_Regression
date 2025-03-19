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
    plot_error_analysis
)


def load_model(model_path):
    """
    Load a trained model from a checkpoint file.
    
    Args:
        model_path (str): Path to the model checkpoint
        
    Returns:
        torch.nn.Module: Loaded model
    """
    # Create model
    model = get_model(model_type="standard", pretrained=False)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Load model state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model


def predict_poses(model, test_loader, device):
    """
    Predict relative poses on the test dataset.
    
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
            pred_rotation, pred_translation = model(img1, img2)
            
            # Convert to numpy
            pred_rotation = pred_rotation.cpu().numpy()
            pred_translation = pred_translation.cpu().numpy()
            rel_pose = rel_pose.cpu().numpy()
            
            # Store predictions and ground truth
            for i in range(len(pred_rotation)):
                # Combine rotation and translation into a single pose
                pred_rel_pose = np.concatenate([pred_rotation[i], pred_translation[i]])
                gt_rel_pose = rel_pose[i]
                
                pred_rel_poses.append(pred_rel_pose)
                gt_rel_poses.append(gt_rel_pose)
    
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


def evaluate_model(model_path):
    """
    Evaluate a trained model on the test dataset.
    
    Args:
        model_path (str): Path to the trained model
        
    Returns:
        dict: Evaluation metrics
    """
    # Set device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directories if they don't exist
    os.makedirs(os.path.join(config.RESULTS_DIR, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(config.RESULTS_DIR, "visualizations"), exist_ok=True)
    
    # Load model
    model = load_model(model_path)
    model = model.to(device)
    print("Model loaded successfully")
    
    # Create data loaders
    _, _, test_loader = create_data_loaders()
    print(f"Test batches: {len(test_loader)}")
    
    # Predict poses
    pred_rel_poses, gt_rel_poses = predict_poses(model, test_loader, device)
    print(f"Predicted {len(pred_rel_poses)} relative poses")
    
    # Get initial pose
    initial_pose = get_initial_pose(config.TEST_SCENES[0])
    print(f"Initial pose: {initial_pose}")
    
    # Compute full trajectories
    pred_trajectory = compute_trajectory(initial_pose, pred_rel_poses)
    gt_trajectory = compute_trajectory(initial_pose, gt_rel_poses)
    print(f"Computed trajectories with {len(pred_trajectory)} poses")
    
    # Calculate ATE
    metrics = calculate_ate(gt_trajectory, pred_trajectory)
    print("Absolute Trajectory Error (ATE) metrics:")
    print(f"  Translation error (mean): {metrics['translation_error_mean']:.4f} m")
    print(f"  Translation error (median): {metrics['translation_error_median']:.4f} m")
    print(f"  Translation error (RMSE): {metrics['translation_error_rmse']:.4f} m")
    print(f"  Rotation error (mean): {metrics['rotation_error_mean']:.4f} degrees")
    print(f"  Rotation error (median): {metrics['rotation_error_median']:.4f} degrees")
    print(f"  Rotation error (RMSE): {metrics['rotation_error_rmse']:.4f} degrees")
    
    # Save metrics
    save_metrics(metrics, config.METRICS_OUTPUT_PATH)
    print(f"Metrics saved to {config.METRICS_OUTPUT_PATH}")
    
    # Save trajectories
    save_trajectory(pred_trajectory, os.path.join(config.RESULTS_DIR, "predictions", "predicted_trajectory.csv"))
    save_trajectory(gt_trajectory, os.path.join(config.RESULTS_DIR, "predictions", "ground_truth_trajectory.csv"))
    print("Trajectories saved")
    
    # Generate visualizations
    plot_trajectory_2d(gt_trajectory, pred_trajectory, config.TRAJECTORY_PLOT_PATH)
    plot_trajectory_3d(gt_trajectory, pred_trajectory, config.TRAJECTORY_3D_PLOT_PATH)
    plot_error_analysis(gt_trajectory, pred_trajectory, config.ERROR_PLOT_PATH)
    print("Visualizations saved")
    
    return metrics


def main():
    """Main function."""
    # Find the best model checkpoint
    best_model_path = config.BEST_MODEL_PATH
    
    if not os.path.exists(best_model_path):
        print(f"Best model not found at {best_model_path}. Looking for other checkpoints...")
        
        # Look for the latest checkpoint
        checkpoint_dir = os.path.join(config.RESULTS_DIR, "checkpoints")
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_epoch_")]
        
        if checkpoints:
            # Sort by epoch number
            checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]), reverse=True)
            best_model_path = os.path.join(checkpoint_dir, checkpoints[0])
            print(f"Using checkpoint: {best_model_path}")
        else:
            # No checkpoints found
            print("No checkpoints found. Please train the model first.")
            return
    
    # Evaluate the model
    metrics = evaluate_model(best_model_path)
    
    # Print summary
    print("\nEvaluation complete!")
    print(f"Translation RMSE: {metrics['translation_error_rmse']:.4f} m")
    print(f"Rotation RMSE: {metrics['rotation_error_rmse']:.4f} degrees")


if __name__ == "__main__":
    main() 