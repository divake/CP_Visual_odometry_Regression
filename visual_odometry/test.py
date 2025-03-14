#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Testing script for the visual odometry system.

This script handles:
- Loading a trained model
- Evaluating on the test set
- Calculating all metrics
- Generating visualizations
- Exporting results to a report
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import logging
import argparse
import json
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from visual_odometry.config import (
    MODEL_CONFIG, EVAL_CONFIG, DATASET_CONFIG, DEVICE, create_directories
)
from visual_odometry.models.base_model import create_model
from visual_odometry.models.loss import create_loss_function
from visual_odometry.dataset import (
    DatasetConfig, RGBDScenesDataset, create_dataloaders
)
from visual_odometry.utils.evaluation import evaluate_trajectory, print_evaluation_results
from visual_odometry.utils.visualization import visualize_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path: str, device: torch.device) -> nn.Module:
    """
    Load a trained model from a checkpoint.
    
    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    # Create model with the same architecture
    model = create_model(MODEL_CONFIG)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move model to device
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    logger.info(f"Loaded model from {model_path}")
    logger.info(f"Model was trained for {checkpoint['epoch']} epochs")
    logger.info(f"Best validation loss: {checkpoint['best_loss']:.6f}")
    
    return model


def test_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> Tuple[Dict[str, Dict[str, float]], np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Test the model on a dataset.
    
    Args:
        model: Model to test
        dataloader: DataLoader for test data
        device: Device to test on
        
    Returns:
        Tuple containing:
        - Dictionary with evaluation results
        - Array of predicted poses
        - Array of ground truth poses
        - List of RGB images (for visualization)
    """
    model.eval()
    
    # Collect all predictions and ground truth
    all_pred_poses = []
    all_gt_poses = []
    all_rgb_images = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            # Move data to device
            for key in batch:
                batch[key] = batch[key].to(device)
            
            # Forward pass
            pred_pose = model(batch)
            
            # Collect poses
            all_pred_poses.append(pred_pose.cpu().numpy())
            all_gt_poses.append(batch['rel_pose'].cpu().numpy())
            
            # Collect RGB images for visualization
            all_rgb_images.append(batch['rgb1'].cpu().numpy())
    
    # Concatenate all poses
    all_pred_poses = np.concatenate(all_pred_poses, axis=0)
    all_gt_poses = np.concatenate(all_gt_poses, axis=0)
    
    # Convert RGB images to uint8 format
    rgb_images = []
    for batch_images in all_rgb_images:
        for i in range(batch_images.shape[0]):
            # Convert from tensor format (C, H, W) to image format (H, W, C)
            img = batch_images[i].transpose(1, 2, 0)
            # Scale from [0, 1] to [0, 255]
            img = (img * 255).astype(np.uint8)
            rgb_images.append(img)
    
    # Evaluate trajectory
    results = evaluate_trajectory(
        all_pred_poses,
        all_gt_poses,
        metrics=EVAL_CONFIG["metrics"]
    )
    
    return results, all_pred_poses, all_gt_poses, rgb_images


def reconstruct_trajectory(
    relative_poses: np.ndarray,
    initial_pose: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Reconstruct a trajectory from relative poses.
    
    Args:
        relative_poses: Array of relative poses of shape (N, 7) [qw, qx, qy, qz, tx, ty, tz]
        initial_pose: Optional initial pose of shape (7,) [qw, qx, qy, qz, tx, ty, tz]
            If not provided, identity pose is used
            
    Returns:
        Array of absolute poses of shape (N+1, 7) [qw, qx, qy, qz, tx, ty, tz]
    """
    n_poses = len(relative_poses)
    
    # Initialize trajectory with identity pose if not provided
    if initial_pose is None:
        initial_pose = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Initialize trajectory array
    trajectory = np.zeros((n_poses + 1, 7))
    trajectory[0] = initial_pose
    
    # Initialize transformation matrix for the first pose
    T_abs = np.eye(4)
    if initial_pose is not None:
        # Convert quaternion to rotation matrix
        qw, qx, qy, qz = initial_pose[:4]
        r = Rotation.from_quat([qx, qy, qz, qw])  # scipy uses [x,y,z,w]
        T_abs[:3, :3] = r.as_matrix()
        T_abs[:3, 3] = initial_pose[4:]
    
    # Reconstruct trajectory
    for i in range(n_poses):
        # Get relative pose
        rel_pose = relative_poses[i]
        
        # Extract quaternion and translation
        qw, qx, qy, qz = rel_pose[:4]
        tx, ty, tz = rel_pose[4:]
        
        # Create transformation matrix for relative pose
        T_rel = np.eye(4)
        r_rel = Rotation.from_quat([qx, qy, qz, qw])  # scipy uses [x,y,z,w]
        T_rel[:3, :3] = r_rel.as_matrix()
        T_rel[:3, 3] = [tx, ty, tz]
        
        # Update absolute pose: T_abs_new = T_abs * T_rel
        T_abs = T_abs @ T_rel
        
        # Extract new pose
        r_abs = Rotation.from_matrix(T_abs[:3, :3])
        quat_abs = r_abs.as_quat()  # [x,y,z,w]
        quat_abs = np.array([quat_abs[3], quat_abs[0], quat_abs[1], quat_abs[2]])  # [w,x,y,z]
        trans_abs = T_abs[:3, 3]
        
        # Store new pose
        trajectory[i+1, :4] = quat_abs
        trajectory[i+1, 4:] = trans_abs
    
    return trajectory


def save_results(
    results: Dict[str, Dict[str, float]],
    output_dir: str,
    filename: str = "evaluation_results.json"
) -> None:
    """
    Save evaluation results to a JSON file.
    
    Args:
        results: Dictionary with evaluation results
        output_dir: Directory to save the results
        filename: Name of the output file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert numpy values to Python types
    def convert_to_python_types(obj):
        if isinstance(obj, dict):
            return {k: convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        else:
            return obj
    
    # Convert results to Python types
    results_py = convert_to_python_types(results)
    
    # Save to JSON file
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w') as f:
        json.dump(results_py, f, indent=4)
    
    logger.info(f"Saved evaluation results to {output_path}")


def main(args):
    """Main function for testing."""
    # Create directories
    create_directories()
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset configuration
    dataset_config = DatasetConfig(
        img_size=DATASET_CONFIG["img_size"],
        max_depth=DATASET_CONFIG["max_depth"]
    )
    
    # Create dataloaders
    dataloaders = create_dataloaders(
        dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Get test dataloader
    test_loader = dataloaders["test"]
    
    # Load model
    model = load_model(args.model_path, DEVICE)
    
    # Test model
    logger.info("Testing model...")
    results, pred_poses, gt_poses, rgb_images = test_model(model, test_loader, DEVICE)
    
    # Print evaluation results
    print_evaluation_results(results)
    
    # Save results
    save_results(results, output_dir, EVAL_CONFIG["report_file"])
    
    # Visualize results
    if EVAL_CONFIG["visualization"]["save_visualizations"]:
        logger.info("Generating visualizations...")
        visualization_dir = os.path.join(output_dir, EVAL_CONFIG["visualization"]["visualization_dir"])
        os.makedirs(visualization_dir, exist_ok=True)
        
        # Use a subset of images for visualization to save memory
        subset_size = min(100, len(rgb_images))
        subset_indices = np.linspace(0, len(rgb_images) - 1, subset_size, dtype=int)
        
        rgb_subset = [rgb_images[i] for i in subset_indices]
        pred_subset = pred_poses[subset_indices]
        gt_subset = gt_poses[subset_indices]
        
        # Reconstruct trajectories
        # This is a simplified reconstruction and might need refinement
        pred_trajectory = reconstruct_trajectory(pred_subset)
        gt_trajectory = reconstruct_trajectory(gt_subset)
        
        # Visualize results
        visualize_results(
            pred_trajectory,
            gt_trajectory,
            results,
            visualization_dir,
            rgb_subset if args.include_images else None
        )
    
    logger.info(f"Testing completed. Results saved to {output_dir}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test visual odometry model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--output_dir", type=str, default="test_results", help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for testing")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--include_images", action="store_true", help="Include images in visualizations")
    args = parser.parse_args()
    
    # Test the model
    main(args) 