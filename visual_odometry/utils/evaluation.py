#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluation metrics for visual odometry.

This module implements various metrics for evaluating visual odometry performance:
- Absolute Trajectory Error (ATE)
- Relative Pose Error (RPE)
- Translation error (in meters)
- Rotation error (in degrees)
- Drift calculation over trajectory length
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
import math

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to Euler angles (roll, pitch, yaw).
    
    Args:
        q: Quaternion [qw, qx, qy, qz]
        
    Returns:
        Euler angles [roll, pitch, yaw] in radians
    """
    qw, qx, qy, qz = q
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw])


def quaternion_angular_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """
    Compute angular distance between two quaternions in degrees.
    
    Args:
        q1: First quaternion [qw, qx, qy, qz]
        q2: Second quaternion [qw, qx, qy, qz]
        
    Returns:
        Angular distance in degrees
    """
    # Normalize quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # Compute dot product
    dot_product = np.clip(np.abs(np.sum(q1 * q2)), -1.0, 1.0)
    
    # Convert to angle in degrees
    angle = 2 * np.arccos(dot_product) * 180.0 / np.pi
    
    return angle


def compute_absolute_trajectory_error(
    pred_poses: np.ndarray, 
    gt_poses: np.ndarray
) -> Dict[str, float]:
    """
    Compute Absolute Trajectory Error (ATE).
    
    ATE measures the global consistency of the trajectory by comparing the
    absolute distances between the estimated and ground truth trajectories.
    
    Args:
        pred_poses: Predicted poses of shape (N, 7) [qw, qx, qy, qz, tx, ty, tz]
        gt_poses: Ground truth poses of shape (N, 7) [qw, qx, qy, qz, tx, ty, tz]
        
    Returns:
        Dictionary with ATE metrics:
        - 'mean': Mean ATE
        - 'median': Median ATE
        - 'std': Standard deviation of ATE
        - 'min': Minimum ATE
        - 'max': Maximum ATE
        - 'rmse': Root Mean Square Error
    """
    if len(pred_poses) != len(gt_poses):
        raise ValueError(f"Number of predicted poses ({len(pred_poses)}) does not match ground truth ({len(gt_poses)})")
    
    # Extract translation components
    pred_translations = pred_poses[:, 4:]
    gt_translations = gt_poses[:, 4:]
    
    # Compute Euclidean distances
    errors = np.linalg.norm(pred_translations - gt_translations, axis=1)
    
    # Compute statistics
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    std_error = np.std(errors)
    min_error = np.min(errors)
    max_error = np.max(errors)
    rmse = np.sqrt(np.mean(np.square(errors)))
    
    return {
        'mean': mean_error,
        'median': median_error,
        'std': std_error,
        'min': min_error,
        'max': max_error,
        'rmse': rmse
    }


def compute_relative_pose_error(
    pred_poses: np.ndarray, 
    gt_poses: np.ndarray,
    delta: int = 1
) -> Dict[str, Dict[str, float]]:
    """
    Compute Relative Pose Error (RPE).
    
    RPE measures the local accuracy of the trajectory over a fixed time interval delta.
    It computes the relative error between pairs of poses separated by delta steps.
    
    Args:
        pred_poses: Predicted poses of shape (N, 7) [qw, qx, qy, qz, tx, ty, tz]
        gt_poses: Ground truth poses of shape (N, 7) [qw, qx, qy, qz, tx, ty, tz]
        delta: Time interval for relative error computation
        
    Returns:
        Dictionary with RPE metrics for translation and rotation:
        - 'translation': Translation RPE statistics
        - 'rotation': Rotation RPE statistics
    """
    if len(pred_poses) != len(gt_poses):
        raise ValueError(f"Number of predicted poses ({len(pred_poses)}) does not match ground truth ({len(gt_poses)})")
    
    if delta >= len(pred_poses):
        raise ValueError(f"Delta ({delta}) must be less than the number of poses ({len(pred_poses)})")
    
    # Number of pose pairs
    n_pairs = len(pred_poses) - delta
    
    # Initialize error arrays
    translation_errors = np.zeros(n_pairs)
    rotation_errors = np.zeros(n_pairs)
    
    for i in range(n_pairs):
        # Extract poses at time i and i+delta
        pred_pose_i = pred_poses[i]
        pred_pose_i_delta = pred_poses[i + delta]
        
        gt_pose_i = gt_poses[i]
        gt_pose_i_delta = gt_poses[i + delta]
        
        # Compute relative poses
        # For simplicity, we'll just compute the difference in translation
        # and the angular distance between quaternions
        
        # Translation error
        pred_translation_diff = pred_pose_i_delta[4:] - pred_pose_i[4:]
        gt_translation_diff = gt_pose_i_delta[4:] - gt_pose_i[4:]
        
        translation_errors[i] = np.linalg.norm(pred_translation_diff - gt_translation_diff)
        
        # Rotation error
        pred_rotation_i = pred_pose_i[:4]
        pred_rotation_i_delta = pred_pose_i_delta[:4]
        
        gt_rotation_i = gt_pose_i[:4]
        gt_rotation_i_delta = gt_pose_i_delta[:4]
        
        # Compute relative rotations (simplified)
        # In practice, you would use quaternion multiplication and inverse
        # Here we just compute the angular distance between the quaternions
        pred_rotation_diff = quaternion_angular_distance(pred_rotation_i, pred_rotation_i_delta)
        gt_rotation_diff = quaternion_angular_distance(gt_rotation_i, gt_rotation_i_delta)
        
        rotation_errors[i] = abs(pred_rotation_diff - gt_rotation_diff)
    
    # Compute statistics for translation errors
    translation_stats = {
        'mean': np.mean(translation_errors),
        'median': np.median(translation_errors),
        'std': np.std(translation_errors),
        'min': np.min(translation_errors),
        'max': np.max(translation_errors),
        'rmse': np.sqrt(np.mean(np.square(translation_errors)))
    }
    
    # Compute statistics for rotation errors
    rotation_stats = {
        'mean': np.mean(rotation_errors),
        'median': np.median(rotation_errors),
        'std': np.std(rotation_errors),
        'min': np.min(rotation_errors),
        'max': np.max(rotation_errors),
        'rmse': np.sqrt(np.mean(np.square(rotation_errors)))
    }
    
    return {
        'translation': translation_stats,
        'rotation': rotation_stats
    }


def compute_translation_error(
    pred_poses: np.ndarray, 
    gt_poses: np.ndarray
) -> Dict[str, float]:
    """
    Compute translation error in meters.
    
    Args:
        pred_poses: Predicted poses of shape (N, 7) [qw, qx, qy, qz, tx, ty, tz]
        gt_poses: Ground truth poses of shape (N, 7) [qw, qx, qy, qz, tx, ty, tz]
        
    Returns:
        Dictionary with translation error metrics
    """
    # Extract translation components
    pred_translations = pred_poses[:, 4:]
    gt_translations = gt_poses[:, 4:]
    
    # Compute Euclidean distances
    errors = np.linalg.norm(pred_translations - gt_translations, axis=1)
    
    # Compute statistics
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    std_error = np.std(errors)
    min_error = np.min(errors)
    max_error = np.max(errors)
    rmse = np.sqrt(np.mean(np.square(errors)))
    
    return {
        'mean': mean_error,
        'median': median_error,
        'std': std_error,
        'min': min_error,
        'max': max_error,
        'rmse': rmse
    }


def compute_rotation_error(
    pred_poses: np.ndarray, 
    gt_poses: np.ndarray
) -> Dict[str, float]:
    """
    Compute rotation error in degrees.
    
    Args:
        pred_poses: Predicted poses of shape (N, 7) [qw, qx, qy, qz, tx, ty, tz]
        gt_poses: Ground truth poses of shape (N, 7) [qw, qx, qy, qz, tx, ty, tz]
        
    Returns:
        Dictionary with rotation error metrics
    """
    n_poses = len(pred_poses)
    errors = np.zeros(n_poses)
    
    for i in range(n_poses):
        # Extract quaternions
        pred_q = pred_poses[i, :4]
        gt_q = gt_poses[i, :4]
        
        # Compute angular distance
        errors[i] = quaternion_angular_distance(pred_q, gt_q)
    
    # Compute statistics
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    std_error = np.std(errors)
    min_error = np.min(errors)
    max_error = np.max(errors)
    rmse = np.sqrt(np.mean(np.square(errors)))
    
    return {
        'mean': mean_error,
        'median': median_error,
        'std': std_error,
        'min': min_error,
        'max': max_error,
        'rmse': rmse
    }


def compute_drift(
    pred_poses: np.ndarray, 
    gt_poses: np.ndarray,
    trajectory_length: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute drift over trajectory length.
    
    Drift is measured as the error at the end of the trajectory divided by the
    total trajectory length.
    
    Args:
        pred_poses: Predicted poses of shape (N, 7) [qw, qx, qy, qz, tx, ty, tz]
        gt_poses: Ground truth poses of shape (N, 7) [qw, qx, qy, qz, tx, ty, tz]
        trajectory_length: Optional total trajectory length in meters
            If not provided, it will be computed from the ground truth poses
        
    Returns:
        Dictionary with drift metrics:
        - 'translation_drift': Translation drift in %
        - 'rotation_drift': Rotation drift in deg/m
        - 'final_translation_error': Final translation error in meters
        - 'final_rotation_error': Final rotation error in degrees
    """
    # Compute final pose error
    final_translation_error = np.linalg.norm(pred_poses[-1, 4:] - gt_poses[-1, 4:])
    final_rotation_error = quaternion_angular_distance(pred_poses[-1, :4], gt_poses[-1, :4])
    
    # Compute trajectory length if not provided
    if trajectory_length is None:
        trajectory_length = 0.0
        for i in range(1, len(gt_poses)):
            trajectory_length += np.linalg.norm(gt_poses[i, 4:] - gt_poses[i-1, 4:])
    
    # Compute drift
    translation_drift = (final_translation_error / trajectory_length) * 100.0  # in %
    rotation_drift = final_rotation_error / trajectory_length  # in deg/m
    
    return {
        'translation_drift': translation_drift,
        'rotation_drift': rotation_drift,
        'final_translation_error': final_translation_error,
        'final_rotation_error': final_rotation_error,
        'trajectory_length': trajectory_length
    }


def evaluate_trajectory(
    pred_poses: Union[np.ndarray, torch.Tensor], 
    gt_poses: Union[np.ndarray, torch.Tensor],
    metrics: List[str] = ['ate', 'rpe', 'translation_error', 'rotation_error', 'drift']
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate trajectory using multiple metrics.
    
    Args:
        pred_poses: Predicted poses of shape (N, 7) [qw, qx, qy, qz, tx, ty, tz]
        gt_poses: Ground truth poses of shape (N, 7) [qw, qx, qy, qz, tx, ty, tz]
        metrics: List of metrics to compute
        
    Returns:
        Dictionary with evaluation results for each metric
    """
    # Convert to numpy if tensors
    if isinstance(pred_poses, torch.Tensor):
        pred_poses = pred_poses.detach().cpu().numpy()
    
    if isinstance(gt_poses, torch.Tensor):
        gt_poses = gt_poses.detach().cpu().numpy()
    
    results = {}
    
    # Compute each requested metric
    for metric in metrics:
        if metric == 'ate':
            results['ate'] = compute_absolute_trajectory_error(pred_poses, gt_poses)
        elif metric == 'rpe':
            results['rpe'] = compute_relative_pose_error(pred_poses, gt_poses)
        elif metric == 'translation_error':
            results['translation_error'] = compute_translation_error(pred_poses, gt_poses)
        elif metric == 'rotation_error':
            results['rotation_error'] = compute_rotation_error(pred_poses, gt_poses)
        elif metric == 'drift':
            results['drift'] = compute_drift(pred_poses, gt_poses)
        else:
            logger.warning(f"Unknown metric: {metric}")
    
    return results


def print_evaluation_results(results: Dict[str, Dict[str, float]]) -> None:
    """
    Print evaluation results in a formatted way.
    
    Args:
        results: Dictionary with evaluation results
    """
    print("\n===== Trajectory Evaluation Results =====")
    
    if 'ate' in results:
        print("\nAbsolute Trajectory Error (ATE):")
        print(f"  Mean: {results['ate']['mean']:.4f} m")
        print(f"  Median: {results['ate']['median']:.4f} m")
        print(f"  RMSE: {results['ate']['rmse']:.4f} m")
        print(f"  Std: {results['ate']['std']:.4f} m")
        print(f"  Min: {results['ate']['min']:.4f} m")
        print(f"  Max: {results['ate']['max']:.4f} m")
    
    if 'rpe' in results:
        print("\nRelative Pose Error (RPE):")
        print("  Translation:")
        print(f"    Mean: {results['rpe']['translation']['mean']:.4f} m")
        print(f"    Median: {results['rpe']['translation']['median']:.4f} m")
        print(f"    RMSE: {results['rpe']['translation']['rmse']:.4f} m")
        
        print("  Rotation:")
        print(f"    Mean: {results['rpe']['rotation']['mean']:.4f} deg")
        print(f"    Median: {results['rpe']['rotation']['median']:.4f} deg")
        print(f"    RMSE: {results['rpe']['rotation']['rmse']:.4f} deg")
    
    if 'translation_error' in results:
        print("\nTranslation Error:")
        print(f"  Mean: {results['translation_error']['mean']:.4f} m")
        print(f"  Median: {results['translation_error']['median']:.4f} m")
        print(f"  RMSE: {results['translation_error']['rmse']:.4f} m")
    
    if 'rotation_error' in results:
        print("\nRotation Error:")
        print(f"  Mean: {results['rotation_error']['mean']:.4f} deg")
        print(f"  Median: {results['rotation_error']['median']:.4f} deg")
        print(f"  RMSE: {results['rotation_error']['rmse']:.4f} deg")
    
    if 'drift' in results:
        print("\nDrift:")
        print(f"  Translation Drift: {results['drift']['translation_drift']:.4f} %")
        print(f"  Rotation Drift: {results['drift']['rotation_drift']:.4f} deg/m")
        print(f"  Final Translation Error: {results['drift']['final_translation_error']:.4f} m")
        print(f"  Final Rotation Error: {results['drift']['final_rotation_error']:.4f} deg")
        print(f"  Trajectory Length: {results['drift']['trajectory_length']:.4f} m")
    
    print("\n=========================================")


if __name__ == "__main__":
    # Example usage
    # Create random poses for testing
    n_poses = 100
    
    # Ground truth poses
    gt_poses = np.zeros((n_poses, 7))
    gt_poses[:, 0] = 1.0  # qw = 1 (identity rotation)
    
    # Add some trajectory (a circle in the xy-plane)
    for i in range(n_poses):
        angle = 2 * np.pi * i / n_poses
        gt_poses[i, 4] = np.cos(angle)  # x
        gt_poses[i, 5] = np.sin(angle)  # y
        gt_poses[i, 6] = 0.01 * i       # z (small upward motion)
    
    # Predicted poses with some noise
    pred_poses = gt_poses.copy()
    
    # Add noise to translation
    translation_noise = np.random.normal(0, 0.05, (n_poses, 3))
    pred_poses[:, 4:] += translation_noise
    
    # Add noise to rotation
    rotation_noise = np.random.normal(0, 0.02, (n_poses, 4))
    pred_poses[:, :4] += rotation_noise
    
    # Normalize quaternions
    for i in range(n_poses):
        pred_poses[i, :4] = pred_poses[i, :4] / np.linalg.norm(pred_poses[i, :4])
    
    # Evaluate trajectory
    results = evaluate_trajectory(pred_poses, gt_poses)
    
    # Print results
    print_evaluation_results(results) 